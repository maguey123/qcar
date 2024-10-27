import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class QCarNMPCController:
    def __init__(self):
        self.dt = 0.1
        self.N = 30  # Increased prediction horizon
        
        self.wheelbase = 0.26
        self.max_steer = np.radians(30)
        self.max_speed = 2.0
        self.min_speed = 0.0
        self.max_accel = 1.0
        self.max_decel = 1.0
        
        # Adjusted weights
        self.Q = np.diag([100, 100, 10, 1, 1])  # Increased weights for x and y
        self.R = np.diag([10, 1])  # Reduced weights for control inputs
        self.P = np.diag([200, 200, 20, 2, 2])  # Increased terminal weights
        
        self.current_state = np.zeros(5)
        self.trajectory = None
        self.current_index = 0
        
        self.setup_mpc()


    def setup_mpc(self):
        # State variables
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        yaw = ca.SX.sym('yaw')
        v = ca.SX.sym('v')
        delta = ca.SX.sym('delta')  # Add steering angle to state
        states = ca.vertcat(x, y, yaw, v, delta)
        n_states = states.size1()
        
        # Control variables
        a = ca.SX.sym('a')  # acceleration
        delta_dot = ca.SX.sym('delta_dot')  # steering rate
        controls = ca.vertcat(a, delta_dot)
        n_controls = controls.size1()
        
        # Vehicle model (kinematic bicycle model with velocity)
        rhs = ca.vertcat(
            v * ca.cos(yaw),
            v * ca.sin(yaw),
            v * ca.tan(delta) / self.wheelbase,
            a,
            delta_dot
        )
        
        # Function to get next state
        f = ca.Function('f', [states, controls], [rhs])
        
        # Decision variables
        X = ca.SX.sym('X', n_states, self.N + 1)
        U = ca.SX.sym('U', n_controls, self.N)
        P = ca.SX.sym('P', n_states + n_states * self.N)
        
        # Cost function
        obj = 0
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            st_next = X[:, k+1]
            st_ref = P[n_states*(k+1):n_states*(k+2)]
            
            obj += ca.mtimes([(st-st_ref).T, self.Q, (st-st_ref)]) + ca.mtimes([con.T, self.R, con])
        
        # Terminal cost
        obj += ca.mtimes([(X[:, -1]-P[-n_states:]).T, self.P, (X[:, -1]-P[-n_states:])])
        
        # Constraints
        g = []
        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            st_next = X[:, k+1]
            st_next_pred = f(st, con)
            g.append(st_next - st_next_pred)
        
        # NLP problem
        OPT_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        nlp_prob = {'f': obj, 'x': OPT_variables, 'g': ca.vertcat(*g), 'p': P}
        
        # Solver options
        opts = {
            'ipopt.max_iter': 100,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }
        
        # Create solver
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
        
        # Prepare boundaries and initial guess
        self.lbx = ca.DM.zeros((n_states*(self.N+1) + n_controls*self.N, 1))
        self.ubx = ca.DM.zeros((n_states*(self.N+1) + n_controls*self.N, 1))
        
        # State constraints
        self.lbx[0:n_states*(self.N+1):n_states] = -ca.inf  # x lower bound
        self.lbx[1:n_states*(self.N+1):n_states] = -ca.inf  # y lower bound
        self.lbx[2:n_states*(self.N+1):n_states] = -ca.inf  # yaw lower bound
        self.lbx[3:n_states*(self.N+1):n_states] = self.min_speed  # velocity lower bound
        self.lbx[4:n_states*(self.N+1):n_states] = -self.max_steer  # steering angle lower bound
        self.ubx[0:n_states*(self.N+1):n_states] = ca.inf   # x upper bound
        self.ubx[1:n_states*(self.N+1):n_states] = ca.inf   # y upper bound
        self.ubx[2:n_states*(self.N+1):n_states] = ca.inf   # yaw upper bound
        self.ubx[3:n_states*(self.N+1):n_states] = self.max_speed  # velocity upper bound
        self.ubx[4:n_states*(self.N+1):n_states] = self.max_steer  # steering angle upper bound
        
        # Control constraints
        self.lbx[n_states*(self.N+1):] = ca.DM.zeros((n_controls*self.N, 1))
        self.ubx[n_states*(self.N+1):] = ca.DM.zeros((n_controls*self.N, 1))
        self.lbx[n_states*(self.N+1)::n_controls] = -self.max_decel
        self.lbx[n_states*(self.N+1)+1::n_controls] = -self.max_steer / self.dt
        self.ubx[n_states*(self.N+1)::n_controls] = self.max_accel
        self.ubx[n_states*(self.N+1)+1::n_controls] = self.max_steer / self.dt
        
        # Equality constraint
        self.lbg = ca.DM.zeros((n_states*self.N, 1))
        self.ubg = ca.DM.zeros((n_states*self.N, 1))

    def solve_mpc(self, current_time):
        x_ref = self.get_reference_trajectory(current_time)
        
        if x_ref is None:
            print("No reference trajectory available")
            return None, None
        
        x0 = self.current_state
        p = ca.vertcat(x0, x_ref.reshape(-1, 1))
        
        x_init = ca.repmat(x0, self.N+1, 1)
        u_init = ca.DM.zeros((2, self.N))
        x_init = ca.reshape(x_init, -1, 1)
        u_init = ca.reshape(u_init, -1, 1)
        
        sol = self.solver(
            x0=ca.vertcat(x_init, u_init),
            lbx=self.lbx, ubx=self.ubx,
            lbg=self.lbg, ubg=self.ubg,
            p=p
        )
        
        x_opt = sol['x']
        n_states = 5
        n_controls = 2
        u_start = n_states * (self.N + 1)
        u_opt = x_opt[u_start:].reshape((n_controls, -1))
        
        a = float(u_opt[0, 0])
        delta_dot = float(u_opt[1, 0])
        
        v_next = self.current_state[3] + a * self.dt
        delta_next = self.current_state[4] + delta_dot * self.dt
        
        v_next = np.clip(v_next, self.min_speed, self.max_speed)
        delta_next = np.clip(delta_next, -self.max_steer, self.max_steer)
        
        return v_next, delta_next

    def get_reference_trajectory(self, current_time):
        if self.trajectory is None:
            print("No trajectory available")
            return None
        
        self.current_index = np.argmin(np.abs(self.trajectory['times'] - current_time))
        
        x_ref = np.zeros((5, self.N))
        for i in range(self.N):
            next_idx = min(self.current_index + i, len(self.trajectory['times']) - 1)
            x_ref[0, i] = self.trajectory['x'][next_idx]
            x_ref[1, i] = self.trajectory['y'][next_idx]
            if i > 0:
                x_ref[2, i] = np.arctan2(x_ref[1, i] - x_ref[1, i-1], x_ref[0, i] - x_ref[0, i-1])
            else:
                x_ref[2, i] = self.current_state[2]
            x_ref[3, i] = self.trajectory['velocity']
            x_ref[4, i] = 0
        
        return x_ref

    def update_state(self, v, delta):
        x, y, yaw, _, _ = self.current_state
        x += v * np.cos(yaw) * self.dt
        y += v * np.sin(yaw) * self.dt
        yaw += v * np.tan(delta) / self.wheelbase * self.dt
        self.current_state = np.array([x, y, yaw, v, delta])

def generate_trajectory():
    t = np.arange(0, 20, 0.1)  # Increased time range
    x = t
    y = 0.5 * np.sin(0.5 * t)  # More gradual sine wave
    return {
        'times': t,
        'x': x,
        'y': y,
        'velocity': 1.0
    }

def main():
    controller = QCarNMPCController()
    controller.trajectory = generate_trajectory()
    controller.current_state = np.array([0, 0, 0, 0, 0])  # Start at the origin

    simulation_time = 20.0  # Increased simulation time
    num_steps = int(simulation_time / controller.dt)
    
    actual_trajectory = []
    reference_trajectory = []

    fig, ax = plt.subplots(figsize=(12, 6))
    line_actual, = ax.plot([], [], 'b-', label='Actual')
    line_reference, = ax.plot([], [], 'r--', label='Reference')
    ax.set_xlim(0, 20)
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.set_title('NMPC Controller Simulation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)

    def animate(i):
        current_time = i * controller.dt
        v, delta = controller.solve_mpc(current_time)
        if v is not None and delta is not None:
            controller.update_state(v, delta)
            actual_trajectory.append(controller.current_state[:2])
        
        reference_point = controller.get_reference_trajectory(current_time)[:2, 0]
        reference_trajectory.append(reference_point)
        
        actual_x, actual_y = zip(*actual_trajectory)
        reference_x, reference_y = zip(*reference_trajectory)
        
        line_actual.set_data(actual_x, actual_y)
        line_reference.set_data(reference_x, reference_y)
        return line_actual, line_reference

    anim = FuncAnimation(fig, animate, frames=num_steps, interval=50, blit=True)
    plt.show()

if __name__ == '__main__':
    main()