#!/usr/bin/env python3

import numpy as np
import casadi as ca
import rospy
from qcar_control.msg import TrajectoryMessage
from std_msgs.msg import Float32, Float64
from geometry_msgs.msg import Vector3Stamped, TransformStamped
from nav_msgs.msg import Odometry

class MPCControllerSimulation:
    def __init__(self):
        rospy.init_node('mpc_controller_simulation')

        
        # Model parameters
        self.a = 0.12960  # distance to front axle from center (m)
        self.b = 0.12960  # distance to rear axle from center (m)
        self.r = 0.033  # wheel radius (m)
        self.R_a = 0.47  # Armature resistance [Ohm]
        self.L_a = 0.5  # Armature inductance [H]
        self.K_t = 0.0027  # Torque constant [N·m/A]
        self.K_b = 0.0027  # Back-EMF constant [V·s/rad]
        self.mass = 3.0  #Qcar weight
        self.J = 0.1  # Wheel moment of inertia [kg·m²]

        # MPC parameters
        self.N = 10  # number of control intervals
        self.dt = 0.1  # time step (s)

        # Weights for the cost function
        self.w_pos = 400.0   # Position tracking weight
        self.w_vel = 1.0    # Velocity tracking weight
        self.w_psi = 100.0    # Heading tracking weight
        self.w_u_motor = 1.0  # Motor control input weight
        self.w_u_steering = 1.0  # Steering control input weight

        # State and input constraints
        self.v_max = 1.0  # Maximum velocity (m/s)
        self.omega_max = np.pi/4  # Maximum angular velocity (rad/s)
        self.a_max = 1.0  # Maximum acceleration (m/s^2)
        self.delta_max = np.pi/6  # Maximum steering angle (rad)
        self.max_voltage = 12.0  # Maximum voltage (V)

        # Initialize state and waypoints
        self.current_state = np.zeros(5)  # [x, y, theta, v, current]
        self.waypoint_times = []
        self.waypoint_x = []
        self.waypoint_y = []
        self.waypoint_velocity = []
        self.current_X_waypoint = 0.0
        self.current_Y_waypoint = 0.0
        self.initial_waypoints_received_time = None
        self.waypoints_received = False
        self.received_velocity = 0.0
        self.latest_motor_current = 0.0

        # Solver statistics
        self.solve_attempts = 0
        self.successful_solves = 0
        self.last_successful_solve_time = None

        # Setup ROS subscribers and publishers
        self.init_subscribers()
        self.init_publishers()

        # Setup MPC problem
        self.setup_mpc()

    def init_subscribers(self):
        rospy.Subscriber("/qcar/trajectory_topic", TrajectoryMessage, self.guid_callback)
        rospy.Subscriber('/qcar/velocity', Vector3Stamped, self.velocity_callback)
        rospy.Subscriber("/vicon/qcar/qcar", TransformStamped, self.vicon_callback)
        rospy.Subscriber('/qcar/motorcurrent', Float32, self.motor_current_callback)

    def init_publishers(self):
        self.pub_cmd_rl = rospy.Publisher('/wheelrl_motor/command', Float32, queue_size=1)
        self.pub_cmd_rr = rospy.Publisher('/wheelrr_motor/command', Float32, queue_size=1)
        self.pub_cmd_fl = rospy.Publisher('/wheelfl_motor/command', Float32, queue_size=1)
        self.pub_cmd_fr = rospy.Publisher('/wheelfr_motor/command', Float32, queue_size=1)
        self.pub_cmd_fls = rospy.Publisher('qcar/base_fl_controller/command', Float64, queue_size=1)
        self.pub_cmd_frs = rospy.Publisher('qcar/base_fr_controller/command', Float64, queue_size=1)
        self.cmd_pub = rospy.Publisher('qcar/user_command', Vector3Stamped, queue_size=1)

    def guid_callback(self, msg):
        if not self.waypoints_received:
            self.initial_waypoints_received_time = rospy.Time.now()
            self.waypoints_received = True
            rospy.loginfo("Initial waypoints received and time stored.")

        self.waypoint_times = msg.waypoint_times
        self.waypoint_x = msg.waypoint_x
        self.waypoint_y = msg.waypoint_y
        self.waypoint_velocity = msg.velocity
        rospy.loginfo(f"Updated waypoints: {len(self.waypoint_x)} points")

    def velocity_callback(self, msg):
        self.received_velocity = msg.vector.x
        self.current_state[3] = self.received_velocity
        rospy.logdebug(f"Received velocity: {self.received_velocity} m/s")

    def vicon_callback(self, msg):
        self.current_state[0] = msg.transform.translation.x
        self.current_state[1] = msg.transform.translation.y
        self.current_state[2] = self.quat_to_yaw(msg.transform.rotation)
        rospy.logdebug(f"Updated position from Vicon: x={self.current_state[0]:.2f}, y={self.current_state[1]:.2f}, theta={self.current_state[2]:.2f}")

    def motor_current_callback(self, msg):
        self.latest_motor_current = msg.data
        self.current_state[4] = self.latest_motor_current
        rospy.logdebug(f"Received motor current: {self.latest_motor_current} A")

    def quat_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def command(self, voltage, steering_angle):
        wheel_speed = voltage / (2 * np.pi * self.r)
        
        vel_cmd_right = Float32(wheel_speed)
        vel_cmd_left = Float32(-wheel_speed)
        ang_cmd = Float64(steering_angle)
        
        self.pub_cmd_rl.publish(vel_cmd_left)
        self.pub_cmd_rr.publish(vel_cmd_right)
        self.pub_cmd_fl.publish(vel_cmd_left)
        self.pub_cmd_fr.publish(vel_cmd_right)
        self.pub_cmd_fls.publish(ang_cmd)
        self.pub_cmd_frs.publish(ang_cmd)
        
        command_msg = Vector3Stamped()
        command_msg.vector.x = (voltage/24.0) * 0.5  # Normalize PWM to 0-0.5 range
        command_msg.vector.y = steering_angle
        self.cmd_pub.publish(command_msg)

    def get_current_waypoint(self):
        if not self.waypoints_received:
            rospy.logwarn("No waypoints received yet")
            return None, None, None

        current_time = rospy.Time.now()
        elapsed_time = (current_time - self.initial_waypoints_received_time).to_sec()

        if not self.waypoint_times:
            rospy.logwarn("Waypoint times list is empty")
            return None, None, None

        for i, wp_time in enumerate(self.waypoint_times):
            if wp_time > elapsed_time:
                return self.waypoint_x[i], self.waypoint_y[i], self.waypoint_velocity[i]

        # If we've passed all waypoints, return the last one
        return self.waypoint_x[-1], self.waypoint_y[-1], self.waypoint_velocity[-1]

    def angle_difference(self, angle1, angle2):
        return ca.atan2(ca.sin(angle2 - angle1), ca.cos(angle2 - angle1))

    def setup_mpc(self):
        opti = ca.Opti()

        # State variables
        x = opti.variable(5, self.N+1)  # [x, y, theta, v, current]

        # Control variables
        u = opti.variable(2, self.N)  # [V, delta]

        # Parameters for current state and waypoint
        x0 = opti.parameter(5)
        waypoint = opti.parameter(2)
        target_velocity = opti.parameter(1)

        # Objective function
        objective = 0
        for k in range(self.N):
            objective += self.w_pos * ca.sumsqr(x[:2, k] - waypoint)
            objective += self.w_vel * ca.sumsqr(x[3, k] - target_velocity)

            desired_heading = ca.atan2(waypoint[1] - x[1, k], waypoint[0] - x[0, k])
            heading_error = self.angle_difference(x[2, k], desired_heading)
            objective += self.w_psi * ca.sumsqr(heading_error)

            objective += self.w_u_motor * ca.sumsqr(u[0, k])
            objective += self.w_u_steering * ca.sumsqr(u[1, k])

        opti.minimize(objective)

        # Dynamic constraints
        for k in range(self.N):
            beta = ca.atan(ca.tan(u[1, k]) * self.a / (self.a + self.b))

            di_a_dt = (u[0, k] - self.R_a * x[4, k] - self.K_b * x[3, k]) / self.L_a
            T = self.K_t * x[4, k]
            dw_dt = (T - self.K_b * x[3, k] * 3) / self.J

            x_next = x[:, k] + self.dt * ca.vertcat(
                x[3, k] * self.r * ca.cos(x[2, k] + beta),
                x[3, k] * self.r * ca.sin(x[2, k] + beta),
                (x[3, k] * self.r * ca.tan(u[1, k])) / (self.a + self.b),
                dw_dt,
                di_a_dt
            )
            opti.subject_to(x[:, k+1] == x_next)

        # Path constraints
        opti.subject_to(opti.bounded(-self.v_max, x[3, :], self.v_max))
        opti.subject_to(opti.bounded(-self.a_max, u[0, :], self.a_max))
        opti.subject_to(opti.bounded(-self.delta_max, u[1, :], self.delta_max))

        # Voltage constraint
        max_acceleration = self.max_voltage * self.K_t / self.J
        opti.subject_to(opti.bounded(-max_acceleration, u[0, :], max_acceleration))

        # Initial condition
        opti.subject_to(x[:, 0] == x0)

        # Create solver
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 500,
            'ipopt.tol': 1e-4
        }
        opti.solver("ipopt", opts)

        self.opti = opti
        self.x = x
        self.u = u
        self.x0 = x0
        self.waypoint = waypoint
        self.target_velocity = target_velocity

    def solve_mpc(self):
        self.solve_attempts += 1
        try:
            current_waypoint_x, current_waypoint_y, current_velocity = self.get_current_waypoint()
            if current_waypoint_x is None or current_waypoint_y is None or current_velocity is None:
                rospy.logwarn("No valid waypoint or velocity available for MPC solve.")
                return

            self.opti.set_value(self.x0, self.current_state)
            self.opti.set_value(self.waypoint, [current_waypoint_x, current_waypoint_y])
            self.opti.set_value(self.target_velocity, current_velocity)

            sol = self.opti.solve()

            acceleration = sol.value(self.u[0, 0])
            steering_angle = sol.value(self.u[1, 0])

            voltage = acceleration * self.J / self.K_t

            self.command(voltage, steering_angle)

            self.successful_solves += 1
            self.last_successful_solve_time = rospy.Time.now()

            self.log_debug(voltage, steering_angle)

        except Exception as e:
            rospy.logerr(f"MPC solver failed: {str(e)}")

    def log_debug(self, voltage, steering_angle):
        current_waypoint_x, current_waypoint_y, current_velocity = self.get_current_waypoint()
        desired_heading = np.arctan2(current_waypoint_y - self.current_state[1], 
                                     current_waypoint_x - self.current_state[0])
        heading_error = self.angle_difference(self.current_state[2], desired_heading)

        rospy.loginfo("-------- MPC Debug Information --------")
        rospy.loginfo(f"Current Position (Vicon): ({self.current_state[0]:.2f}, {self.current_state[1]:.2f})")
        rospy.loginfo(f"Current Heading (Vicon): {self.current_state[2]:.2f}")
        rospy.loginfo(f"Current Velocity (QCarNode): {self.current_state[3]:.2f}")
        rospy.loginfo(f"Current Motor Current: {self.current_state[4]:.2f} A")
        rospy.loginfo(f"Control Input - Voltage: {voltage:.6f}, Steering Angle: {steering_angle:.6f}")
        rospy.loginfo(f"Current Waypoint: ({current_waypoint_x:.2f}, {current_waypoint_y:.2f})")
        rospy.loginfo(f"Target Velocity: {current_velocity:.2f}")
        rospy.loginfo(f"Distance to waypoint: {np.sqrt((self.current_state[0]-current_waypoint_x)**2 + (self.current_state[1]-current_waypoint_y)**2):.4f}")
        rospy.loginfo(f"Heading error: {heading_error:.4f}")
        rospy.loginfo(f"Solve attempts: {self.solve_attempts}, Successful solves: {self.successful_solves}")
        if self.last_successful_solve_time:
            time_since_last_solve = (rospy.Time.now() - self.last_successful_solve_time).to_sec()
            rospy.loginfo(f"Time since last successful solve: {time_since_last_solve:.2f} seconds")
        rospy.loginfo("----------------------------------------")

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        self.command(0.0, 0.0)
        while not rospy.is_shutdown():
            self.command(0.0, 0.0)
            rate.sleep()


if __name__ == '__main__':
    try:
        controller = MPCControllerSimulation()
        controller.run()
    except rospy.ROSInterruptException:
        pass