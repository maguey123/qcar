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
        self.J = 0.05  # Wheel moment of inertia [kg·m²]

        # MPC parameters
        self.N = 20  # number of control intervals
        self.dt = 0.1  # time step (s)

        # Weights for the cost function
        self.w_pos = 100.0   # Position tracking weight
        self.w_vel = 1.0    # Velocity tracking weight
        self.w_psi = 10.0    # Heading tracking weight
        self.w_u_motor = 700.0  # Motor control input weight
        self.w_u_steering = 10.0  # Steering control input weight


        # State and input constraints
        self.v_max = 2.0  # Maximum velocity (m/s)
        self.omega_max = np.pi/4  # Maximum angular velocity (rad/s)
        self.a_max = 2.0  # Maximum acceleration (m/s^2)
        self.delta_max = np.pi/6  # Maximum steering angle (rad)
        self.max_voltage = 12.0  # Maximum voltage (V)

        # Initialize state and waypoints
        self.current_state = np.zeros(5)  # [x, y, theta, v, current]
        self.waypoint_times = []
        self.waypoint_x = []
        self.waypoint_y = []
        self.current_X_waypoint = 0.0
        self.current_Y_waypoint = 0.0
        self.initial_waypoints_received_time = None  # Changed from waypoints_received_time
        self.waypoints_received = False  # New flag to check if waypoints have been received

        # Solver statistics
        self.solve_attempts = 0
        self.successful_solves = 0
        self.last_successful_solve_time = None

        # Setup ROS subscribers and publishers
        self.init_guid_sub()
        self.init_nav_sub()
        self.init_cmd_pub()

        # Setup MPC problem
        self.setup_mpc()

    def init_guid_sub(self):
        self.sub_guid = rospy.Subscriber("/qcar/trajectory_topic", TrajectoryMessage, self.guid_callback)
        rospy.Subscriber('/qcar/velocity', Vector3Stamped, self.velocity_callback)

    def velocity_callback(self, msg):
        self.received_velocity = msg.vector.x

    def guid_callback(self, msg):
        if not self.waypoints_received:
            self.initial_waypoints_received_time = rospy.Time.now()
            self.waypoints_received = True
            rospy.loginfo("Initial waypoints received and time stored.")

        self.waypoint_times = msg.waypoint_times
        self.waypoint_x = msg.waypoint_x
        self.waypoint_y = msg.waypoint_y
        rospy.loginfo(f"Updated waypoints: {len(self.waypoint_x)} points")

    def init_nav_sub(self):
        self.sub_nav = rospy.Subscriber("/odom", Odometry, self.nav_callback)

    def nav_callback(self, msg):
        self.current_state[0] = msg.pose.pose.position.x  # x
        self.current_state[1] = msg.pose.pose.position.y  # y
        self.current_state[2] = self.quat_to_yaw(msg.pose.pose.orientation)  # theta
        self.current_state[3] = np.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)  # v
        # Assuming current is not provided in the odometry message

    def quat_to_yaw(self, q):
        return np.arctan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z))

    def init_cmd_pub(self):
        self.pub_cmd_rl = rospy.Publisher('/wheelrl_motor/command', Float32, queue_size=1)
        self.pub_cmd_rr = rospy.Publisher('/wheelrr_motor/command', Float32, queue_size=1)
        self.pub_cmd_fl = rospy.Publisher('/wheelfl_motor/command', Float32, queue_size=1)
        self.pub_cmd_fr = rospy.Publisher('/wheelfr_motor/command', Float32, queue_size=1)
        self.pub_cmd_fls = rospy.Publisher('qcar/base_fl_controller/command', Float64, queue_size=1)
        self.pub_cmd_frs = rospy.Publisher('qcar/base_fr_controller/command', Float64, queue_size=1)
        self.cmd_pub = rospy.Publisher('qcar/user_command', Vector3Stamped, queue_size=1)



    def command(self, voltage, steering_angle):
        # Convert voltage to angular velocity
        wheel_speed = voltage / (2 * np.pi * self.r)
        
        # Create ROS messages
        vel_cmd_right = Float32(wheel_speed)
        vel_cmd_left = Float32(-wheel_speed)  # Negative for left side
        ang_cmd = Float64(steering_angle)
        

        # Publish commands
        self.pub_cmd_rl.publish(vel_cmd_left)
        self.pub_cmd_rr.publish(vel_cmd_right)
        self.pub_cmd_fl.publish(vel_cmd_left)
        self.pub_cmd_fr.publish(vel_cmd_right)
        self.pub_cmd_fls.publish(ang_cmd)
        self.pub_cmd_frs.publish(ang_cmd)

        # Calculate corresponding omega (0-90 rad/s)
        command_msg = Vector3Stamped()
        command_msg.vector.x = (voltage/12.0)*0.5  # Normalize PWM to 0-0.5 range
        command_msg.vector.y = steering_angle

        self.cmd_pub.publish(command_msg)

    def get_current_waypoint(self):
        if not self.waypoints_received:
            rospy.logwarn("No waypoints received yet")
            return None, None

        current_time = rospy.Time.now()
        elapsed_time = (current_time - self.initial_waypoints_received_time).to_sec()

        if not self.waypoint_times:
            rospy.logwarn("Waypoint times list is empty")
            return None, None

        for i, wp_time in enumerate(self.waypoint_times):
            if wp_time > elapsed_time:
                self.current_X_waypoint = self.waypoint_x[i]
                self.current_Y_waypoint = self.waypoint_y[i]
                return self.current_X_waypoint, self.current_Y_waypoint

        # If we've passed all waypoints, return the last one
        self.current_X_waypoint = self.waypoint_x[-1]
        self.current_Y_waypoint = self.waypoint_y[-1]
        return self.current_X_waypoint, self.current_Y_waypoint

    def angle_difference(self, angle1, angle2):
        """Calculate the smallest angle difference using CasADi operations."""
        return ca.atan2(ca.sin(angle2 - angle1), ca.cos(angle2 - angle1))

    def setup_mpc(self):
        opti = ca.Opti()

        # State variables
        x = opti.variable(4, self.N+1)  # [x, y, theta, v]

        # Control variables
        u = opti.variable(2, self.N)  # [a, delta]

        # Parameters for current state and waypoint
        x0 = opti.parameter(4)
        waypoint = opti.parameter(2)

        # Objective function
        objective = 0
        for k in range(self.N):
            objective += self.w_pos * ca.sumsqr(x[:2,k] - waypoint)
            objective += self.w_vel * ca.sumsqr(x[3,k] - self.v_max)
            
            # Calculate desired heading
            desired_heading = ca.atan2(waypoint[1] - x[1,k], waypoint[0] - x[0,k])
            
            # Use angle_difference in the objective function
            heading_error = self.angle_difference(x[2,k], desired_heading)
            objective += self.w_psi * ca.sumsqr(heading_error)
            
            # Separate weights for motor and steering control inputs
            objective += self.w_u_motor * ca.sumsqr(u[0,k])  # Motor control (acceleration)
            objective += self.w_u_steering * ca.sumsqr(u[1,k])  # Steering control
        
        opti.minimize(objective)

        # Dynamic constraints
        for k in range(self.N):
            x_next = x[:,k] + self.dt * ca.vertcat(
                x[3,k] * ca.cos(x[2,k]),
                x[3,k] * ca.sin(x[2,k]),
                x[3,k] * ca.tan(u[1,k]) / (self.a + self.b),
                u[0,k]
            )
            opti.subject_to(x[:,k+1] == x_next)

        # Path constraints
        opti.subject_to(opti.bounded(-self.v_max, x[3,:], self.v_max))
        opti.subject_to(opti.bounded(-self.a_max, u[0,:], self.a_max))
        opti.subject_to(opti.bounded(-self.delta_max, u[1,:], self.delta_max))

        # Voltage constraint
        max_acceleration = self.max_voltage * self.K_t / self.J
        opti.subject_to(opti.bounded(-max_acceleration, u[0,:], max_acceleration))

        # Initial condition
        opti.subject_to(x[:,0] == x0)

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

    def solve_mpc(self):
        self.solve_attempts += 1
        try:
            current_waypoint_x, current_waypoint_y = self.get_current_waypoint()
            if current_waypoint_x is None or current_waypoint_y is None:
                rospy.logwarn("No valid waypoint available for MPC solve.")
                return

            # Set initial state and waypoint
            self.opti.set_value(self.x0, self.current_state[:4])  # [x, y, theta, v]
            self.opti.set_value(self.waypoint, [current_waypoint_x, current_waypoint_y])

            # Solve the optimization problem
            sol = self.opti.solve()

            # Extract the control inputs
            acceleration = sol.value(self.u[0,0])
            steering_angle = sol.value(self.u[1,0])

            # Convert acceleration to motor voltage (simplified conversion)
            voltage = acceleration * self.J / self.K_t

            self.command(voltage, steering_angle)

            self.successful_solves += 1
            self.last_successful_solve_time = rospy.Time.now()

            # Log debug information
            self.log_debug(voltage, steering_angle)

        except Exception as e:
            rospy.logerr(f"MPC solver failed: {str(e)}")

    def log_debug(self, voltage, steering_angle):
        current_waypoint_x, current_waypoint_y = self.get_current_waypoint()
        desired_heading = np.arctan2(current_waypoint_y - self.current_state[1], 
                                     current_waypoint_x - self.current_state[0])
        heading_error = self.angle_difference(self.current_state[2], desired_heading)

        rospy.loginfo("-------- MPC Debug Information --------")
        rospy.loginfo(f"Current Position: ({self.current_state[0]:.2f}, {self.current_state[1]:.2f})")
        rospy.loginfo(f"Current Heading: {self.current_state[2]:.2f}")
        rospy.loginfo(f"Current Velocity: {self.current_state[3]:.2f}")
        rospy.loginfo(f"Control Input - Voltage: {voltage:.6f}, Steering Angle: {steering_angle:.6f}")
        rospy.loginfo(f"Current Waypoint: ({current_waypoint_x:.2f}, {current_waypoint_y:.2f})")
        rospy.loginfo(f"Distance to waypoint: {np.sqrt((self.current_state[0]-current_waypoint_x)**2 + (self.current_state[1]-current_waypoint_y)**2):.4f}")
        rospy.loginfo(f"Heading error: {heading_error:.4f}")
        rospy.loginfo(f"Solve attempts: {self.solve_attempts}, Successful solves: {self.successful_solves}")
        rospy.loginfo(f"Received Velocity: {self.received_velocity:.2f}")
        if self.last_successful_solve_time:
            time_since_last_solve = (rospy.Time.now() - self.last_successful_solve_time).to_sec()
            rospy.loginfo(f"Time since last successful solve: {time_since_last_solve:.2f} seconds")
        rospy.loginfo("----------------------------------------")

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.solve_mpc()
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = MPCControllerSimulation()
        controller.run()
    except rospy.ROSInterruptException:
        pass