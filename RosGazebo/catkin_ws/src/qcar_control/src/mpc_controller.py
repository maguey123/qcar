#!/usr/bin/env python3

import numpy as np
import casadi as ca
import rospy
from qcar_control.msg import TrajectoryMessage
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3Stamped, TransformStamped

import pandas as pd
from shapely.geometry import Polygon, Point

# Set matplotlib backend to 'Agg' before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2  # Import OpenCV for video writing
from PIL import Image

from matplotlib import patches
from matplotlib.transforms import Affine2D

import shutil
import datetime
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # For speedometer

class MPCControllerSimulation:
    def __init__(self):
        rospy.init_node('mpc_controller_simulation')

        # Vehicle and Motor Parameters
        self.a = 0.128  # Distance to front axle from center (m)
        self.b = 0.128  # Distance to rear axle from center (m)
        self.L = self.a + self.b  # Wheelbase length (m)
        self.r_wheel = 0.033  # Wheel radius (m)
        self.mass = 2.846  # Vehicle mass (kg)
        self.J = 0.01  # Wheel moment of inertia (kg·m²)
        self.c_d = 0.77  # Damping coefficient (adjust as needed)

        # Qcar constraints
        self.gear_ratio = 10.49  # Gear ratio

        # Motor constants
        self.R_a = 0.47  # Armature resistance (Ohm)
        self.L_a = 0.7  # Armature inductance (H)
        self.K_t = 0.0023  # Torque constant (N·m/A)
        self.K_b = 0.0023  # Back-EMF constant (V·s/rad)
        self.V_max_default = 11.0  # Default Maximum voltage (V)
        self.V_max = self.V_max_default  # Initialize V_max with default value

        # Maximum current calculation
        self.I_max = self.V_max / self.R_a  # Max current (A)
        self.I_min = -self.I_max  # Min current (A) (allowing for regenerative braking)

        # Maximum torque calculation
        self.tau_max = self.K_t * self.I_max * self.gear_ratio
        self.tau_min = -self.tau_max  # Assuming regenerative braking

        # Steering constraints
        self.delta_max = 0.5  # Maximum steering angle (rad)
        self.delta_min = -self.delta_max

        # Initialize predicted_trajectory
        self.predicted_trajectory = []

        # Time step and horizon
        self.freq = 20
        self.dt = 1 / self.freq  # Time step (s)
        self.N = int(1.0 / self.dt)  # Prediction horizon (number of steps)
        self.horizon_duration = self.N * self.dt  # Total horizon duration (s)

        # Weights for the cost function
        self.w_pos = 1000.0   # Position tracking weight
        self.w_psi = 0.00     # Heading tracking weight
        self.w_v = 100.0      # Velocity tracking weight
        self.w_u_p = 0.0     # Torque  input weight
        self.w_u_d = 10.0    # Steering control input weight
        self.steering_delta = 00.0  # Steering Rate of Change punishment

        # Slew rate limits for steering
        self.delta_rate_limit = (2 * self.delta_max) / 0.16  # rad/s
        self.delta_delta_max = self.delta_rate_limit * self.dt  # rad per time step

        self.steering_cmd_prev = 0.0  # Initialize previous steering command
        self.PWM_cmd_prev = 0.0  # Initialize previous PWM command

        # Define maximum change in PWM per time step
        self.delta_PWM_max_up = 1.0 # Maximum increase in PWM per time step
        self.delta_PWM_max_down = -1.0  # Maximum decrease in PWM per time step

        # Initialize state and waypoints
        self.current_state = np.zeros(4)  # [x, y, psi, v]
        # Initial position will be updated when first Vicon data is received

        self.waypoint_times = []
        self.waypoint_x = []
        self.waypoint_y = []
        self.waypoint_velocity = []
        self.initial_waypoints_received_time = None
        self.waypoints_received = False

        # Initialize motor current
        self.motor_current = 0.0  # Initialize motor current

        # Read and process the track boundaries
        self.read_and_process_track_boundaries()

        # Setup ROS subscribers and publishers
        self.init_subscribers()
        self.init_publishers()

        # Setup MPC problem
        self.setup_mpc()

        # Flags for plotting
        self.initial_position_plotted = False
        self.pause = False

        # Initialize video writer
        self.init_video_writer()

        # List to store vehicle positions for plotting (optional)
        self.trajectory_history = []

        # Current Waypoint (to be highlighted)
        self.current_waypoint = None

        # Predicted Trajectory for Horizon Plotting
        self.predicted_trajectory = []

        # Index of the current waypoint based on time
        self.current_waypoint_idx = 0

    def init_video_writer(self):
        """
        Initialize the video writer using OpenCV.
        """
        # Define the codec and create VideoWriter object
        # 'mp4v' is a commonly supported codec for MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_filename = './src/qcar_control/trajectory_video.mp4'
        self.fps = 10  # Frames per second

        # Define figure size in inches and DPI to calculate pixel dimensions
        self.fig_width = 10
        self.fig_height = 8
        self.dpi = 100  # Dots per inch

        # Calculate pixel dimensions
        self.frame_width = int(self.fig_width * self.dpi)
        self.frame_height = int(self.fig_height * self.dpi)

        # Initialize VideoWriter
        self.video_writer = cv2.VideoWriter(
            self.video_filename,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )

        if not self.video_writer.isOpened():
            rospy.logerr(f"Failed to open video writer with filename {self.video_filename}")
            rospy.signal_shutdown("VideoWriter initialization failed.")

        rospy.loginfo(f"Video writer initialized. Saving video to {self.video_filename}")

    def init_subscribers(self):
        rospy.Subscriber("/qcar/trajectory_topic", TrajectoryMessage, self.guid_callback)
        rospy.Subscriber('/qcar/velocity', Vector3Stamped, self.velocity_callback)
        rospy.Subscriber("/vicon/qcar/qcar", TransformStamped, self.vicon_callback)
        rospy.Subscriber('/qcar/motorcurrent', Float32, self.motor_current_callback)  # Added subscriber
        rospy.Subscriber('/qcar/battery_voltage', Float32, self.battery_voltage_callback)  # Added Subscriber

    def init_publishers(self):
        self.cmd_pub = rospy.Publisher('qcar/user_command', Vector3Stamped, queue_size=1)

    def battery_voltage_callback(self, msg):
        """
        Callback function to handle incoming battery voltage data.
        Updates V_max and dependent variables.
        """
        self.V_max = msg.data - 0.2
        # Update dependent variables based on the new V_max
        self.I_max = self.V_max / self.R_a  # Max current (A)
        self.I_min = -self.I_max  # Min current (A)
        self.tau_max = self.K_t * self.I_max * self.gear_ratio
        self.tau_min = -self.tau_max
        rospy.logdebug(f"Battery voltage updated: V_max={self.V_max:.2f} V")

    def motor_current_callback(self, msg):
        """
        Callback function to handle incoming motor current data.
        """
        self.motor_current = msg.data
        rospy.logdebug(f"Received motor current: {self.motor_current:.4f} A")

    def read_and_process_track_boundaries(self):
        # Read the cones.csv file
        try:
            cones_df = pd.read_csv('./src/qcar_guidance/vicon_data.csv')
        except FileNotFoundError:
            rospy.logerr("vicon_data.csv file not found in './src/qcar_guidance/'. Please check the file path.")
            rospy.signal_shutdown("Missing vicon_data.csv file.")
            return

        # Separate inner (yellow) and outer (blue) cones
        inner_cones = cones_df[cones_df['tag'] == 'yellow']
        outer_cones = cones_df[cones_df['tag'] == 'blue']

        # Get the x and y coordinates
        inner_coords = list(zip(inner_cones['x'], inner_cones['y']))
        outer_coords = list(zip(outer_cones['x'], outer_cones['y']))

        # Ensure the polygons are properly closed
        if inner_coords and inner_coords[0] != inner_coords[-1]:
            inner_coords.append(inner_coords[0])
        if outer_coords and outer_coords[0] != outer_coords[-1]:
            outer_coords.append(outer_coords[0])

        # Create polygons for the inner and outer boundaries
        self.inner_boundary_polygon = Polygon(inner_coords)
        self.outer_boundary_polygon = Polygon(outer_coords)

        # Check if polygons are valid
        if not self.inner_boundary_polygon.is_valid:
            self.inner_boundary_polygon = self.inner_boundary_polygon.buffer(0)
            rospy.logwarn("Fixed invalid inner boundary polygon using buffer.")
        if not self.outer_boundary_polygon.is_valid:
            self.outer_boundary_polygon = self.outer_boundary_polygon.buffer(0)
            rospy.logwarn("Fixed invalid outer boundary polygon using buffer.")

        # Shrink the track boundaries by 0.05 meters
        shrink_distance = -0.05  # Negative value to shrink
        self.shrunken_outer_boundary = self.outer_boundary_polygon.buffer(shrink_distance)
        self.shrunken_inner_boundary = self.inner_boundary_polygon.buffer(-shrink_distance)

        # Compute the track area after shrinking
        self.track_polygon = self.shrunken_outer_boundary.difference(self.shrunken_inner_boundary)

        # Extract boundary constraints
        self.outer_boundary_constraints = self.compute_boundary_constraints(self.shrunken_outer_boundary)
        self.inner_boundary_constraints = self.compute_boundary_constraints(self.shrunken_inner_boundary)

        rospy.loginfo("Track boundaries processed and constraints computed.")

    def compute_boundary_constraints(self, boundary_polygon):
        # Extract the exterior coordinates of the polygon
        coords = list(boundary_polygon.exterior.coords)
        constraints = []
        num_points = len(coords) - 1  # Last point is the same as the first point

        for i in range(num_points):
            p1 = coords[i]
            p2 = coords[i + 1]

            # Compute the line coefficients a*x + b*y = c
            a = p2[1] - p1[1]
            b = p1[0] - p2[0]
            c = a * p1[0] + b * p1[1]

            # Normalize coefficients
            norm = np.hypot(a, b)
            if norm == 0:
                rospy.logwarn(f"Degenerate edge detected between points {p1} and {p2}. Skipping.")
                continue  # Skip degenerate edges
            a /= norm
            b /= norm
            c /= norm

            # Determine the inequality direction
            # We'll ensure that the interior of the polygon satisfies a*x + b*y <= c
            midpoint = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            normal_point = (midpoint[0] + a, midpoint[1] + b)
            point_inside = Point(normal_point).within(boundary_polygon)

            if point_inside:
                # Inequality is a*x + b*y <= c
                constraints.append((a, b, c))
            else:
                # Flip the inequality
                a *= -1
                b *= -1
                c *= -1
                constraints.append((a, b, c))

            # Log the constraint for debugging
            rospy.logdebug(f"Constraint {i}: a={a:.4f}, b={b:.4f}, c={c:.4f}, point_inside={point_inside}")

        return constraints

    def plot_track_and_position(self):
        # Plot the track boundaries
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height), dpi=self.dpi)
        # Plot outer boundary
        x_outer, y_outer = self.outer_boundary_polygon.exterior.xy
        ax.plot(x_outer, y_outer, 'b-', label='Outer Boundary (Original)')
        # Plot inner boundary
        x_inner, y_inner = self.inner_boundary_polygon.exterior.xy
        ax.plot(x_inner, y_inner, 'y-', label='Inner Boundary (Original)')

        # Plot shrunken outer boundary
        x_shrunk_outer, y_shrunk_outer = self.shrunken_outer_boundary.exterior.xy
        ax.plot(x_shrunk_outer, y_shrunk_outer, 'b--', label='Outer Boundary (Shrunken)')

        # Plot shrunken inner boundary
        x_shrunk_inner, y_shrunk_inner = self.shrunken_inner_boundary.exterior.xy
        ax.plot(x_shrunk_inner, y_shrunk_inner, 'y--', label='Inner Boundary (Shrunken)')

        # Plot received trajectory if available
        if self.waypoints_received and len(self.waypoint_x) > 0:
            ax.plot(self.waypoint_x, self.waypoint_y, 'g.-', label='Received Trajectory')  # Updated label

        # Plot historical vehicle trajectory
        if len(self.trajectory_history) > 1:
            traj_x, traj_y = zip(*self.trajectory_history)
            ax.plot(traj_x, traj_y, 'r-', label='Vehicle Trajectory')

        # Plot current position
        ax.plot(self.current_state[0], self.current_state[1], 'ro', label='Current Position')

        # === Updated Direction of Travel Indicator ===
        # Calculate the direction of travel by adding steering angle to heading
        direction_length = 0.2  # Length of the direction arrow
        direction = self.current_state[2] + self.steering_cmd_prev  # psi + delta
        direction_x = direction_length * np.cos(direction)
        direction_y = direction_length * np.sin(direction)
        ax.arrow(self.current_state[0], self.current_state[1],
                direction_x, direction_y,
                head_width=0.05, head_length=0.1, fc='b', ec='b', label='Direction of Travel')
        # ============================================

        # Plot predicted trajectory over the horizon
        if self.predicted_trajectory:
            pred_x, pred_y = zip(*self.predicted_trajectory)
            ax.plot(pred_x, pred_y, 'm--', label='Predicted Trajectory')
            ax.plot(pred_x[-1], pred_y[-1], 'ms', label='End of Prediction')

        # Plot motor current as an annotation
        ax.annotate(f"Motor Current: {self.motor_current:.2f} A",
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))
        
                # Plot steering angle as an annotation
        ax.annotate(f"Steering Angle: {np.degrees(self.steering_cmd_prev):.2f}°",
                    xy=(0.05, 0.80), xycoords='axes fraction',
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

        # Highlight the current waypoint if available
        if self.current_waypoint is not None:
            ax.plot(self.current_waypoint[0], self.current_waypoint[1],
                    marker='*', markersize=15, markeredgecolor='k', markerfacecolor='magenta',
                    label='Current Waypoint')

        # Mark waypoints outside the shrunken track
        for idx, (x_wp, y_wp) in enumerate(zip(self.waypoint_x, self.waypoint_y)):
            wp_point = Point(x_wp, y_wp)
            if not self.track_polygon.contains(wp_point):
                ax.plot(x_wp, y_wp, 'rx')  # Mark as red x
                rospy.logwarn(f"Waypoint {idx} at ({x_wp:.2f}, {y_wp:.2f}) is outside the shrunken track boundaries.")

        # Add Velocity Annotation
        ax.annotate(f"Velocity: {self.current_state[3]:.2f} m/s",
                    xy=(0.05, 0.90), xycoords='axes fraction',
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc="cyan", alpha=0.5))

        # Add Heading Angle Annotation
        heading_deg = np.degrees(self.current_state[2])
        ax.annotate(f"Heading: {heading_deg:.2f}°",
                    xy=(0.05, 0.85), xycoords='axes fraction',
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.5))

        ax.legend()
        ax.set_aspect('equal')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Track Boundaries, Received Trajectory, Current Position, Predicted Trajectory, and Motor Current')

        # === Begin Speedometer Integration ===
        # Create an inset axes for the speedometer
        speedo_ax = inset_axes(ax, width="20%", height="20%", loc='upper right', borderpad=2)

        # Draw the speedometer background
        speedo_ax.set_xlim(-1.1, 1.1)
        speedo_ax.set_ylim(-1.1, 1.1)
        speedo_ax.axis('off')  # Hide the speedometer axes

        # Draw the outer semi-circle
        outer_circle = patches.Wedge((0, 0), 1, 0, 180, facecolor='lightgray', edgecolor='black')
        speedo_ax.add_patch(outer_circle)

        # Draw tick marks and labels
        max_speed_mps = 3.0  # Maximum speed in m/s
        tick_interval = 0.5   # Tick every 0.5 m/s
        for i in range(0, int(max_speed_mps / tick_interval) + 1):
            speed = i * tick_interval  # Current speed value
            angle = (speed / max_speed_mps) * 180  # Map speed to angle (0 to 180 degrees)
            angle_rad = np.radians(angle)
            x_outer_tick = np.cos(angle_rad) * 0.9
            y_outer_tick = np.sin(angle_rad) * 0.9
            x_inner_tick = np.cos(angle_rad) * 0.8
            y_inner_tick = np.sin(angle_rad) * 0.8
            speedo_ax.plot([x_inner_tick, x_outer_tick], [y_inner_tick, y_outer_tick], color='black', linewidth=2)
            # Add labels
            label_x = np.cos(angle_rad) * 1.0
            label_y = np.sin(angle_rad) * 1.0
            speedo_ax.text(label_x, label_y, f"{speed:.1f}", horizontalalignment='center',
                        verticalalignment='center', fontsize=8, fontweight='bold')

        # Draw the needle
        speed_mps = self.current_state[3]  # Current speed in m/s
        speed_mps = np.clip(speed_mps, 0, max_speed_mps)  # Ensure speed is within 0-max_speed_mps

        # Calculate the angle for the needle
        needle_angle = (speed_mps / max_speed_mps) * 180  # 0 m/s -> 0 degrees, max_speed_mps m/s -> 180 degrees
        needle_rad = np.radians(needle_angle)
        needle_length = 0.8
        needle_x = needle_length * np.cos(needle_rad)
        needle_y = needle_length * np.sin(needle_rad)
        speedo_ax.plot([0, needle_x], [0, needle_y], color='red', linewidth=2)

        # Add a center circle
        center_circle = patches.Circle((0, 0), 0.05, facecolor='black')
        speedo_ax.add_patch(center_circle)

        # Add speed text below the speedometer
        speedo_ax.text(0, -0.3, f"{speed_mps:.2f} m/s", horizontalalignment='center',
                    verticalalignment='center', fontsize=10, fontweight='bold')
        # === End Speedometer Integration ===

        # Draw the canvas and convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Write the frame to the video
        self.video_writer.write(img_bgr)

        plt.close(fig)

        rospy.logdebug("Frame written to video.")



    def guid_callback(self, msg):
        """
        Callback function for receiving waypoints.
        Initializes MPC and processes track boundaries upon first reception of waypoints.
        """
        if not self.waypoints_received:
            rospy.loginfo("Initial waypoints received. Processing track boundaries and setting up MPC.")
            # Process track boundaries
            self.read_and_process_track_boundaries()

            # Setup MPC
            self.setup_mpc()

            # Initialize waypoint reception time after MPC is set up
            self.initial_waypoints_received_time = rospy.Time.now()
            self.waypoints_received = True
            self.mpc_initialized = True
            rospy.loginfo("MPC initialized and waypoint reception time recorded.")

            self.waypoint_times = msg.waypoint_times
            self.waypoint_x = msg.waypoint_x
            self.waypoint_y = msg.waypoint_y
            self.waypoint_velocity = msg.velocity
            rospy.loginfo(f"Updated waypoints: {len(self.waypoint_x)} points")

            # Reset current waypoint index
            self.current_waypoint_idx = 0

            # Plot the trajectory each time new waypoints are received
            self.plot_track_and_position()

    def velocity_callback(self, msg):
        self.current_state[3] = msg.vector.x  # Forward velocity in m/s
        rospy.logdebug(f"Received velocity: {self.current_state[3]:.2f} m/s")

    def vicon_callback(self, msg):
        self.current_state[0] = msg.transform.translation.x
        self.current_state[1] = msg.transform.translation.y
        self.current_state[2] = self.quat_to_yaw(msg.transform.rotation)

        self.trajectory_history.append((self.current_state[0], self.current_state[1]))
        rospy.logdebug(f"Updated position from Vicon: x={self.current_state[0]:.2f}, "
                       f"y={self.current_state[1]:.2f}, psi={self.current_state[2]:.2f}")

        if self.waypoints_received and self.initial_waypoints_received_time is not None:
            current_time = rospy.Time.now()
            elapsed_time = (current_time - self.initial_waypoints_received_time).to_sec()

            # Calculate target time for the horizon
            target_time = elapsed_time

            # Find the index of the first waypoint with waypoint_time >= target_time
            indices = np.where(np.array(self.waypoint_times) >= target_time)[0]
            if len(indices) > 0:
                self.current_waypoint_idx = indices[0]
                if self.current_waypoint_idx < len(self.waypoint_x):
                    self.current_waypoint = [self.waypoint_x[self.current_waypoint_idx],
                                             self.waypoint_y[self.current_waypoint_idx]]
                else:
                    self.current_waypoint = None

        # Check if current position is inside the shrunken track
        current_point = Point(self.current_state[0], self.current_state[1])
        if not self.track_polygon.contains(current_point):
            rospy.logwarn(f"Current position ({self.current_state[0]:.2f}, {self.current_state[1]:.2f}) is outside the shrunken track boundaries.")
            # Create and populate the command message
            command_msg = Vector3Stamped()
            command_msg.vector.x = 0  # PWM command (normalized)
            command_msg.vector.y = 0

            self.cmd_pub.publish(command_msg)
            self.pause = True
            rospy.logwarn("Vehicle has gone out of track boundaries. Pausing control.")

            # Log the bounds of the track polygon for debugging
            minx, miny, maxx, maxy = self.track_polygon.bounds
            rospy.logdebug(f"Track polygon bounds: minx={minx:.2f}, miny={miny:.2f}, maxx={maxx:.2f}, maxy={maxy:.2f}")
        else:
            rospy.logdebug(f"Current position ({self.current_state[0]:.2f}, {self.current_state[1]:.2f}) is inside the shrunken track boundaries.")

        # Plot initial position after receiving first Vicon data
        if not self.initial_position_plotted:
            self.plot_track_and_position()
            self.initial_position_plotted = True

    def quat_to_yaw(self, q):
        """
        Convert quaternion to yaw angle (psi).
        """
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def setup_mpc(self):
        opti = ca.Opti()

        N = self.N  # Prediction horizon

        # State variables
        x = opti.variable(4, N+1)  # [x, y, psi, v]

        # Control variables
        u = opti.variable(2, N)  # [tau, delta]

        # Parameters for initial state and target
        x0 = opti.parameter(4)
        waypoints = opti.parameter(2, N)  # Waypoints over the horizon
        v_refs = opti.parameter(1, N)     # Reference velocities over the horizon
        x_ref_terminal = opti.parameter(2)  # Terminal waypoint at time N
        v_ref_terminal = opti.parameter(1)  # Terminal velocity reference at time N
        u_prev = opti.parameter()          # Previous steering angle command

        # Define dynamics
        def f(x, u):
            # Unpack state variables
            x_pos = x[0]
            y_pos = x[1]
            psi = x[2]
            v = x[3]

            # Torque at the wheel is control input u[0]
            tau_wheel = u[0]

            # Longitudinal acceleration
            a_long = (tau_wheel / (self.mass * self.r_wheel)) - (self.c_d * v) / self.mass

            return ca.vertcat(
                v * ca.cos(psi),            # dx/dt = v * cos(psi)
                v * ca.sin(psi),            # dy/dt = v * sin(psi)
                v / self.L * ca.tan(u[1]),  # dpsi/dt = v / L * tan(delta)
                a_long                      # dv/dt = longitudinal acceleration
            )

        # Objective function
        J = 0
        for k in range(N):
            # Position error
            J += self.w_pos * ca.sumsqr(x[:2, k] - waypoints[:, k])**2

            # Heading error
            desired_heading = ca.atan2(waypoints[1, k] - x[1, k],
                                       waypoints[0, k] - x[0, k])
            heading_error = ca.atan2(ca.sin(desired_heading - x[2, k]),
                                     ca.cos(desired_heading - x[2, k]))
            J += self.w_psi * heading_error**2

            # Velocity error
            J += self.w_v * (x[3, k] - v_refs[0, k])**2

            # Control effort
            J += self.w_u_p * (u[0, k])**2  # Torque effort

            # Penalize steering rate change
            J += self.w_u_d * u[1, k]**2

            # Terminal cost
            J += self.steering_delta * (((u[1, k] - self.steering_cmd_prev)**2))

        opti.minimize(J)

        # Dynamics constraints
        for k in range(N):
            x_next = x[:, k] + self.dt * f(x[:, k], u[:, k])
            opti.subject_to(x[:, k+1] == x_next)

        # Input constraints
        opti.subject_to(opti.bounded(self.tau_min, u[0, :], self.tau_max))  # Torque limits
        opti.subject_to(opti.bounded(self.delta_min, u[1, :], self.delta_max))  # Steering limits

        # State constraints
        opti.subject_to(opti.bounded(0.0, x[3, :], 3.0))  # Velocity limits

        # Slew rate constraints for steering angle
        opti.subject_to(opti.bounded(-self.delta_delta_max,
                                     u[1, 0] - u_prev, self.delta_delta_max))
        for k in range(1, N):
            opti.subject_to(opti.bounded(-self.delta_delta_max,
                                         u[1, k] - u[1, k-1], self.delta_delta_max))

        # Initial condition
        opti.subject_to(x[:, 0] == x0)

        # Solver options
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        opti.solver('ipopt', opts)

        self.opti = opti
        self.x = x
        self.u = u
        self.x0 = x0
        self.waypoints = waypoints
        self.v_refs = v_refs
        self.u_prev = u_prev
        self.x_ref_terminal = x_ref_terminal
        self.v_ref_terminal = v_ref_terminal
    def get_waypoints_over_horizon(self):
        rospy.logdebug("Entering get_waypoints_over_horizon method")
        
        if not self.waypoints_received:
            rospy.logwarn("No waypoints received yet")
            return None, None, None, None

        current_time = rospy.Time.now()
        elapsed_time = (current_time - self.initial_waypoints_received_time).to_sec()

        if not self.waypoint_times:
            rospy.logwarn("Waypoint times list is empty")
            return None, None, None, None

        horizon_times = [elapsed_time + k * self.dt for k in range(self.N + 1)]

        waypoints_x = []
        waypoints_y = []
        v_refs = []

        for t in horizon_times:
            idx = np.searchsorted(self.waypoint_times, t)

            idx += 1

            if idx >= len(self.waypoint_times):
                idx = len(self.waypoint_times) - 1
            waypoints_x.append(self.waypoint_x[idx])
            waypoints_y.append(self.waypoint_y[idx])
            v_refs.append(self.waypoint_velocity[idx])

        x_ref_terminal = np.array([waypoints_x[-1], waypoints_y[-1]])
        v_ref_terminal = np.array([v_refs[-1]])

        waypoints = np.vstack((waypoints_x[:-1], waypoints_y[:-1]))
        v_refs = np.array([v_refs[:-1]])

        rospy.logdebug(f"Returning: waypoints shape: {waypoints.shape}, v_refs shape: {v_refs.shape}, "
                    f"x_ref_terminal shape: {x_ref_terminal.shape}, v_ref_terminal shape: {v_ref_terminal.shape}")
        
        return waypoints, v_refs, x_ref_terminal, v_ref_terminal
    
    def solve_mpc(self):
        try:
            rospy.logdebug("Entering solve_mpc method")
            result = self.get_waypoints_over_horizon()
            rospy.logdebug(f"get_waypoints_over_horizon returned {len(result)} values")

            if len(result) != 4:
                rospy.logerr(f"Unexpected number of return values from get_waypoints_over_horizon: {len(result)}")
                return

            waypoints, v_refs, x_ref_terminal, v_ref_terminal = result

            if waypoints is None or v_refs is None or x_ref_terminal is None or v_ref_terminal is None:
                rospy.logwarn("No valid waypoints or velocities available for MPC solve.")
                return

            rospy.logdebug(f"Setting MPC parameters: waypoints shape: {waypoints.shape}, "
                           f"v_refs shape: {v_refs.shape}, x_ref_terminal shape: {x_ref_terminal.shape}, "
                           f"v_ref_terminal shape: {v_ref_terminal.shape}")

            # Set the initial state
            self.opti.set_value(self.x0, self.current_state)

            self.opti.set_value(self.waypoints, waypoints)
            self.opti.set_value(self.v_refs, v_refs)
            self.opti.set_value(self.x_ref_terminal, x_ref_terminal)
            self.opti.set_value(self.v_ref_terminal, v_ref_terminal)
            self.opti.set_value(self.u_prev, self.steering_cmd_prev)

            # Set initial guesses
            self.opti.set_initial(self.u, np.zeros((2, self.N)))
            self.opti.set_initial(self.x, np.tile(self.current_state.reshape(-1, 1),
                                                  (1, self.N+1)))

            sol = self.opti.solve()

            tau_cmd = sol.value(self.u[0, 0])
            steering_cmd = sol.value(self.u[1, 0])

            self.command(tau_cmd, steering_cmd)

            if self.current_waypoint_idx < len(self.waypoint_x):
                self.current_waypoint = [self.waypoint_x[self.current_waypoint_idx],
                                         self.waypoint_y[self.current_waypoint_idx]]
            else:
                self.current_waypoint = None

            self.predicted_trajectory = sol.value(self.x)[:2, :].T.tolist()

            self.plot_track_and_position()

            self.log_debug(tau_cmd, steering_cmd, waypoints, v_refs)

        except Exception as e:
            rospy.logerr(f"MPC solver failed: {str(e)}")
            import traceback
            rospy.logerr("Exception traceback: " + traceback.format_exc())

    def command(self, tau_cmd, steering_angle):
        # Motor current required for the commanded torque
        I_commanded = tau_cmd / (self.K_t * self.gear_ratio)

        # Motor angular speed
        omega_motor = (self.current_state[3] / self.r_wheel) * self.gear_ratio

        # Applied voltage required
        V_applied_cmd = self.R_a * I_commanded + self.K_b * omega_motor

        # Limit the voltage command
        V_applied_cmd = np.clip(V_applied_cmd, -self.V_max, self.V_max)

        # Normalize voltage to PWM duty cycle (assuming PWM is between 0 and 1)
        PWM_cmd = (V_applied_cmd + self.V_max) / (2 * self.V_max)  # Normalize between 0 and 1

        # Clip PWM to [0, 1]
        PWM_cmd = np.clip(PWM_cmd, 0.0, 1.0)

        # Ensure PWM rate constraints are respected in command
        delta_PWM = PWM_cmd - self.PWM_cmd_prev
        if delta_PWM > self.delta_PWM_max_up:
            delta_PWM = self.delta_PWM_max_up
        elif delta_PWM < self.delta_PWM_max_down:
            delta_PWM = self.delta_PWM_max_down
        PWM_cmd = self.PWM_cmd_prev + delta_PWM

        # Clip again in case of numerical issues
        PWM_cmd = np.clip(PWM_cmd, 0.0, 1.0)

        # Create and populate the command message
        command_msg = Vector3Stamped()
        command_msg.vector.x = PWM_cmd  # PWM command (normalized between 0 and 1)
        command_msg.vector.y = steering_angle

        self.cmd_pub.publish(command_msg)

        # Update previous commands
        self.PWM_cmd_prev = PWM_cmd
        self.steering_cmd_prev = steering_angle

        rospy.loginfo(f"Commanding PWM: {PWM_cmd * 100.0:.2f}%, Steering Angle: "
                      f"{np.degrees(steering_angle):.2f} degrees")

    def angle_difference(self, angle1, angle2):
        """
        Compute the smallest difference between two angles.
        """
        return np.arctan2(np.sin(angle2 - angle1), np.cos(angle2 - angle1))

    def log_debug(self, tau_cmd, steering_cmd, waypoints, v_refs):
        if waypoints.shape[1] == 0:
            rospy.logwarn("No waypoints to log.")
            return

        current_wp_idx = self.current_waypoint_idx
        if current_wp_idx < len(self.waypoint_x):
            current_waypoint_x = self.waypoint_x[current_wp_idx]
            current_waypoint_y = self.waypoint_y[current_wp_idx]
            current_velocity = self.waypoint_velocity[current_wp_idx]
        else:
            # If all waypoints are reached, set to current state
            current_waypoint_x = self.current_state[0]
            current_waypoint_y = self.current_state[1]
            current_velocity = self.current_state[3]

        desired_heading = np.arctan2(current_waypoint_y - self.current_state[1],
                                     current_waypoint_x - self.current_state[0])
        heading_error = self.angle_difference(self.current_state[2], desired_heading)

        # Calculate distance to waypoint
        distance_to_wp = np.sqrt(
            (self.current_state[0] - current_waypoint_x)**2 +
            (self.current_state[1] - current_waypoint_y)**2
        )

        # Calculate PWM command for logging
        I_commanded = tau_cmd / (self.K_t * self.gear_ratio)
        omega_motor = (self.current_state[3] / self.r_wheel) * self.gear_ratio
        V_applied_cmd = self.R_a * I_commanded + self.K_b * omega_motor
        PWM_cmd = (V_applied_cmd + self.V_max) / (2 * self.V_max)
        PWM_cmd = np.clip(PWM_cmd, 0.0, 1.0)

        debug_info = [
            ("Current Position (Vicon)", f"({self.current_state[0]:.2f}, {self.current_state[1]:.2f})"),
            ("Current Heading (Vicon)", f"{np.degrees(self.current_state[2]):.2f} degrees"),
            ("Current Velocity", f"{self.current_state[3]:.2f} m/s"),
            ("Motor Current", f"{self.motor_current:.4f} A"),
            ("Torque Command", f"{tau_cmd:.6f} N·m"),
            ("PWM Command", f"{PWM_cmd * 100.0:.2f}%"),
            ("Steering Angle Command", f"{np.degrees(steering_cmd):.2f} degrees"),
            ("Current Waypoint", f"({current_waypoint_x:.2f}, {current_waypoint_y:.2f})"),
            ("Target Velocity", f"{current_velocity:.2f} m/s"),
            ("Distance to Waypoint", f"{distance_to_wp:.4f} m"),
            ("Heading Error", f"{np.degrees(heading_error):.2f} degrees")
        ]

        rospy.loginfo("-------- MPC Debug Information --------")
        for key, value in debug_info:
            rospy.loginfo(f"{key}: {value}")
        rospy.loginfo("----------------------------------------")

    def run(self):
        rate = rospy.Rate(self.freq)  # 50 Hz

        while not rospy.is_shutdown():
            if not self.pause:
                self.solve_mpc()
            rate.sleep()

        # Upon shutdown, send zero commands and release video writer
        self.command(0.0, 0.0)
        self.video_writer.release()
        rospy.loginfo("Video writer released and node shutdown.")

        # Now, copy the 'qcar_control' folder to the destination with timestamp and tuning parameters
        try:
            # Get current timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Collect tuning values
            w_pos = self.w_pos
            w_psi = self.w_psi
            w_v = self.w_v
            w_u_p = self.w_u_p
            w_u_d = self.w_u_d

            # Calculate the time period of the horizon
            time_period = self.horizon_duration  # in seconds

            # Create folder name
            folder_name = (
                f"{timestamp}_wPos{w_pos}_wPsi{w_psi}_wV{w_v}_"
                f"wUp{w_u_p}_wUd{w_u_d}_"
                f"Horizon{time_period:.2f}s"
            )

            # Define source and destination paths
            source_folder = os.path.dirname(os.path.realpath(__file__))
            # Get the parent directory of 'qcar_control'
            parent_dir = os.path.dirname(source_folder)

            # Define the destination path
            destination_root = "/mnt/c/Users/root/Documents/controlCopy"
            destination_folder = os.path.join(destination_root, folder_name)

            # Copy the entire parent directory (which should be the project root)
            shutil.copytree(parent_dir, destination_folder)

            rospy.loginfo(f"Copied entire project folder to {destination_folder}")

        except Exception as e:
            rospy.logerr(f"Failed to copy 'qcar_control' folder: {str(e)}")
            import traceback
            rospy.logerr("Exception traceback: " + traceback.format_exc())

if __name__ == '__main__':
    try:
        controller = MPCControllerSimulation()
        controller.run()
    except rospy.ROSInterruptException:
        pass
