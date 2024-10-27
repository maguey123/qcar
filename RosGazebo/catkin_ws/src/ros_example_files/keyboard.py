#!/usr/bin/env python3

import numpy as np
import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3Stamped, TransformStamped
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import threading
import datetime
import os
import shutil
import pygame

import csv
import time

class QCarDataLogger:
    def __init__(self, filename='qcar_data_log.csv'):
        self.filename = filename
        self.init_csv()

    def init_csv(self):
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Timestamp', 'X Position', 'Y Position', 'Heading', 'Velocity', 
                             'Motor Current', 'Motor Voltage', 'Steering Angle'])

    def log_data(self, x_pos, y_pos, heading, velocity, motor_current, battery_voltage, pwm, steering_angle):
        timestamp = time.time()
        motor_voltage = pwm * battery_voltage
        
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, x_pos, y_pos, heading, velocity, motor_current, 
                             motor_voltage, steering_angle])

class QCarKeyboardController:
    def __init__(self):
        rospy.init_node('qcar_keyboard_controller')

        # Vehicle and Motor Parameters
        self.a = 0.12960
        self.b = 0.12960
        self.L = self.a + self.b
        self.r_wheel = 0.033
        self.mass = 2.75
        self.J = 0.01
        self.c_d = 0.01
        self.gear_ratio = 3.5
        self.data_logger = QCarDataLogger()

        self.trajectory_history = []

        # Motor constants
        self.R_a = 0.47
        self.L_a = 0.5
        self.K_t = 0.0027
        self.K_b = 0.0027
        self.V_max_default = 11.0
        self.V_max = self.V_max_default

        # Maximum torque calculation
        self.I_max = self.V_max / self.R_a
        self.tau_max_motor = self.K_t * self.I_max
        self.tau_max = self.tau_max_motor * self.gear_ratio
        self.tau_min = -self.tau_max

        # Steering constraints
        self.delta_max = 0.5  # Maximum steering angle (rad)
        self.delta_min = -0.5  # Minimum steering angle (rad)

        # Initialize state
        self.current_state = np.zeros(4)  # [x, y, psi, v]

        # Initialize motor current and battery voltage
        self.motor_current = 0.0
        self.battery_voltage = self.V_max_default

        # Setup ROS subscribers and publishers
        self.init_subscribers()
        self.init_publishers()

        # Initialize video writer
        self.init_video_writer()

        # List to store vehicle positions for plotting
        self.trajectory_history = []

        # Control variables
        self.pwm_command = 0.0
        self.steering_angle = 0.0

        # PWM acceleration and deceleration rates
        self.pwm_accel_rate = 0.1  # Increase PWM by this amount per frame when key is held
        self.pwm_decel_rate = 0.1  # Decrease PWM by this amount per frame when key is released

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("QCar Keyboard Controller")
        self.clock = pygame.time.Clock()

        # Keyboard state
        self.key_state = {
            pygame.K_w: False,
            pygame.K_s: False,
            pygame.K_a: False,
            pygame.K_d: False
        }

        # Timeout variables
        self.last_input_time = time.time()
        self.timeout_duration = 3.0  # 3 seconds timeout
        self.is_publishing = True

        # Start the Pygame event loop in a separate thread
        self.running = True
        threading.Thread(target=self.pygame_event_loop).start()

    def init_subscribers(self):
        rospy.Subscriber('/qcar/velocity', Vector3Stamped, self.velocity_callback)
        rospy.Subscriber("/vicon/qcar/qcar", TransformStamped, self.vicon_callback)
        rospy.Subscriber('/qcar/motorcurrent', Float32, self.motor_current_callback)
        rospy.Subscriber('/qcar/battery_voltage', Float32, self.battery_voltage_callback)

    def init_publishers(self):
        self.cmd_pub = rospy.Publisher('qcar/user_command', Vector3Stamped, queue_size=1)

    def battery_voltage_callback(self, msg):
        self.V_max = msg.data - 0.2
        self.I_max = self.V_max / self.R_a
        self.tau_max_motor = self.K_t * self.I_max
        self.tau_max = self.tau_max_motor * self.gear_ratio
        self.tau_min = -self.tau_max

    def motor_current_callback(self, msg):
        self.motor_current = msg.data

    def init_video_writer(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_filename = './trajectory_video.mp4'
        self.fps = 10
        self.fig_width, self.fig_height = 10, 8
        self.dpi = 100
        self.frame_width = int(self.fig_width * self.dpi)
        self.frame_height = int(self.fig_height * self.dpi)
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

    def velocity_callback(self, msg):
        self.current_state[3] = msg.vector.x

    def vicon_callback(self, msg):
        self.current_state[0] = msg.transform.translation.x
        self.current_state[1] = msg.transform.translation.y
        self.current_state[2] = self.quat_to_yaw(msg.transform.rotation)
        self.trajectory_history.append((self.current_state[0], self.current_state[1]))
        self.plot_track_and_position()
        self.data_logger.log_data(
            self.current_state[0],  # x position
            self.current_state[1],  # y position
            self.current_state[2],  # heading
            self.current_state[3],  # velocity
            self.motor_current,
            self.battery_voltage,
            self.pwm_command,
            self.steering_angle
        )

    def quat_to_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def pygame_event_loop(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in self.key_state:
                        self.key_state[event.key] = True
                        self.last_input_time = time.time()
                        self.is_publishing = True
                elif event.type == pygame.KEYUP:
                    if event.key in self.key_state:
                        self.key_state[event.key] = False
                        self.last_input_time = time.time()

            self.update_control()
            self.update_gui()
            self.check_timeout()
            self.clock.tick(30)  # 30 FPS

        pygame.quit()

    def update_control(self):
        # Update PWM
        if self.key_state[pygame.K_w]:
            self.pwm_command = min(1.0, self.pwm_command + self.pwm_accel_rate)
        elif self.key_state[pygame.K_s]:
            self.pwm_command = max(-1.0, self.pwm_command - self.pwm_accel_rate)
        else:
            # Decelerate when no key is pressed
            if self.pwm_command > 0:
                self.pwm_command = max(0, self.pwm_command - self.pwm_decel_rate)
            elif self.pwm_command < 0:
                self.pwm_command = min(0, self.pwm_command + self.pwm_decel_rate)

        # Update steering
        if self.key_state[pygame.K_a]:
            self.steering_angle = min(self.delta_max, self.steering_angle + 0.5)
        elif self.key_state[pygame.K_d]:
            self.steering_angle = max(self.delta_min, self.steering_angle - 0.5)
        else:
            # Center steering when no key is pressed
            if self.steering_angle > 0:
                self.steering_angle = max(0, self.steering_angle - 0.5)
            elif self.steering_angle < 0:
                self.steering_angle = min(0, self.steering_angle + 0.5)

    def update_gui(self):
        self.screen.fill((255, 255, 255))  # White background
        font = pygame.font.Font(None, 36)
        
        pwm_text = font.render(f"PWM: {self.pwm_command:.2f}", True, (0, 0, 0))
        steering_text = font.render(f"Steering: {self.steering_angle:.2f}", True, (0, 0, 0))
        velocity_text = font.render(f"Velocity: {self.current_state[3]:.2f}", True, (0, 0, 0))
        publishing_text = font.render(f"Publishing: {'Yes' if self.is_publishing else 'No'}", True, (0, 0, 0))
        
        self.screen.blit(pwm_text, (10, 10))
        self.screen.blit(steering_text, (10, 50))
        self.screen.blit(velocity_text, (10, 90))
        self.screen.blit(publishing_text, (10, 130))
        
        pygame.display.flip()

    def check_timeout(self):
        if time.time() - self.last_input_time > self.timeout_duration:
            if self.is_publishing:
                self.is_publishing = False
                rospy.loginfo("No input for 3 seconds. Stopping message publishing.")


    def command(self):
        if self.is_publishing:
            # Create and publish command message
            command_msg = Vector3Stamped()
            command_msg.vector.x = self.pwm_command
            command_msg.vector.y = self.steering_angle
            self.cmd_pub.publish(command_msg)
        else:
            # Publish zero commands when not publishing
            command_msg = Vector3Stamped()
            command_msg.vector.x = 1.0
            command_msg.vector.y = 0.0
            self.cmd_pub.publish(command_msg)

    def plot_track_and_position(self):
        fig, ax = plt.subplots(figsize=(self.fig_width, self.fig_height), dpi=self.dpi)

        if len(self.trajectory_history) > 1:
            traj_x, traj_y = zip(*self.trajectory_history)
            ax.plot(traj_x, traj_y, 'r-', label='Vehicle Trajectory')

        ax.plot(self.current_state[0], self.current_state[1], 'ro', label='Current Position')

        ax.annotate(f"Motor Current: {self.motor_current:.2f} A",
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=10, ha='left', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

        ax.legend()
        ax.set_aspect('equal')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Vehicle Trajectory and Motor Current')

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.video_writer.write(img_bgr)
        plt.close(fig)

    def run(self):
        rate = rospy.Rate(20)  # 20 Hz

        while not rospy.is_shutdown() and self.running:
            self.command()
            rate.sleep()

        # Upon shutdown, send zero commands and release video writer
        command_msg = Vector3Stamped()
        command_msg.vector.x = 0.0
        command_msg.vector.y = 0.0
        self.cmd_pub.publish(command_msg)

        self.video_writer.release()
        rospy.loginfo("Video writer released and node shutdown.")

        # Optionally, copy the data to a destination folder
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{timestamp}_KeyboardControl"
            source_folder = os.path.dirname(os.path.realpath(__file__))
            parent_dir = os.path.dirname(source_folder)
            destination_root = "/mnt/c/Users/root/Documents/controlCopy"
            destination_folder = os.path.join(destination_root, folder_name)
            shutil.copytree(parent_dir, destination_folder)
            rospy.loginfo(f"Copied entire project folder to {destination_folder}")
        except Exception as e:
            rospy.logerr(f"Failed to copy project folder: {str(e)}")

if __name__ == '__main__':
    try:
        controller = QCarKeyboardController()
        controller.run()
    except rospy.ROSInterruptException:
        pass