#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Vector3Stamped, TransformStamped
from std_msgs.msg import Float32
from qcar_control.msg import TrajectoryMessage  # Adjust if necessary

import threading
import keyboard  # Make sure to install the 'keyboard' package
import csv
import datetime
import os

class ManualControlDataLogger:
    def __init__(self):
        rospy.init_node('manual_control_data_logger', anonymous=True)

        # Publishers
        self.cmd_pub = rospy.Publisher('qcar/user_command', Vector3Stamped, queue_size=10)

        # Subscribers
        rospy.Subscriber('/vicon/qcar/qcar', TransformStamped, self.vicon_callback)
        rospy.Subscriber('/qcar/motorcurrent', Float32, self.motor_current_callback)
        rospy.Subscriber('/qcar/battery_voltage', Float32, self.battery_voltage_callback)
        rospy.Subscriber('/qcar/velocity', Vector3Stamped, self.velocity_callback)

        # Control parameters
        self.pwm = 0.0  # Initial PWM command (0.0 to 1.0)
        self.steering_angle = 0.0  # Initial steering angle (radians)

        # Limits
        self.pwm_max = 1.0
        self.pwm_min = 0.0
        self.steering_max = 0.5235987756  # 30 degrees in radians
        self.steering_min = -0.5235987756  # -30 degrees in radians

        # Increment steps
        self.pwm_step = 0.05
        self.steering_step = 0.05  # radians (~2.86 degrees)

        # Data storage
        self.data_lock = threading.Lock()
        self.data = {
            'timestamp': [],
            'north': [],
            'east': [],
            'psi': [],
            'motor_current': [],
            'pwm': [],
            'velocity': []
        }

        # State variables
        self.north = 0.0
        self.east = 0.0
        self.psi = 0.0
        self.motor_current = 0.0
        self.velocity = 0.0
        self.battery_voltage = 11.0  # Default value

        # CSV file setup
        self.setup_csv()

        # Start keyboard listening in a separate thread
        self.listener_thread = threading.Thread(target=self.keyboard_listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()

        rospy.on_shutdown(self.shutdown_hook)

        rospy.loginfo("Manual Control Data Logger Node Initialized.")
        rospy.loginfo("Use arrow keys to control the car: Up=Forward, Down=Brake, Left=Left, Right=Right.")
        rospy.loginfo("Press 'Esc' to exit.")

    def setup_csv(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_path = os.path.expanduser("~/qcar_data_logs")
        os.makedirs(folder_path, exist_ok=True)
        self.csv_filename = os.path.join(folder_path, f"data_log_{timestamp}.csv")

        self.csv_file = open(self.csv_filename, mode='w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.data.keys())
        self.csv_writer.writeheader()

        rospy.loginfo(f"Data will be logged to {self.csv_filename}")

    def vicon_callback(self, msg):
        with self.data_lock:
            self.north = msg.transform.translation.x
            self.east = msg.transform.translation.y
            self.psi = self.quat_to_yaw(msg.transform.rotation)

    def motor_current_callback(self, msg):
        with self.data_lock:
            self.motor_current = msg.data

    def battery_voltage_callback(self, msg):
        with self.data_lock:
            self.battery_voltage = msg.data

    def velocity_callback(self, msg):
        with self.data_lock:
            self.velocity = msg.vector.x  # Assuming forward velocity is in x

    def quat_to_yaw(self, q):
        """
        Convert quaternion to yaw angle (psi).
        """
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return rospy.get_param('~yaw_conversion', rospy.get_param('~psi_conversion', 0.0)) if (sindy_cosp == 0 and cosy_cosp ==0 ) else rospy.get_param('~yaw_conversion', rospy.get_param('~psi_conversion', 0.0))

    def keyboard_listener(self):
        while not rospy.is_shutdown():
            try:
                if keyboard.is_pressed('up'):
                    self.increase_pwm()
                if keyboard.is_pressed('down'):
                    self.decrease_pwm()
                if keyboard.is_pressed('left'):
                    self.steer_left()
                if keyboard.is_pressed('right'):
                    self.steer_right()
                if keyboard.is_pressed('esc'):
                    rospy.signal_shutdown("Escape key pressed.")
                rospy.sleep(0.1)  # Small delay to prevent high CPU usage
            except:
                pass

    def increase_pwm(self):
        if self.pwm < self.pwm_max:
            self.pwm += self.pwm_step
            self.pwm = min(self.pwm, self.pwm_max)
            self.publish_command()
            rospy.loginfo(f"Increased PWM to {self.pwm:.2f}")

    def decrease_pwm(self):
        if self.pwm > self.pwm_min:
            self.pwm -= self.pwm_step
            self.pwm = max(self.pwm, self.pwm_min)
            self.publish_command()
            rospy.loginfo(f"Decreased PWM to {self.pwm:.2f}")

    def steer_left(self):
        if self.steering_angle < self.steering_max:
            self.steering_angle += self.steering_step
            self.steering_angle = min(self.steering_angle, self.steering_max)
            self.publish_command()
            rospy.loginfo(f"Steered left to {self.steering_angle:.2f} radians")

    def steer_right(self):
        if self.steering_angle > self.steering_min:
            self.steering_angle -= self.steering_step
            self.steering_angle = max(self.steering_angle, self.steering_min)
            self.publish_command()
            rospy.loginfo(f"Steered right to {self.steering_angle:.2f} radians")

    def publish_command(self):
        command_msg = Vector3Stamped()
        command_msg.vector.x = self.pwm  # PWM command (0.0 to 1.0)
        command_msg.vector.y = self.steering_angle  # Steering angle in radians
        self.cmd_pub.publish(command_msg)

    def log_data(self):
        with self.data_lock:
            entry = {
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                'north': self.north,
                'east': self.east,
                'psi': self.psi,
                'motor_current': self.motor_current,
                'pwm': self.pwm,
                'velocity': self.velocity
            }
            self.csv_writer.writerow(entry)
            self.csv_file.flush()

    def shutdown_hook(self):
        rospy.loginfo("Shutting down. Sending zero commands to stop the car.")
        # Send zero PWM and steering commands
        command_msg = Vector3Stamped()
        command_msg.vector.x = 0.0
        command_msg.vector.y = 0.0
        self.cmd_pub.publish(command_msg)
        rospy.sleep(1)  # Give time for the command to be sent

        # Close CSV file
        self.csv_file.close()
        rospy.loginfo(f"Data logging completed. File saved as {self.csv_filename}")

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.log_data()
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = ManualControlDataLogger()
        controller.run()
    except rospy.ROSInterruptException:
        pass
