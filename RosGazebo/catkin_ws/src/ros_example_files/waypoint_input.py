#!/usr/bin/env python3

import sys
import numpy as np
import rospy
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QTextEdit, QLineEdit, QGridLayout, QScrollArea)
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer
from geometry_msgs.msg import Vector3Stamped, TransformStamped
from qcar_control.msg import TrajectoryMessage

class MPCVisualizerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MPC Controller Visualizer")
        self.setGeometry(100, 100, 1000, 600)

        # Initialize ROS node
        rospy.init_node('mpc_visualizer', anonymous=True)
        
        # Subscribe to necessary topics
        rospy.Subscriber('/vicon/qcar/qcar', TransformStamped, self.vicon_callback)
        rospy.Subscriber('qcar/user_command', Vector3Stamped, self.command_callback)
        
        # Publisher for trajectory
        self.trajectory_pub = rospy.Publisher('/qcar/trajectory_topic', TrajectoryMessage, queue_size=10)

        # Initialize car state and waypoints
        self.car_x = 0
        self.car_y = 0
        self.car_heading = 0
        self.waypoints = []
        self.intercept_times = []
        self.velocities = []
        self.path = []

        # Create main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create canvas for visualization
        self.canvas = QWidget()
        self.canvas.setMinimumSize(500, 500)
        self.canvas.paintEvent = self.paint_event
        main_layout.addWidget(self.canvas)

        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        main_layout.addWidget(control_panel)

        # Create input fields
        input_widget = QWidget()
        input_layout = QGridLayout(input_widget)
        input_layout.addWidget(QLabel("X:"), 0, 0)
        self.x_input = QLineEdit()
        input_layout.addWidget(self.x_input, 0, 1)
        input_layout.addWidget(QLabel("Y:"), 0, 2)
        self.y_input = QLineEdit()
        input_layout.addWidget(self.y_input, 0, 3)
        input_layout.addWidget(QLabel("Time:"), 1, 0)
        self.time_input = QLineEdit()
        input_layout.addWidget(self.time_input, 1, 1)
        input_layout.addWidget(QLabel("Velocity:"), 1, 2)
        self.velocity_input = QLineEdit()
        input_layout.addWidget(self.velocity_input, 1, 3)
        control_layout.addWidget(input_widget)

        # Add waypoint button
        add_waypoint_button = QPushButton("Add Waypoint")
        add_waypoint_button.clicked.connect(self.add_waypoint)
        control_layout.addWidget(add_waypoint_button)

        # Clear waypoints button
        clear_waypoints_button = QPushButton("Clear Waypoints")
        clear_waypoints_button.clicked.connect(self.clear_waypoints)
        control_layout.addWidget(clear_waypoints_button)

        # Send trajectory button
        send_trajectory_button = QPushButton("Send Trajectory")
        send_trajectory_button.clicked.connect(self.send_trajectory)
        control_layout.addWidget(send_trajectory_button)

        # Waypoint list display
        self.waypoint_display = QTextEdit()
        self.waypoint_display.setReadOnly(True)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.waypoint_display)
        scroll_area.setWidgetResizable(True)
        control_layout.addWidget(QLabel("Waypoints:"))
        control_layout.addWidget(scroll_area)

        # Debug information display
        self.debug_display = QTextEdit()
        self.debug_display.setReadOnly(True)
        control_layout.addWidget(QLabel("Debug Information:"))
        control_layout.addWidget(self.debug_display)

        # Timer for updating the GUI
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(100)  # Update every 100 ms

    def paint_event(self, event):
        painter = QPainter(self.canvas)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw waypoints
        painter.setPen(QPen(Qt.blue, 6))
        for waypoint in self.waypoints:
            painter.drawPoint(waypoint[0], waypoint[1])

        # Draw car path
        painter.setPen(QPen(Qt.red, 2))
        for i in range(1, len(self.path)):
            painter.drawLine(self.path[i-1][0], self.path[i-1][1], self.path[i][0], self.path[i][1])

        # Draw car
        painter.setPen(QPen(Qt.black, 2))
        painter.setBrush(QColor(255, 200, 0))
        painter.translate(self.car_x, self.car_y)
        painter.rotate(-np.degrees(self.car_heading))
        painter.drawRect(-10, -5, 20, 10)
        painter.resetTransform()

    def add_waypoint(self):
        try:
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            time = float(self.time_input.text())
            velocity = float(self.velocity_input.text())
            
            self.waypoints.append((x, y))
            self.intercept_times.append(time)
            self.velocities.append(velocity)
            
            self.update_waypoint_display()
            self.canvas.update()
            
            # Clear input fields
            self.x_input.clear()
            self.y_input.clear()
            self.time_input.clear()
            self.velocity_input.clear()
        except ValueError:
            rospy.logwarn("Invalid input. Please enter numeric values for all fields.")

    def update_waypoint_display(self):
        display_text = "X\tY\tTime\tVelocity\n"
        for i, (waypoint, time, velocity) in enumerate(zip(self.waypoints, self.intercept_times, self.velocities)):
            display_text += f"{waypoint[0]:.2f}\t{waypoint[1]:.2f}\t{time:.2f}\t{velocity:.2f}\n"
        self.waypoint_display.setText(display_text)

    def clear_waypoints(self):
        self.waypoints.clear()
        self.intercept_times.clear()
        self.velocities.clear()
        self.update_waypoint_display()
        self.canvas.update()


    def send_trajectory(self):
        msg = TrajectoryMessage()
        msg.waypoint_x = [w[0] for w in self.waypoints]
        msg.waypoint_y = [w[1] for w in self.waypoints]
        msg.waypoint_times = self.intercept_times
        msg.velocity = self.velocities  # Send the entire list of velocities
        self.trajectory_pub.publish(msg)
        rospy.loginfo("Trajectory sent")

    def vicon_callback(self, msg):
        self.car_x = msg.transform.translation.x * 10 + 250  # Scale and offset for visualization
        self.car_y = msg.transform.translation.y * 10 + 250
        
        qx = msg.transform.rotation.x
        qy = msg.transform.rotation.y
        qz = msg.transform.rotation.z
        qw = msg.transform.rotation.w
        self.car_heading = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        self.path.append((self.car_x, self.car_y))
        if len(self.path) > 100:
            self.path.pop(0)

    def command_callback(self, msg):
        voltage = msg.vector.x * 12.0 / 100  # Convert back from PWM to voltage
        steering_angle = msg.vector.y
        debug_text = f"Voltage: {voltage:.2f} V\nSteering Angle: {np.degrees(steering_angle):.2f} deg"
        self.debug_display.setText(debug_text)

    def update_gui(self):
        self.canvas.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MPCVisualizerGUI()
    gui.show()
    sys.exit(app.exec_())