#!/usr/bin/env python3
"""
System Identification Data Collection Node - 5 Second Test Run
Generates sinusoidal PWM input and collects system response data at 100 Hz.
"""

import rospy
import math
import csv
import os
from datetime import datetime
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32
import numpy as np

class SystemIdentification:
    def __init__(self):
        rospy.init_node('system_identification', anonymous=True)
        
        # Parameters
        self.freq = 1.3  # Sine wave frequency (Hz)
        self.amplitude = 0.5  # PWM amplitude
        self.publish_rate = 15.0  # Publishing rate (Hz)
        self.test_duration = 30.0  # Run for 5 seconds
        
        # Initialize data storage
        self.battery_voltage = None  # Initialize as None to check if we've received data
        self.motor_current = 0.0
        self.velocity = Vector3Stamped().vector
        self.last_battery_update = rospy.Time.now()
        
        # Setup publishers and subscribers
        self.cmd_pub = rospy.Publisher('/qcar/user_command', Vector3Stamped, queue_size=1)
        
        # Setup subscribers with more debug info
        rospy.loginfo("Setting up subscribers...")
        
        self.battery_sub = rospy.Subscriber('/qcar/battery_voltage', Float32, self.battery_voltage_callback)
        rospy.loginfo("Subscribed to /qcar/battery_voltage")
        
        self.current_sub = rospy.Subscriber('/qcar/motorcurrent', Float32, self.motor_current_callback)
        rospy.loginfo("Subscribed to /qcar/motor_current")
        
        self.velocity_sub = rospy.Subscriber('/qcar/velocity', Vector3Stamped, self.velocity_callback)
        rospy.loginfo("Subscribed to /qcar/velocity")

        # Wait for battery voltage reading
        timeout = rospy.Duration(5.0)  # 5 second timeout
        start_wait = rospy.Time.now()
        
        rospy.loginfo("Waiting for initial battery voltage reading...")
        while self.battery_voltage is None and not rospy.is_shutdown():
            if rospy.Time.now() - start_wait > timeout:
                rospy.logwarn("Timeout waiting for battery voltage. Check if topic is publishing:")
                rospy.logwarn("Try running: rostopic echo /battery_voltage")
                rospy.logwarn("Using default voltage of 11.1V")
                self.battery_voltage = 11.1
                break
            rospy.sleep(0.05)  # Increased frequency for faster waiting
            rospy.loginfo_throttle(1, "Waiting for battery voltage...")
        
        # Setup CSV logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_dir = './src/qcar_control/'
        self.csv_filename = f"sysid_data_{timestamp}.csv"
        self.csv_path = os.path.join(self.csv_dir, self.csv_filename)
        
        # Ensure directory exists
        os.makedirs(self.csv_dir, exist_ok=True)
        
        # Open CSV file
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'timestamp', 'elapsed_time', 'pwm_command', 'battery_voltage',
            'motor_current', 'velocity_x'
        ])
        
        rospy.loginfo(f"Battery voltage: {self.battery_voltage:.2f}V")
        rospy.loginfo(f"System identification node initialized. Data will be saved to {self.csv_path}")

        # Initialize data buffer for efficient CSV writing
        self.data_buffer = []
        self.buffer_size = 100  # Adjust buffer size as needed

    def battery_voltage_callback(self, msg):
        """Callback function for battery voltage"""
        self.battery_voltage = msg.data
        self.last_battery_update = rospy.Time.now()
        rospy.logdebug(f"Battery voltage updated: {self.battery_voltage:.2f}V")

    def motor_current_callback(self, msg):
        """Callback function for motor current"""
        self.motor_current = msg.data

    def velocity_callback(self, msg):
        """Callback function for velocity"""
        self.velocity = msg.vector

    def check_battery_voltage(self):
        """Check if we're still receiving battery voltage updates"""
        time_since_update = (rospy.Time.now() - self.last_battery_update).to_sec()
        if time_since_update > 1.0:  # Haven't received update in 1 second
            rospy.logwarn_throttle(1, f"No battery voltage update in {time_since_update:.1f} seconds")
            rospy.logwarn_throttle(1, "Check if battery_voltage topic is publishing")
            return False
        return True

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        start_time = rospy.Time.now()
        last_log_time = start_time

        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            elapsed_time = (current_time - start_time).to_sec()
            
            # Check if we've run for the specified test duration
            if elapsed_time >= self.test_duration:
                rospy.loginfo("5 second test completed. Shutting down...")
                break
            
            # Generate sinusoidal PWM command
            pwm_command = self.amplitude * math.sin(2 * math.pi * self.freq * elapsed_time)
            
            # Create command message
            cmd_msg = Vector3Stamped()
            cmd_msg.header.stamp = current_time
            cmd_msg.vector.x = pwm_command
            cmd_msg.vector.y = 0.0
            cmd_msg.vector.z = 0.0
            
            # Publish command
            self.cmd_pub.publish(cmd_msg)
            
            # Check battery voltage status
            self.check_battery_voltage()
            
            # Buffer data for CSV writing
            self.data_buffer.append([
                rospy.get_time(),
                elapsed_time,
                pwm_command,
                self.battery_voltage,
                self.motor_current,
                self.velocity.x
            ])
            
            # Write to CSV in batches to improve performance
            if len(self.data_buffer) >= self.buffer_size:
                self.csv_writer.writerows(self.data_buffer)
                self.csv_file.flush()
                self.data_buffer = []
            
            # Print status every 0.5 seconds
            if (elapsed_time - (last_log_time - start_time).to_sec()) >= 0.5:
                rospy.loginfo(f"Time: {elapsed_time:.1f}s | PWM: {pwm_command:.3f} | "
                              f"Batt: {self.battery_voltage:.2f}V | Current: {self.motor_current:.3f}A | "
                              f"Velocity: {self.velocity.x:.2f}m/s")
                last_log_time = current_time
            
            rate.sleep()
        
        # Write any remaining data in buffer
        if self.data_buffer:
            self.csv_writer.writerows(self.data_buffer)
            self.csv_file.flush()
            self.data_buffer = []

    def shutdown(self):
        """Cleanup on node shutdown"""
        rospy.loginfo("Shutting down System Identification Node...")
        # Send zero command
        cmd_msg = Vector3Stamped()
        cmd_msg.vector.x = 0.0
        cmd_msg.vector.y = 0.0
        cmd_msg.vector.z = 0.0
        self.cmd_pub.publish(cmd_msg)
        rospy.sleep(0.1)
        
        # Write any remaining data in buffer
        if hasattr(self, 'data_buffer') and self.data_buffer:
            self.csv_writer.writerows(self.data_buffer)
            self.csv_file.flush()
            self.data_buffer = []
        
        # Close CSV file
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
            rospy.loginfo(f"Data saved to {self.csv_path}")

if __name__ == '__main__':
    try:
        sysid = SystemIdentification()
        rospy.on_shutdown(sysid.shutdown)
        sysid.run()
    except rospy.ROSInterruptException:
        pass
