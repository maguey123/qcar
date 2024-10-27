#!/usr/bin/env python3

import rospy
import csv
import os
import argparse
from geometry_msgs.msg import TransformStamped

class ViconDataSaver:
    def __init__(self, tag):
        rospy.init_node('vicon_data_saver', anonymous=True)
        self.tag = tag
        self.last_save_time = rospy.Time.now()
        self.save_interval = rospy.Duration(3.0)  # 5 seconds
        self.latest_x = 0
        self.latest_y = 0
        self.csv_filename = f'vicon_data_{self.tag}.csv'
        self.save_count = 0
        
        # Create or open the CSV file
        file_path = os.path.expanduser("/mnt/c/Users/root/Documents/RosGazebo/catkin_ws/src/ros_example_files/")
        self.csv_filename = os.path.join(file_path, self.csv_filename)
        os.makedirs(os.path.dirname(self.csv_filename), exist_ok=True)
        
        file_exists = os.path.isfile(self.csv_filename)
        self.csv_file = open(self.csv_filename, 'a')  # Open in append mode
        self.csv_writer = csv.writer(self.csv_file)
        if not file_exists:
            self.csv_writer.writerow(['tag', 'x', 'y'])  # Write header if file is new

        # Subscribe to the Vicon topic
        rospy.Subscriber('/vicon/cone/cone', TransformStamped, self.vicon_callback)

    def vicon_callback(self, data):
        self.latest_x = data.transform.translation.x
        self.latest_y = data.transform.translation.y
        
        current_time = rospy.Time.now()
        if (current_time - self.last_save_time) >= self.save_interval:
            self.save_data()
            self.last_save_time = current_time

    def save_data(self):
        self.csv_writer.writerow([self.tag, self.latest_x, self.latest_y])
        self.csv_file.flush()  # Ensure data is written to file
        self.save_count += 1
        rospy.loginfo("SAVED!")
        print("SAVED!")  # Print to screen
        
        if self.save_count % 10 == 0:
            os.system('clear')  # Clear the terminal every 10 saves
            print(f"Cleared terminal. Total saves: {self.save_count}")

    def run(self):
        rospy.spin()

    def shutdown(self):
        self.csv_file.close()
        rospy.loginfo("Vicon data saver node shutdown. CSV file closed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save Vicon data with a specified tag")
    parser.add_argument("tag", help="Tag to be used in the CSV file")
    args = parser.parse_args()

    try:
        saver = ViconDataSaver(args.tag)
        saver.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'saver' in locals():
            saver.shutdown()
