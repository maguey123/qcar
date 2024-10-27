#! /usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
import rospy
from qcar_guidance.msg import TrajectoryMessage

def read_cone_positions(csv_file):
    blue_cones_x, blue_cones_y = [], []
    yellow_cones_x, yellow_cones_y = [], []
    
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['tag'] == 'blue':
                blue_cones_x.append(float(row['x']))
                blue_cones_y.append(float(row['y']))
            elif row['tag'] == 'yellow':
                yellow_cones_x.append(float(row['x']))
                yellow_cones_y.append(float(row['y']))
    
    return blue_cones_x, blue_cones_y, yellow_cones_x, yellow_cones_y

def midpoint(point1, point2):
    return (point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2

def generate_trajectory(blue_cones_x, blue_cones_y, yellow_cones_x, yellow_cones_y):
    # Ensure all lists have the same length
    min_length = min(len(blue_cones_x), len(yellow_cones_x))
    blue_cones_x = blue_cones_x[:min_length]
    blue_cones_y = blue_cones_y[:min_length]
    yellow_cones_x = yellow_cones_x[:min_length]
    yellow_cones_y = yellow_cones_y[:min_length]

    midpoints = np.empty([2, min_length])
    
    for i in range(min_length):
        x_midpoint, y_midpoint = midpoint([blue_cones_x[i], blue_cones_y[i]], [yellow_cones_x[i], yellow_cones_y[i]])
        midpoints[0, i] = x_midpoint
        midpoints[1, i] = y_midpoint
    
    # Add the first point to the end to create a closed loop
    midpoints = np.column_stack((midpoints, midpoints[:, 0]))
    
    # Generate more points along the trajectory
    midpoints_distance = np.cumsum(np.sqrt(np.sum(np.diff(midpoints, axis=1)**2, axis=0)))
    midpoints_distance_total = midpoints_distance[-1]
    midpoints_alpha = np.linspace(0, 1, int(round(8 * midpoints_distance_total)))
    midpoints_distance = np.insert(midpoints_distance, 0, 0)
    midpoints_distance = midpoints_distance / midpoints_distance[-1]
    midpoints_interpolator = interp1d(midpoints_distance, midpoints, kind='quadratic', axis=1)
    midpoints = midpoints_interpolator(midpoints_alpha)
    
    return midpoints

def calculate_velocities(midpoints, max_velocity=1.0, initial_velocity=0.0, max_acceleration=1.0):
    waypoint_times = [0]
    velocities = [initial_velocity]
    current_velocity = initial_velocity

    for i in range(1, len(midpoints[0])):
        waypoint_distance = math.dist([midpoints[0, i-1], midpoints[1, i-1]], [midpoints[0, i], midpoints[1, i]])
        
        if i <= 10:
            time_to_reach_max = (max_velocity - current_velocity) / max_acceleration
            distance_to_reach_max = current_velocity * time_to_reach_max + 0.5 * max_acceleration * time_to_reach_max**2
            
            if waypoint_distance <= distance_to_reach_max:
                delta_v = math.sqrt(2 * max_acceleration * waypoint_distance)
                new_velocity = min(current_velocity + delta_v, max_velocity)
            else:
                new_velocity = max_velocity

            average_velocity = (current_velocity + new_velocity) / 2
            segment_time = waypoint_distance / average_velocity
            
            waypoint_times.append(waypoint_times[-1] + segment_time)
            velocities.append(new_velocity)
            current_velocity = new_velocity
        else:
            segment_time = waypoint_distance / max_velocity
            waypoint_times.append(waypoint_times[-1] + segment_time)
            velocities.append(max_velocity)

    return waypoint_times, velocities

def main():
    rospy.init_node('csv_trajectory_generator')
    
    csv_file = "./src/qcar_guidance/cone_pos.csv"  # Replace with your CSV file path
    blue_cones_x, blue_cones_y, yellow_cones_x, yellow_cones_y = read_cone_positions(csv_file)
    
    midpoints = generate_trajectory(blue_cones_x, blue_cones_y, yellow_cones_x, yellow_cones_y)
    waypoint_times, velocities = calculate_velocities(midpoints)
    
    trajectory_publisher = rospy.Publisher('/qcar/trajectory_topic', TrajectoryMessage, queue_size=0)
    
    rate = rospy.Rate(0.2)
    while not rospy.is_shutdown():
        trajectory_msg = TrajectoryMessage()
        trajectory_msg.waypoint_times = waypoint_times
        trajectory_msg.waypoint_x = midpoints[0].tolist()
        trajectory_msg.waypoint_y = midpoints[1].tolist()
        trajectory_msg.velocity = velocities
        trajectory_publisher.publish(trajectory_msg)
        rospy.loginfo("Trajectory Published")
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass