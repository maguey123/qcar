#! /usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
import rospy
from qcar_guidance.msg import TrajectoryMessage
from enum import IntEnum, unique, Enum
from plot import plot_corners, plot_path, plot_trajectory, plot_track_boundary_and_centerline
from track import Track
from trajectory import Trajectory
from vehicle import Vehicle
import os

@unique
class Method(IntEnum):
    CURVATURE = 0
    COMPROMISE = 1
    DIRECT = 2

class vehicle_param(Enum):
    QCAR = {
        "name": "QCAR",
        "mass": 3.0,
        "frictionCoefficient": 0.8,
        "width": 1.0,
        "engineMap": {
            "v": np.array([2.0]),
            "f": np.array([70.0])
        }
    }

METHOD = Method.CURVATURE  # Change this to your desired method

# Corner detection parameters
K_MIN = 0.31
PROXIMITY = 0.2
LENGTH = 0.1

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

def calculate_velocities(midpoints, max_velocity=1.0, initial_velocity=0.0, max_acceleration=0.5):
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

def calculate_heading_angles(x_positions, y_positions):
    heading_angles = []
    for i in range(len(x_positions)):
        if i == len(x_positions) - 1:
            dx = x_positions[0] - x_positions[i]
            dy = y_positions[0] - y_positions[i]
        else:
            dx = x_positions[i+1] - x_positions[i]
            dy = y_positions[i+1] - y_positions[i]
        heading = math.atan2(dy, dx)
        heading_angles.append(heading)
    return heading_angles

def main():
    rospy.init_node('csv_trajectory_generator')
    
    csv_file = "./src/qcar_guidance/vicon_data.csv"  # Replace with your CSV file path
    blue_cones_x, blue_cones_y, yellow_cones_x, yellow_cones_y = read_cone_positions(csv_file)
    left = [yellow_cones_x, yellow_cones_y]
    right=[blue_cones_x, blue_cones_y]
    track = Track(left=left, right=right, car_width=vehicle_param.QCAR.value["width"])
    vehicle = Vehicle(vehicle_param.QCAR)
    trajectory = Trajectory(track, vehicle, initial_velocity=0.0)  # 5.0 m/s as an example

    if METHOD is Method.CURVATURE:
        # print("[ Minimising curvature ]")
        run_time = trajectory.minimise_curvature()

    trajectory.update_velocity()
    lap_time = trajectory.lap_time()

    waypoint_times, x_positions, y_positions, velocities = trajectory.get_trajectory_data()
    
    # Calculate heading angles
    heading_angles = calculate_heading_angles(x_positions, y_positions)
    
    # Output to CSV
    output_dir = "/mnt/c/Users/root/Documents/RosGazebo/catkin_ws"
    output_file = os.path.join(output_dir, "trajectory_waypoints.csv")
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['x', 'y', 'vel', 'time', 'heading']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for x, y, vel, time, heading in zip(x_positions, y_positions, velocities, waypoint_times, heading_angles):
            writer.writerow({'x': x, 'y': y, 'vel': vel, 'time': time, 'heading': heading})
    
    rospy.loginfo(f"Waypoints saved to {output_file}")
    
    trajectory_publisher = rospy.Publisher('/qcar/trajectory_topic', TrajectoryMessage, queue_size=0)
    
    rate = rospy.Rate(0.2)
    while not rospy.is_shutdown():
        trajectory_msg = TrajectoryMessage()
        trajectory_msg.waypoint_times = waypoint_times
        trajectory_msg.waypoint_x = trajectory.path.position(trajectory.s)[0].tolist()
        trajectory_msg.waypoint_y = trajectory.path.position(trajectory.s)[1].tolist()
        trajectory_msg.velocity = velocities
        trajectory_msg.heading = heading_angles
        trajectory_publisher.publish(trajectory_msg)
        rospy.loginfo("Trajectory Published")
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass