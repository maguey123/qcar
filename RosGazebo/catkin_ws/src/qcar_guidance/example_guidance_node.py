#! /usr/bin/env python3

from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import GetWorldProperties
import rospy
import math
import numpy as np
import matplotlib.pyplot as plt
from qcar_guidance.msg import TrajectoryMessage
from time import sleep
from scipy.interpolate import interp1d

def midpoint(point1,point2):
    x_midpoint = (point1[0] + point2[0])/2
    y_midpoint = (point1[1] + point2[1])/2
    return x_midpoint, y_midpoint

def get_cone_positions():
    try:
        model_names_get = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        cone_state_get = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        model_names = model_names_get()
        model_names_length = len(model_names.model_names)
        cone_position_array = np.empty([model_names_length, 2])
        blue_cones_x = []
        blue_cones_y = []
        yellow_cones_x = []
        yellow_cones_y = []

        complete = 0
        index = 0
        while complete == 0:
            search_item = "blue_cone_{}".format(index)
            if search_item in model_names.model_names:
                current_cone = cone_state_get(search_item,"")
                blue_cones_x.append(current_cone.pose.position.x)
                blue_cones_y.append(current_cone.pose.position.y)
                search_item = "yellow_cone_{}".format(index)
                current_cone = cone_state_get(search_item,"")
                yellow_cones_x.append(current_cone.pose.position.x)
                yellow_cones_y.append(current_cone.pose.position.y)
            else:
                complete = 1
            index += 1

        return blue_cones_x,blue_cones_y, yellow_cones_x, yellow_cones_y

    except rospy.ServiceException as e:
        rospy.loginfo("Get Model State service call failed:  {0}".format(e))

def get_qcar_state():
    try:
        qcar_state_get = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        qcar_state = qcar_state_get("qcar", "")
        
        current_x = qcar_state.pose.position.x
        current_y = qcar_state.pose.position.y
        
        return current_x, current_y
    
    except rospy.ServiceException as e:
        rospy.loginfo("Get Model State service call failed:  {0}".format(e))

if __name__ == '__main__':
    rospy.init_node('example_guidance_node')

    # prevents the node from running whilst ros/gazebo is starting up
    rosrunning = 0
    while rosrunning==0:
        sleep(1)
        if rospy.Time.now() != rospy.Time():
            rosrunning = 1

    trajectory_publisher = rospy.Publisher('/qcar/trajectory_topic', TrajectoryMessage, queue_size = 0)

    blue_cones_x,blue_cones_y, yellow_cones_x, yellow_cones_y = get_cone_positions()
    current_x, current_y = get_qcar_state()

    midpoints = np.empty([2,len(blue_cones_x)+2])
    midpoints[0,0] = current_x
    midpoints[1,0] = current_y
    max_velocity = 0.6
    initial_velocity = 0.0
    max_acceleration = 1.0  # m/s²

    for i in range(0,len(blue_cones_x)):
        x_midpoint, y_midpoint = midpoint([blue_cones_x[i],blue_cones_y[i]], [yellow_cones_x[i],yellow_cones_y[i]])
        midpoints[0,i+1] = x_midpoint
        midpoints[1,i+1] = y_midpoint
        
    midpoints[0,-1] = midpoints[0,0]
    midpoints[1,-1] = midpoints[1,0]
    blue_cones_x.append(blue_cones_x[0])
    blue_cones_y.append(blue_cones_y[0])
    yellow_cones_x.append(yellow_cones_x[0])
    yellow_cones_y.append(yellow_cones_y[0])

    midpoints_distance = (np.cumsum(np.sqrt(np.sum(np.diff(midpoints, axis=1)**2, axis=0))))
    midpoints_distance_total = midpoints_distance[-1]
    midpoints_alpha = np.linspace(0,1,int(round(8*midpoints_distance_total)))
    midpoints_distance = np.insert(midpoints_distance, 0, 0)
    midpoints_distance = midpoints_distance/midpoints_distance[-1]
    midpoints_interpolator = interp1d(midpoints_distance, midpoints, kind = 'quadratic', axis = 1)
    midpoints = midpoints_interpolator(midpoints_alpha)

    waypoint_times = [0]
    velocities = [initial_velocity]
    current_velocity = initial_velocity

    for i in range(1, len(midpoints[0])):
        waypoint_distance = math.dist([midpoints[0,i-1],midpoints[1,i-1]], [midpoints[0,i],midpoints[1,i]])
        
        if i <= 10:
            # Calculate acceleration-limited velocity for first 10 waypoints
            time_to_reach_max = (max_velocity - current_velocity) / max_acceleration
            distance_to_reach_max = current_velocity * time_to_reach_max + 0.5 * max_acceleration * time_to_reach_max**2
            
            if waypoint_distance <= distance_to_reach_max:
                # We don't reach max velocity within this segment
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
            # Use constant max velocity for remaining waypoints
            segment_time = waypoint_distance / max_velocity
            waypoint_times.append(waypoint_times[-1] + segment_time)
            velocities.append(max_velocity)

    rate = rospy.Rate(0.2)
    while not rospy.is_shutdown():
        trajectoryTopic = TrajectoryMessage()
        trajectoryTopic.waypoint_times = waypoint_times
        trajectoryTopic.waypoint_x = midpoints[0].tolist()
        trajectoryTopic.waypoint_y = midpoints[1].tolist()
        trajectoryTopic.velocity = velocities
        trajectory_publisher.publish(trajectoryTopic)
        rospy.loginfo("Trajectory Published")
        rate.sleep()