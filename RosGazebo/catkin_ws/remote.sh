
#!/bin/bash

# Set ROS environment variable
export ROS_IP=100.66.238.50
export ROS_MASTER_URI=http://100.119.77.43:11311
exec bash --init-file <(echo "source devel/setup.bash")
