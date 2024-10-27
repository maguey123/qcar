
#!/bin/bash

# Set ROS environment variable
export ROS_IP=100.66.238.50
export ROS_MASTER_URI=http://localhost:11311
exec bash --init-file <(echo "source devel/setup.bash")
