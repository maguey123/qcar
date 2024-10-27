#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to print messages
print_message() {
    echo "**************************************************"
    echo "$1"
    echo "**************************************************"
}


# List Windows users
if ! ls /mnt/c/Users/; then
    echo "Error: Unable to access Windows filesystem. Make sure you're running this in WSL."
    exit 1
fi

# Prompt for Windows username
read -p "Enter your Windows username, Displayed Above: " WIN_USERNAME

if [ -z "$WIN_USERNAME" ]; then
    echo "Error: Username cannot be empty."
    exit 1
fi

# Define the path to the Windows Documents folder
DOCUMENTS_PATH="/mnt/c/Users/$WIN_USERNAME/Documents"

# Check if the Windows Documents folder exists
if [ ! -d "$DOCUMENTS_PATH" ]; then
    echo "Error: Documents folder not found for user $WIN_USERNAME."
    echo "Please check the username and ensure you have access to the Windows filesystem."
    exit 1
fi

# Remove existing RosGazebo folder if it exists
if [ -d "$DOCUMENTS_PATH/RosGazebo" ]; then
    print_message "Removing existing RosGazebo folder"
    rm -rf "$DOCUMENTS_PATH/RosGazebo"
fi

# Create RosGazebo folder in Windows Documents
print_message "Creating RosGazebo folder in Windows Documents"
mkdir -p "$DOCUMENTS_PATH/RosGazebo"

# Add ROS repository
print_message "Adding ROS repository"
if ! sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'; then
    echo "Error: Failed to add ROS repository."
    exit 1
fi

# Install curl
print_message "Installing curl"
if ! sudo apt install -y curl; then
    echo "Error: Failed to install curl."
    exit 1
fi

# Add ROS keys
print_message "Adding ROS keys"
if ! curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -; then
    echo "Error: Failed to add ROS keys."
    exit 1
fi

# Update package list
print_message "Updating package list"
if ! sudo apt update; then
    echo "Error: Failed to update package list."
    exit 1
fi

# Install ROS Noetic
print_message "Installing ROS Noetic (full desktop version)"
if ! sudo apt install -y ros-noetic-desktop-full; then
    echo "Error: Failed to install ROS Noetic."
    exit 1
fi

# Install additional tools
print_message "Installing additional tools"
if ! sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential git; then
    echo "Error: Failed to install additional tools."
    exit 1
fi

# Initialize rosdep
print_message "Initializing rosdep"
if ! sudo rosdep init; then
    echo "Warning: rosdep init failed. It might be already initialized."
fi

if ! rosdep update; then
    echo "Error: Failed to update rosdep."
    exit 1
fi

# Source ROS setup file
print_message "Sourcing ROS setup file"
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

print_message "ROS Noetic installation completed!"
echo "RosGazebo folder created at: $DOCUMENTS_PATH/RosGazebo"

# Change to RosGazebo directory
cd "$DOCUMENTS_PATH/RosGazebo" || exit 1

# Create and setup catkin workspace
if [ ! -d "catkin_ws" ]; then
    mkdir catkin_ws
fi
cd catkin_ws || exit 1

# Clone UON-QCAR-BASE repository
if ! git clone https://github.com/jacobdavo/UON-QCAR-BASE.git; then
    echo "Error: Failed to clone UON-QCAR-BASE repository."
    exit 1
fi

cd UON-QCAR-BASE || exit 1
mv src ..
cd ..
rm -rf UON-QCAR-BASE

source /opt/ros/noetic/setup.bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
echo "cd /mnt/c/Users/$WIN_USERNAME/Documents/RosGazebo/catkin_ws" >> ~/.bashrc
echo "source devel/setup.bash" >> ~/.bashrc

# Build the catkin workspace
if ! catkin_make; then
    echo "Error: Failed to build the catkin workspace."
    exit 1
fi

source devel/setup.bash

print_message "Setup completed successfully!"
