# Setup & Startup Guide for BU's Jackal Robots

## Startup

To power up the Jackal and drive it press the power button on Jackal’s HMI panel. It will take about 30s for the internal PC to finish booting. Then press the PS button on the Sony Bluetooth controller to sync the controller to Jackal. Once the small blue LED on the controller goes solid, it means that its paired.

For slow speed mode, hold the L1 trigger button, and push the left thumbstick forward to drive the Jackal. For full speed mode, hold the R1 trigger.

Reference to the Jackal manual: https://www.generationrobots.com/media/Jackal_Clearpath_Robotics_Userguide.pdf

## Prerequisites
1. Setup Realsense_camera: Follow instructions in https://wiki.bu.edu/robotics/index.php?title=Jackal and https://www.clearpathrobotics.com/assets/guides/kinetic/jackal/additional_sim_worlds.html

    or 

    Equip Jackal with an Intel RealSense D435 camera
    
    Step 1:  install the required packages using the following command
    ``` 
    sudo apt-get install ros-$ROS_DISTRO-realsense2-camera ros-$ROS_DISTRO-realsense2-description ros-$ROS_DISTRO-gazebo-plugins 
    ```

    Step 2: Add the following to your ~/.bashrc 
    ```
    export JACKAL_URDF_EXTRAS=$HOME/PATH_TO_VISION_BASED_NAVIGATION_TTT_FOLDER/vision_based_navigation_ttt/urdf/realsense.urdf.xacro
    ```
  
2. Download Clearpath package using the following command 
     ```
   sudo apt-get install ros-$ROS_Distro-jackal-simulator ros-$ROS_Distro-jackal-desktop ros-$ROS_Distro-jackal-navigation
     ```
3. Download Tensorflow if you are planning on using the CNN to get tau values
    ```
    pip install tensorflow
    
    ```
4. Apriltags Packages
    Follow instructions in https://github.com/AprilRobotics/apriltag_ros.

## Setup:
  1. Create a catkin workspace
  2. Clone the repository in the src folder of your catkin workspace
  3. Add the line below to your ~/.bashrc:
  ```
  export GAZEBO_MODEL_PATH=<path_to_src_folder>/src/<pacakge name>/models
  ```
  
This is what my bashrc file looks like for your reference

![Screenshot from 2022-10-21 11-42-51](https://user-images.githubusercontent.com/98136555/203122995-43e2dc0f-d416-4e50-a8de-1335949a1bbe.png)

  
<!-- ## Package_4: Control_Mix 
  
In this package e combined the optical flow and fiducial markers algorithms together so that the robot can switch to optical-flow-based navigation as a backup option whenever fiducial landmarks are not visible.-->

## Docker 

This Repository can be found in this docker https://hub.docker.com/repository/docker/olagh/jackal_packages_bu/general. Make sure you pull the latest version of it.

run ``` docker pull olagh/jackal_packages_bu:v4 ```
you will need to create a docker-compose.yaml file and paste the following in it:

```
version: '3'

services:
    jackal_bu_packages:
        privileged: true
        image: "olagh/jackal_packages_bu:v4"
        network_mode: host
        environment:
            - ROS_MASTER_URI=$ROS_MASTER_URI
            - ROS_IP=$ROS_IP
            - ROS_DOMAIN_ID=$ROS_DOMAIN_ID
            - DISPLAY=$DISPLAY
        volumes:
            - /tmp/.X11-unix/:/tmp/.X11-unix/
        tty: true
```
and set all of these variables:

- ROS_MASTER_URI
- ROS_IP
- ROS_DOMAIN_ID

In the ~/.bashrc file like this:

```
export ROS_IP=<you ip-address>
export ROS_MASTER_URI=http://$ROS_IP:11311
export ROS_DOMAIN_ID=<can be any number>
```

Then cd to the location of this docker-compose.yaml and run ``` docker compose up ```

in another terminal run ```xhost +``` to be able to view graphics on you screeen then run  ``` docker exec -it jackal-simulation-bu_packages-jackal_bu_packages-1 bash ```

