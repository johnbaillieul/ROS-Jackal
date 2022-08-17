# Setup & Startup Guide for BU's Jackal Robots
## Setup


## Startup

## Prerequisites
Realsense_camera see instruction from https://wiki.bu.edu/robotics/index.php?title=Jackal or https://www.clearpathrobotics.com/assets/guides/kinetic/jackal/additional_sim_worlds.html

Clearpath package to simulate Jackal UGV that can be installed by running: sudo apt-get install ros-melosic-jackal-simulator ros-melodic-jackal-desktop ros-melodic-jackal-navigation

## How to Use Repo:
The repository contains four packages. To run the repo you need to have a catkin worskspace. clone the repo in src file of your catkin workspace
add the following to your ~/.bashrc file: export GAZEBO_MODEL_PATH=<path_to_src_folder>/src/<pacakge name>/models or if you already have GAZEBO_MODEL_PATH in your file add the path.
 

# Package_1: PID_apriltag 
It applies PID control to move the jackal from one apriltag to the other apriltags. Using the realsense camera and the apriltag_ros package the robot detects an apriltag and moves towards it. Once the jackal is close enough it starts spinning in its place to detect another apriltag and moves towards it. 

The repo also includes a template folder that can be used to add new apriltags.

## How to run the package
Run export JACKAL_URDF_EXTRAS=$HOME/Desktop/realsense.urdf.xacro. Mind that "HOME/Desktop/" is where I have the realsense.urdf.xacro file located.
Then run roslaunch pid_apriltag pariltag_jackal.launch which open gazebo with the apriltags and the jackal. The lauch file also launches the continuous detection file used to detect april tags.

## Apriltag detection
To check if the apriltag detection is running open another terminal and run rqt_image_view then select /tag_detections_image. You should get a frame on the tag as you can see in the image below.

![apriltag_detection](https://user-images.githubusercontent.com/98136555/174672373-d72a295f-3395-450c-9431-b8182b44308c.png)

## Note when using apriltags
Be aware that having lighting in your world that is too bright or too dark can cause wrong detections

## Demo
https://user-images.githubusercontent.com/98136555/175075007-c5c22281-5b6c-486d-bbde-5046a4e6a989.mp4

# Package_2: Output_feedback_controller
This package 

# Package_3: Vision_based_navigation_ttt

## How to run the package
For ease of use each environment has its launch file however it is possible to simulate the desired world by specifying your desired world in the launch file at this line: arg name="world_name" value="$(find vision_based_navigation_ttt)/GazeboWorlds/<desired .world file>"/.After chosing the launch file run  roslaunch vision_based_navigation_ttt <your chosen file>.launch. Then run rosrun vision_based_navigation_ttt optical_flow.py, rosrun vision_based_navigation_ttt tau_computation.py and rosrun vision_based_navigation_ttt controller.py in separate terminals.

## Custom worlds 
To test in Gazebo, custom worlds where created the resemble the environment being tested on in the lab. 
 To do add pictures of dif worlds
 
## Performance can be affected by lighting 

https://user-images.githubusercontent.com/98136555/185210652-f371b74c-7054-4f63-95b3-365b9713b741.mp4

## Building up custom gazebo worlds by using wall segments

https://user-images.githubusercontent.com/98136555/185213284-8d2cfa97-f4ec-4a5c-a24f-7408b699c902.mp4

## Demo
 in simulation and real environment
 
# Package_4: Control_Mix 
In this package e combined the optical flow and fiducial markers algorithms together so that the robot can switch to optical-flow-based navigation as a backup option whenever fiducial landmarks are not visible.














