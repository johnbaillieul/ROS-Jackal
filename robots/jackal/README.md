# Setup & Startup Guide for BU's Jackal Robots
## Setup


## Startup


# Apriltag_gazebo 
The repository include the code that runs the jackal between two apriltags. Using the realsense camera the apriltag_ros package the robot detects an apriltag and moves towards it. Once the jackal is close enough it starts spinning in its place to detect teh other april tag and moves towards it. The node responsible for the following is detect.py.

The repo also includes a template folder that can be used to add new apriltags and a script that generates apriltag models is also included.

## Prerequisites
apriltag package from https://github.com/AprilRobotics/apriltag
apriltag_ros package from https://github.com/AprilRobotics/apriltag_ros
Realsense_camera see instruction from https://wiki.bu.edu/robotics/index.php?title=Jackal or https://www.clearpathrobotics.com/assets/guides/kinetic/jackal/additional_sim_worlds.html
Clearpath package to simulate Jackal UGV that can be installed by running: sudo apt-get install ros-melosic-jackal-simulator ros-melodic-jackal-desktop ros-melodic-jackal-navigation


## How to Use Repo:
clone the repo in you ~/catkin_ws/src
add the following to your ~/.bashrc file: export GAZEBO_MODEL_PATH=~/catkin_ws/src/apriltag_gazebo/models or if you already have GAZEBO_MODEL_PATH in you file and the path ~/catkin_ws/src/apriltag_gazebo/models.
 
Run export JACKAL_URDF_EXTRAS=$HOME/Desktop/realsense.urdf.xacro. Mind that "HOME/Desktop/" is where I have the realsense.urdf.xacro file located. The run 
roslaunch apriltag_gazebo jack_trial.launch which open gazebo with the apriltags and the jackal. The lauch file also launches the continuous detection file used to detect april tags.

In another terminal run rqt_image_view and select /tag_detections_image. When apriltag is detected you will view it as follow

![apriltag_detection](https://user-images.githubusercontent.com/98136555/174672373-d72a295f-3395-450c-9431-b8182b44308c.png)

Then run rosrun apriltag_gazebo detec.py to run the node that move the jackal between apritags.


## How to Use generate_apriltag_models.py:
Place the file in the models folder, run the script then enter the apriltag's id and pose. This will automatically generate the model folder named apriltag_id. You need to add the image of the apriltag in the meshes folder and name it apriltag_id. "Note that you should replace id with the apriltag's id".

## Demo


https://user-images.githubusercontent.com/98136555/175075007-c5c22281-5b6c-486d-bbde-5046a4e6a989.mp4






