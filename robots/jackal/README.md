# Setup & Startup Guide for BU's Jackal Robots
## Setup


## Startup

## Prerequisites
Realsense_camera see instruction from https://wiki.bu.edu/robotics/index.php?title=Jackal or https://www.clearpathrobotics.com/assets/guides/kinetic/jackal/additional_sim_worlds.html

Clearpath package to simulate Jackal UGV that can be installed by running: sudo apt-get install ros-melosic-jackal-simulator ros-melodic-jackal-desktop ros-melodic-jackal-navigation

## How to Use Repo:
The repository contains four packages. To run the repo you need to have a catkin worskspace. clone the repo in src file of your catkin workspace
add the following to your ~/.bashrc file: export GAZEBO_MODEL_PATH=<path_to_src_folder>/src/<pacakge name>/models or if you already have GAZEBO_MODEL_PATH in your file add the path.
This is what my bashrc file looks like for your reference

![Screenshot from 2022-10-21 11-42-51](https://user-images.githubusercontent.com/98136555/203122995-43e2dc0f-d416-4e50-a8de-1335949a1bbe.png)


# Package_1: PID_apriltag 
It applies PID control to move the jackal from one apriltag to the other apriltags. Using the realsense camera and the apriltag_ros package the robot detects an apriltag and moves towards it. Once the jackal is close enough it starts spinning in its place to detect another apriltag and moves towards it. 

The repo also includes a template folder that can be used to add new apriltags.
x-special/nautilus-clipboard
copy

## How to run the package
The realsense urdf.xacro file is located in project vision_based_navigation. 
You can either keep it there and run "export JACKAL_URDF_EXTRAS=$HOME/catkin_ws/src/vision_based_navigation_ttt/urdf/BU_Jackal.urdf.xacro". Or you can change the location of the file but the new location has to be reflected in the export command.

Mind you that the export command has to be run every time before launching a file. To avoid that, you can add that command to your ~/.bashrc file. 

You can also add that line to your ~/.bashrc file so you dont have to run it everytime you need to work with the realsense camera. 

Then run

  roslaunch pid_apriltag apriltag_jackal.launch 

which opens gazebo with the apriltags and the jackal. The lauch file also launches the continuous detection file used to detect april tags.

## Apriltag detection
To check if the apriltag detection is running open another terminal and run rqt_image_view then select /tag_detections_image. You should get a frame on the tag as you can see in the image below.

![apriltag_detection](https://user-images.githubusercontent.com/98136555/174672373-d72a295f-3395-450c-9431-b8182b44308c.png)

## Note when using apriltags
Be aware that having lighting in your world that is too bright or too dark can cause wrong detections

## Demo
https://user-images.githubusercontent.com/98136555/187280048-6306f278-5905-4383-acf9-1cb138191c13.mp4

# Package_2: Output_feedback_controller
This package 

# Package_3: Vision_based_navigation_ttt

## How to run the package
For ease of use each environment has its launch file however it is possible to simulate the desired world by specifying your desired world in the launch file at this line: arg name="world_name" value="$(find vision_based_navigation_ttt)/GazeboWorlds/<desired .world file>"/.After chosing the launch file run  roslaunch vision_based_navigation_ttt <your chosen file>.launch. Then run rosrun vision_based_navigation_ttt optical_flow.py, rosrun vision_based_navigation_ttt tau_computation.py and rosrun vision_based_navigation_ttt controller.py in separate terminals.

## Custom worlds 
To test in Gazebo, custom worlds where created the resemble the environment being tested on in the lab. 
 To do add pictures of dif worlds
 
## Performance can be affected by lighting 

 # 2 lights
 
https://user-images.githubusercontent.com/98136555/185210652-f371b74c-7054-4f63-95b3-365b9713b741.mp4
 
 # three lights

https://user-images.githubusercontent.com/98136555/185215284-977937bf-bd99-4416-b706-a4d0d101c430.mp4


## Building up custom gazebo worlds by using wall segments

https://user-images.githubusercontent.com/98136555/185213284-8d2cfa97-f4ec-4a5c-a24f-7408b699c902.mp4

## Demo
 In simulation and real environment
 
## Machine learning models
  # The folder called trained_model_parameters contain the parameters of different cnn_models trained.
  model 2 only takes on image as input and outputs an array with the tau values in each Region of Interest
  model 3 takes two images along with the velocity and outputs an array with tau values in each Region of Interest
  model 4 is an updated version of model 3 where the difference is that it uses average pooling instead of max pooling
  model 5 is an updated version of model 4 where the labels are modified. All the previous models were trained with labels corresponding to a infinity tau_value as -1 however in this model that it changed to 15. And results show that it works better than the other models.
  model 6 takes two images along with the velocity and outputs two arrays on with the tau values in each Region of Interest and the other array with a flag that shows if the predicited value is valid or not.
  
  All of the models are not robust enough to work in environments other than the ones with boxes. One thing to be noted is that white walls are of a problem cause the model so the training data sho
 
# Package_4: Control_Mix 
In this package e combined the optical flow and fiducial markers algorithms together so that the robot can switch to optical-flow-based navigation as a backup option whenever fiducial landmarks are not visible.














