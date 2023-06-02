# USE THE CNN_MODEL BRANCH
## Setup


## Startup

## Prerequisites
Realsense_camera see instruction from https://wiki.bu.edu/robotics/index.php?title=Jackal or https://www.clearpathrobotics.com/assets/guides/kinetic/jackal/additional_sim_worlds.html

Clearpath package to simulate Jackal UGV that can be installed by running: sudo apt-get install ros-melodic-jackal-simulator ros-melodic-jackal-desktop ros-melodic-jackal-navigation

## How to Use Repo:
The repository contains four packages. To run the repo you need to have a catkin worskspace. clone the repo in src file of your catkin workspace
add the following to your ~/.bashrc file: export GAZEBO_MODEL_PATH=<path_to_src_folder>/src/<pacakge name>/models or if you already have GAZEBO_MODEL_PATH in your file add the path.
This is what my bashrc file looks like for your refrence



# Package_1: PID_apriltag 
It applies PID control to move the jackal from one apriltag to the other apriltags. Using the realsense camera and the apriltag_ros package the robot detects an apriltag and moves towards it. Once the jackal is close enough it starts spinning in its place to detect another apriltag and moves towards it. 

The repo also includes a template folder that can be used to add new apriltags.
x-special/nautilus-clipboard
copy

## How to run the package
Create your customized URDF file, for example $HOME/Desktop/realsense.urdf.xacro. Put the following in it:

********************************************************************************************************************************************************
<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro">

  <link name="front_realsense" />

  <!--
    The gazebo plugin aligns the depth data with the Z axis, with X=left and Y=up
    ROS expects the depth data along the X axis, with Y=left and Z=up
    This link only exists to give the gazebo plugin the correctly-oriented frame
  -->
  <link name="front_realsense_gazebo" />
  <joint name="front_realsense_gazebo_joint" type="fixed">
    <parent link="front_realsense"/>
    <child link="front_realsense_gazebo"/>
    <origin xyz="0.0 0 0" rpy="-1.5707963267948966 0 -1.5707963267948966"/>
  </joint>

  <gazebo reference="front_realsense">
    <turnGravityOff>true</turnGravityOff>
    <sensor type="depth" name="front_realsense_depth">
      <update_rate>30</update_rate>
      <camera>
        <!-- 75x65 degree FOV for the depth sensor -->
        <horizontal_fov>1.5184351666666667</horizontal_fov>
        <vertical_fov>1.0122901111111111</vertical_fov>

        <image>
          <width>640</width>
          <height>480</height>
          <format>RGB8</format>
        </image>
        <clip>
          <!-- give the color sensor a maximum range of 50m so that the simulation renders nicely -->
          <near>0.01</near>
          <far>50.0</far>
        </clip>
      </camera>
      <plugin name="kinect_controller" filename="libgazebo_ros_openni_kinect.so">
        <baseline>0.2</baseline>
        <alwaysOn>true</alwaysOn>
        <updateRate>30</updateRate>
        <cameraName>realsense</cameraName>
        <imageTopicName>color/image_raw</imageTopicName>
        <cameraInfoTopicName>color/camera_info</cameraInfoTopicName>
        <depthImageTopicName>depth/image_rect_raw</depthImageTopicName>
        <depthImageInfoTopicName>depth/camera_info</depthImageInfoTopicName>
        <pointCloudTopicName>depth/color/points</pointCloudTopicName>
        <frameName>front_realsense_gazebo</frameName>
        <pointCloudCutoff>0.105</pointCloudCutoff>
        <pointCloudCutoffMax>8.0</pointCloudCutoffMax>
        <distortionK1>0.00000001</distortionK1>
        <distortionK2>0.00000001</distortionK2>
        <distortionK3>0.00000001</distortionK3>
        <distortionT1>0.00000001</distortionT1>
        <distortionT2>0.00000001</distortionT2>
        <CxPrime>0</CxPrime>
        <Cx>0</Cx>
        <Cy>0</Cy>
        <focalLength>0</focalLength>
        <hackBaseline>0</hackBaseline>
      </plugin>
    </sensor>
  </gazebo>

  <link name="front_realsense_lens">
    <visual>
      <origin xyz="0.02 0 0" rpy="${pi/2} 0 ${pi/2}" />
      <geometry>
        <mesh filename="package://realsense2_description/meshes/d435.dae" />
      </geometry>
      <material name="white" />
    </visual>
  </link>

  <joint type="fixed" name="front_realsense_lens_joint">
    <!-- Offset the camera 5cm forwards and 1cm up -->
    <origin xyz="0.05 0 0.01" rpy="0 0 0" />
    <parent link="front_mount" />
    <child link="front_realsense_lens" />
  </joint>
  <joint type="fixed" name="front_realsense_joint">
    <origin xyz="0.025 0 0" rpy="0 0 0" />
    <parent link="front_realsense_lens" />
    <child link="front_realsense" />
  </joint>
</robot>

********************************************************************************************************************************************************
refrence: https://www.clearpathrobotics.com/assets/guides/kinetic/jackal/additional_sim_worlds.html

Then run "export JACKAL_URDF_EXTRAS=$HOME/Desktop/realsense.urdf.xacro". You can also add that line to your ~/.bashrc file so you dont have to run it everytime you need to work with the realsense camera. Then run roslaunch pid_apriltag pariltag_jackal.launch which open gazebo with the apriltags and the jackal. The lauch file also launches the continuous detection file used to detect april tags.

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
 in simulation and real environment
 
# Package_4: Control_Mix 
In this package e combined the optical flow and fiducial markers algorithms together so that the robot can switch to optical-flow-based navigation as a backup option whenever fiducial landmarks are not visible.














