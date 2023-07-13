## PID_apriltag 
It applies PID control to move the jackal from one apriltag to the other apriltags. Using the realsense camera and the apriltag_ros package the robot detects an apriltag and moves towards it. Once the jackal is close enough it starts spinning in its place to detect another apriltag and moves towards it. 

The repository also includes a template folder that can be used to add new apriltags.

https://user-images.githubusercontent.com/98136555/187280048-6306f278-5905-4383-acf9-1cb138191c13.mp4


### How to Run the Package:
1. Run the command below. The realsense urdf.xacro file is by default located in the vision_based_navigation project, make sure to update the command below if you change it's location.
  ```
   export JACKAL_URDF_EXTRAS=$HOME/catkin_ws/src/vision_based_navigation_ttt_ml/urdf/realsense.urdf.xacro
   ```
&ensp; &ensp; &ensp; Note that the export command has to be run every time before launching a file. To avoid that, you can add that command to your ~/.bashrc file. To get to your bashrc file run nano " ~/.bashrc "

2. Run the command below which opens gazebo with the apriltags and the Jackal. The lauch file also launches the continuous detection file used to detect april tags.
```
roslaunch pid_apriltag apriltags_jackal.launch
```
3. to run the robot

```
rosrun pid_apriltag detec.py
```
### Apriltag Detection
To check if the apriltag detection is running:
1. Open a new terminal
2. Run ```rqt_image_view ```
3. Select /tag_detections_image
4. You should be able to see a frame on the tag similar to the image below:
<img width="360" alt="Screen Shot 2023-01-08 at 12 47 17 AM" src="https://user-images.githubusercontent.com/98136555/211182420-8cb5ad5f-b38d-4616-aac0-fccca0a94971.png">


Note that when using apriltags, having lighting in your world that is too bright or too dark can cause wrong detections.

### How to Run the Package on the Jackal:

1. Connect the jackal to a monitor and run "ifconfig" to obtain the IP address of the jackal.
2. SSH to the jackal from your desktop using the following command: "ssh -X clearpath@<ip_address>". You will be prompted to enter the password.
3. Open four terminals (ensure that you have SSHed to the jackal in each terminal before running the commands):
  a. In the first terminal, launch the Realsense camera: "roslaunch realsense2_camera rs_rgbd.launch". To verify that it is running, you can either use rqt to view the image or list the     topics and check if the camera is listed by executing the command "rostopic list". Note: This command needs to be run in a separate terminal where you have already SSHed to the jackal.
  b. In the second terminal, run "roslaunch apriltag_ros continuous_detections_.launch".
  c. In the third terminal, execute "rosrun tf static_transform_publisher -0.18 0 0.23 0 0 0 front_mount camera_color_optical_frame 10".
  d. In the fourth terminal, run "rosrun jackal_iros detec.py".
