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
