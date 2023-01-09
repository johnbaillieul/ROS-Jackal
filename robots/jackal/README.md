# Setup & Startup Guide for BU's Jackal Robots

## Startup

## Prerequisites
1. Setup Realsense_camera: see instruction from https://wiki.bu.edu/robotics/index.php?title=Jackal or https://www.clearpathrobotics.com/assets/guides/kinetic/jackal/additional_sim_worlds.html

2. Download Clearpath package: To simulate Jackal UGV that can be installed by running: sudo apt-get install ros-melosic-jackal-simulator ros-melodic-jackal-desktop ros-melodic-jackal-navigation

## Setup:
  1. Create a catkin workspace
  2. Clone the repository in the src folder of your catkin workspace
  3. Add the line below to your ~/.bashrc:
  ```
  export GAZEBO_MODEL_PATH=<path_to_src_folder>/src/<pacakge name>/models
  ```
  
This is what my bashrc file looks like for your reference

![Screenshot from 2022-10-21 11-42-51](https://user-images.githubusercontent.com/98136555/203122995-43e2dc0f-d416-4e50-a8de-1335949a1bbe.png)


## Packages
## Package_1: PID_apriltag 
It applies PID control to move the jackal from one apriltag to the other apriltags. Using the realsense camera and the apriltag_ros package the robot detects an apriltag and moves towards it. Once the jackal is close enough it starts spinning in its place to detect another apriltag and moves towards it. 

The repository also includes a template folder that can be used to add new apriltags.

https://user-images.githubusercontent.com/98136555/187280048-6306f278-5905-4383-acf9-1cb138191c13.mp4


### How to Run the Package:
1. Run the command below. The realsense urdf.xacro file is by default located in the vision_based_navigation project, make sure to update the command below if you change it's location.
  ```
   export JACKAL_URDF_EXTRAS=$HOME/catkin_ws/src/vision_based_navigation_ttt/urdf/BU_Jackal.urdf.xacro
   ```
&ensp; &ensp; &ensp; Note that the export command has to be run every time before launching a file. To avoid that, you can add that command to your ~/.bashrc file. 

2. Run the command below which opens gazebo with the apriltags and the Jackal. The lauch file also launches the continuous detection file used to detect april tags.
```
roslaunch pid_apriltag apriltag_jackal.launch
```

### Apriltag Detection
To check if the apriltag detection is running:
1. Open a new terminal
2. Run ```rqt_image_view ```
3. Select /tag_detections_image
4. You should be able to see a frame on the tag similar to the image below:
<img width="360" alt="Screen Shot 2023-01-08 at 12 47 17 AM" src="https://user-images.githubusercontent.com/98136555/211182420-8cb5ad5f-b38d-4616-aac0-fccca0a94971.png">


Note that when using apriltags, having lighting in your world that is too bright or too dark can cause wrong detections.

## Package_2: Output_feedback_controller
TODO

## Package_3: Vision_based_navigation_ttt

### How to Run the Package
For ease of use each environment has its launch file however it is possible to simulate the desired world by specifying your desired world in the launch file at this line: arg name="world_name" value="$(find vision_based_navigation_ttt)/GazeboWorlds/<desired .world file>"/.After chosing the launch file run  roslaunch vision_based_navigation_ttt <your chosen file>.launch. Then run rosrun vision_based_navigation_ttt optical_flow.py, rosrun vision_based_navigation_ttt tau_computation.py and rosrun vision_based_navigation_ttt controller.py in separate terminals.

### Custom Worlds 
Multiple custom worlds were created in Gazebo to resemble the environment being tested on in the lab. 
TODO add pictures of dif worlds
  
You can build up custom gazebo worlds by using wall segments. The video below shows how this can be done.

https://user-images.githubusercontent.com/98136555/185213284-8d2cfa97-f4ec-4a5c-a24f-7408b699c902.mp4
 
### Performance can be affected by lighting 

 #### 2 lights
 
https://user-images.githubusercontent.com/98136555/185210652-f371b74c-7054-4f63-95b3-365b9713b741.mp4
 
 #### three lights

https://user-images.githubusercontent.com/98136555/185215284-977937bf-bd99-4416-b706-a4d0d101c430.mp4
 
### Machine learning models

  In this approach we aim to calculate the Optical flow, the motion between consecutive frames of sequences caused by relative motion between a camera and an object, by using a machine learning model instead of Lucas Kanade method. The model generated is expected to predict the tau_value in each region of interest given images of the scene. To train such a model, we collected images with the distances of the robot from the obstacles in each region of interest. To get the distances we used a 2D lidar. The code that does that is in the data_collection.py file. 
To collect data, first you need to launch gazebo with this command 
      roslaunch vision_based_navigation_ttt L_shaped_corridor.launch front_laser:=1
  then run 
  
    rosrun vision_based_navigation_ttt tau_computattion_lidar.py 
    rosrun vision_based_navigation_ttt controller_not_modified.py 
    rosrun vision_based_navigation_ttt data_collection.py 

The images are saved in training_images folder and the distance values are save in the tau_values folder. The distances are then divided by the           velocity to get the true tau_value.
  
For now all the data where in a simulated gazebo environments. To change the colors of the boxes in the enviroment whenever you launch a file you can add 
<node name="change_env" pkg="vision_based_navigation_ttt" type="change_env.py"/> to your launch file.

The folder called trained_model_parameters contains the parameters of the trained different cnn models. Each model containes two classes one to train the model and the other to inference.
  
#### Model 2 
  
  This model only takes one image as input and outputs an array with the tau values in each Region of Interest. 
  
  #### Model 3
  ********************* NEEDS EDITING *******************************************************************
  Is an updated version of model 3 where the difference is that it uses average pooling instead of max pooling
  
  Total trainable params: 1,777,083
  metric achieved 
  training_loss: 0.2101 - training_accuracy: 0.8752 - 
  validation_loss: 0.3516 - validation_accuracy: 0.8247
  test loss: 0.36275410652160645, test accuracy: 0.8163265585899353
  
  When the model was trained with labels that dont have a -1 flag value the metrics achieved were as follow:
  
  #### metrics achieved:
  
  training-loss: 0.2706 - training-accuracy: 0.8840 
  
  validation_loss: 0.4683 - validation_accuracy: 0.8547
  
  test loss: 0.48675239086151123, test accuracy: 0.8511404395103455
  
  what we did here is changing the tau_values with -1 value to  15. Note that the -1 fal value corresponding to a infinity tau_value.
  
  #### Model 4
  
  Takes two images along with the velocity and outputs two arrays on with the tau values in each Region of Interest and the other array with a flag that shows if the predicited value is valid or not.
  
  ![model_with_shape_info](https://user-images.githubusercontent.com/98136555/203197318-052f6550-b1ab-4583-a185-ace9e60a9ef7.png)

  
  #### Model 5
  Same as model 6 but instead of predicting whether the tau value is valid or not after appending the velocity it is predicited before that because the image alone should be suffice to predict the tau value's validity.
  Model architecture:
  
  ![model_with_shape_info](https://user-images.githubusercontent.com/98136555/203196927-e1a5df6a-b659-4cb5-899a-96971d8fb24e.png)

  #### Model 6
  The input to this model is an image of the region of interest unlike in all the other models where the input was the entire image and the output is only one tau value. It proved to be computationally expensive becasue it has to be run 5 time to get the tau value in exh of the 5 regions of interest.
  Model architecture:
  ![model_ROI](https://user-images.githubusercontent.com/98136555/203196713-d184d217-4d4c-4703-9a3e-b70578cf4f85.png)

  
  ### Takeaway
  
  All of the models are not robust enough to work in environments other than the ones with boxes. One thing to be noted is that white walls are of a problem cause the model so the training data sho
 
## Package_4: Control_Mix 
In this package e combined the optical flow and fiducial markers algorithms together so that the robot can switch to optical-flow-based navigation as a backup option whenever fiducial landmarks are not visible.














