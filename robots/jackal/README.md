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
    export JACKAL_URDF_EXTRAS=$HOME/PATH_TO_VISION_BASED_NAVIGATION_TTT_FOLDER/vision_based_navigation_ttt/urdf/BU_Jackal.urdf.xacro/realsense.urdf.xacro
    ```
  
2. Download Clearpath package using the following command 
     ```
   sudo apt-get install ros-$ROS_Distro-jackal-simulator ros-$ROS_Distro-jackal-desktop ros-$ROS_Distro-jackal-navigation
     ```
3. Download Tensorflow if you are planning on using the CNN to get tau values
    ```
    pip install tensorflow
    
    ```

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

## Package_2: Output_feedback_controller
TODO

## Package_3: Vision_based_navigation_ttt
This package provides a means for a mobile robot equipped with a monocular camera to navigate in unknown environments using a visual quantity called time-to-transit (tau). The package includes code that utilizes computer vision techniques, specifically the Lucas-Kanade method, to estimate time-to-transit by calculating sparse optical flow. Additionally, the package offers an alternative method for computing tau values by employing a Deep Neural Network (DNN)-based technique to predict tau values directly from a couple of successive frames, and it also utilizes lidar to calculate tau values.

Moreover, the package includes a deep learning model that predicts the shape of the path ahead, which further enhances the robot's capability to navigate in an unknown environment.

The diagram of the ROS framework is shown in the figure

<img src="https://github.com/johnbaillieul/ROS-Jackal/blob/cnn_model/robots/jackal/vision_based_navigation_ttt/assets/diagram.png"/>

### How to Run the Package
To launch Gazebo you need to run 
``` 
roslaunch vision_based_navigation_ttt <your chosen file from launch folders>.launch 
```
To simulate your desired world specify it in the launch file at line: 

  ```
  arg name="world_name" value="$(find vision_based_navigation_ttt)/GazeboWorlds/<files with your desired world found in GazeboWorlds folder>.world" 
  ```
  
  #### Ways to calculate the tau values:
  
      To get tau values from optical flow run: 

      ```
      rosrun vision_based_navigation_ttt optical_flow.py
      rosrun vision_based_navigation_ttt tau_computation.py 
      ```

      To get tau values from velodyne(3D lidar) run:

      ```
      rosrun vision_based_navigation_ttt tau_computation_velodyne.py 
      ```
      To get tau values from lidar(2D lidar) run:

      ```
      rosrun vision_based_navigation_ttt tau_computation_lidar.py 
      ```

      To get tau values from CNN model run:

      ```
      rosrun vision_based_navigation_ttt tau_computation_cnn.py
      ```
      This window will show two values the top one is the cnn prediction and thebottom one is from the lidar. You can choose the parameters that you want that are available in the trained_model_parameters folder just change the model name in line tf.keras.models.load_model. Not that there are models that take velocities as input and others dont so make sure to choose the function that calculates tau values according tothe model you chose.


  #### Contollers available:
  
      To use the controller with sense and act phases, run 

      ```
      rosrun vision_based_navigation_ttt controller.py 
      ```

      To use the controller with only an act phase, run 

      ```
      rosrun vision_based_navigation_ttt controller_act_bias.py 
      ```

### Custom Worlds 
Multiple custom worlds were created in Gazebo to resemble the environment being tested on in the lab. 
 
<table border="0">
 <tr>
    <td><b style="font-size:30px">T_shaped corridor</b></td>
    <td><b style="font-size:30px">L_shaped corridor</b></td>
    <td><b style="font-size:30px">U_shaped corridor</b></td>
    <td><b style="font-size:30px">House Garden</b></td>
 </tr>
 <tr>
    <td>
<img src="https://github.com/johnbaillieul/ROS-Jackal/blob/cnn_model/robots/jackal/vision_based_navigation_ttt/assets/T_shaped.png"/> 
     </td>
     <td>
<img src="https://github.com/johnbaillieul/ROS-Jackal/blob/cnn_model/robots/jackal/vision_based_navigation_ttt/assets/L_shaped.png"/>
      </td>
     <td>
<img src="https://github.com/johnbaillieul/ROS-Jackal/blob/cnn_model/robots/jackal/vision_based_navigation_ttt/assets/U_shaped.png"/>
      </td>
      <td>
<img src="https://github.com/johnbaillieul/ROS-Jackal/blob/cnn_model/robots/jackal/vision_based_navigation_ttt/assets/House_garden.png"/>
      </td>
 </tr>
</table>

You can build up custom gazebo worlds by using wall segments. The video below shows how this can be done.

<img src="https://user-images.githubusercontent.com/98136555/185213284-8d2cfa97-f4ec-4a5c-a24f-7408b699c902.mp4" width=50% height=50%/>


### Performance
  Peformance can be affected by lighting as shown in the videos below.
  
  <table border="0">
 <tr>
    <td><b style="font-size:30px">Two lights</b></td>
    <td><b style="font-size:30px">Three lights</b></td>
 </tr>
 <tr>
    <td>
 
https://user-images.githubusercontent.com/98136555/185210652-f371b74c-7054-4f63-95b3-365b9713b741.mp4</td>
    <td>
      
https://user-images.githubusercontent.com/98136555/185215284-977937bf-bd99-4416-b706-a4d0d101c430.mp4</td>
 </tr>
</table>

 
### CNN-Based τ Predicition

The aim is to introduce a Convolutional Neural Network (DNN) that automatically estimates values of tau in the 5 regions of interests from a couple of images, without explicitly computing optical flow. It is reasonable to think that this network learns a form of optical flow in an unsupervised manner through its hidden layers.

  #### Data Collection
 
  To train the CNN, two consecutive images and the corresponding tau values in the respective regions of the images are required. The data_collection.py file is utilized for saving the images and tau values. The tau values are obtained from depth measurements using a lidar, as this provides the most reliable method for obtaining time-to-transit values. The node requires /image_raw and /tau_values topics to receive the required data.
  
  
  To collect data in simulation using a 2D lidar, run the commands below:
  ```
  roslaunch vision_based_navigation_ttt <name-of-launch-file>.launch front_laser:=1
  rosrun vision_based_navigation_ttt tau_computattion_lidar.py 
  rosrun vision_based_navigation_ttt controller.py 
  rosrun vision_based_navigation_ttt data_collection.py 
  ```

  By default, the images will be saved in the ```training_images``` folder and the distances are saved in the ```tau_values``` folder. 
  
  #### Available Model Architectures to Train :
  
  ##### 1. cnn_auto_ml
  This model uses "AutoKeras" which is an AutoML System. It takes two successive colored images as input, and outputs the distance in each region of interest. The distance is then converted to ```tau_value``` by dividing it by the robot's velocity.

 ###### Demo:
 <table border="0">
 <tr>
    <td><b style="font-size:30px">Model ran in an environment it was not trained on</b></td>
    <td><b style="font-size:30px">Model ran in a T-shaped corridor</b></td>
 </tr>
 <tr>
    <td>

https://user-images.githubusercontent.com/98136555/211264448-130d28b4-0fb9-4551-9ef9-4cc48a1fa0b1.mp4

 </td>
    <td>

https://user-images.githubusercontent.com/98136555/211263011-e2469251-4f1f-49e2-b989-e46dfc45e910.mp4
  </td>
 </tr>
</table> 
  
  ###### Model Architecture:
  
  <img src="https://user-images.githubusercontent.com/98136555/211239897-3d31f95e-03bc-45ba-96e7-9a65a0e81cef.png" width=25% height=25%/>
  
  
  ##### 2. cnn_colored_output_distance_in_each_roi
This model takes two colored images as input, and outputs an array that contains the distance in each roi.
   ###### Demo:

   <table border="0">
 <tr>
    <td><b style="font-size:30px">Model ran in an environment it was trained on</b></td>
    <td><b style="font-size:30px">Model ran in a T-shaped corridor</b></td>
 </tr>
 <tr>
    <td>

https://user-images.githubusercontent.com/98136555/211262755-43a8d499-1b23-40f4-a373-ea8c67d1b607.mp4

 </td>
    <td>

https://user-images.githubusercontent.com/98136555/211262738-a77bb3e2-d42a-404e-9bba-cd417e688f82.mp4

  </td>
 </tr>
</table>

   ###### Model Architecture:
   
  <img src="https://user-images.githubusercontent.com/98136555/211247640-d3bb4dd1-b210-4fbd-adc4-8059609093ae.png" width=25% height=25%/>

  ##### 3. cnn_grayscale_output_tau_value_in_each_roi
  This model takes two grayscale images and the velocity as input, and outputs an array that contains the ```tau_values``` in each roi.
  
   ###### Model Architecture:
   
   <img src="https://user-images.githubusercontent.com/98136555/211253489-fc6b081e-af00-4c99-a85f-3cd9153b509c.png" width=25% height=25%/>

  ##### 4. cnn_output_tau_value_in_each_roi_and_validity
  
  The model takes two successive images along with the velocity as input, and outputs two arrays one contains the tau values in each region of interest , and the other contains a flag that shows if the predicited value is valid or not.
  
  ###### Model Architecture:
  
  <img src="https://user-images.githubusercontent.com/98136555/203196927-e1a5df6a-b659-4cb5-899a-96971d8fb24e.png" width=25% height=25%/>

  ##### 5. cnn_input_roi_image
  
  Unlike the previous models, this model takes two successive images of the region of interest as input, and outputs the tau value in that region. This model is computationally expensive since it has to run 5 times to get the tau value for the 5 regions of interest.
  
   ###### Model Architecture:
   
 <img src="https://user-images.githubusercontent.com/98136555/203196713-d184d217-4d4c-4703-9a3e-b70578cf4f85.png" width=25% height=25%/>
  
  #### Collected data and trained models:
  Due to the large size of the datasets and the trained models they are saved on a shared drive.
  
  ### CNN-Based Turn Detection:
  For this we trained several well-known architectures and assessed their performance, and the ResNet50v2 architecture demonstrated the highest performance.
  
  The file reponsible for this is publish_shape_corridor.py, that subscribers to the image topic and produces a vector of three binary elements, indicating the presence of a left, right, or straight path respectively. 
  
  The dataset required to train this model can be collected using the automatic_label_sim.py file. It's worth noting that this function is currently only compatible with a limited number of predefined environments. If you need to add another environment, you will have to map the x, y, and theta coordinates of the robot to determine what turns are visible from that position. Another way to collect the dataset is to label the images manually using the GUI available in the manual_label_turns.py file.
  
  ### Lab Results:
  For predicting tau values, the model employed is based on the resnetv2-101 architecture, which was trained on data gathered from the real robot. On the other hand, to predict the shape of the corridor, specifically the upcoming turns visible in the image, a model based on the resnetv2-50 architecture is used. This model was trained using simulated and real data. The models used were selected based on their superior performance.
  
  <img src="https://github.com/johnbaillieul/ROS-Jackal/blob/cnn_model/robots/jackal/vision_based_navigation_ttt/assets/IROS23_lab_exp.mp4"/>
  
## Package_4: Control_Mix 
  
In this package e combined the optical flow and fiducial markers algorithms together so that the robot can switch to optical-flow-based navigation as a backup option whenever fiducial landmarks are not visible.
  
TODO














