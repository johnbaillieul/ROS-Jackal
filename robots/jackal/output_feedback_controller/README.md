This package provides a robust output feedback controller implementation for a robotic system, as described in the paper titled "Robust Sample-Based Output-Feedback Path Planning" (https://arxiv.org/pdf/2105.14118.pdf), which can be used in Gazebo simulation.

# To use the package:
to launch gazebo with with the jackal in a onves environment 
```
roslaunch output_feedback_controller vertex_walls_world.launch
```

This is what the environment will look like.

<img src="https://user-images.githubusercontent.com/98136555/235841853-a2cd2266-a101-40ff-944c-35b6ba4f0b6e.png" width="500" height="400">

Then run the following to get the robot to move in the environment
```
run rosrun output_feedback_controller feedback_controller.py 
```
This file uses the K_gains matrix saved in the csv folder of the package.

You can also visualize the robot and the detected apriltags in Rviz by running 
```
  roslaunch jackal_viz view_robot.launch
```
This is what you will get 

<img src="https://user-images.githubusercontent.com/98136555/235842424-e17e6987-079a-45f0-a7ad-4205d90e9f79.png" width="500" height="400">


# Custom usage of the package

1. You must create a convex Gazebo environment with apriltags. 

Here's how you can do it:

A. Install the apriltag-ros package available at https://github.com/AprilRobotics/apriltag_ros.

B. Use Blender to create a .dae model of a cube with the image of the April tag on one of the faces, using the wrap texture technique. The models folder already contains models of the 20 apriltags that can be used. If you want different sizes for the boxes or different apriltags, you will need to create your own. Once you have the dae file, add the new folder to the models file with the name you want, and make sure to follow the same format as the available models.

C. Modify the config/settings.yaml and config/tags.yaml files in the apriltag_ros package. In the settings.yaml file, identify the tag_family you are using and set publish_tf to true to receive transformation messages. In the tags.yaml file, add the tag ids that you used along with the size. You can get the size by measuring the length of the side of the black square of the April tag. Adding a name is optional. For example:

standalone_tags:
  [{id: 1, size: 0.81 , name: tag_1},
  {id: 2, size: 0.81 , name: tag_2}
  ]
A PDF of the April tags can be found here https://www.dotproduct3d.com/uploads/8/5/1/1/85115558/apriltags1-20.pdf

D. Create the environment from either boxes or walls, but make sure it's a convex environment, and add the apriltags preferably at the vertices. More detailed instructions can be found at https://wiki.bu.edu/robotics/index.php?title=Jackal.

2. Get the K_gains for the controller by running the MATLAB code available at https://bitbucket.org/tronroberto/codecontrol/.

To obtain the K-gains from the MATLAB code, follow these steps:

Get the location of the apriltags from Gazebo. Note: you can use the Gazebo GetModel service to get the locations.
Get the location of the vertices in the environment with respect to the center frame.
An example of how to create the environment in MATLAB can be found at ~/codemeta/control/RRTstar_CBFCLF/homograph/test_journal.m.

#Things to note:

To use the K-gains matrix you need to have the location of the robot in the world frame for that multiple transformations should be done.

<img src="https://user-images.githubusercontent.com/98136555/235844022-5cc91bdb-7613-4c7a-ac83-268f04a63ce3.png" width="700" height="400">

To get the transformation of the robot in the world environment you can do the following:

1) Subscribe to the /apriltag_detections ROS topic provided by the realsense_ros package to obtain the transformation of the apriltag in the realsense_gazebo link.
2) Determine the transformation from the apriltag frame set in the apriltag_ros package to the frame in Gazebo. This is a static transformation, and you can verify the frame in RViz.you can check how that frames are different in the image above.
3) Get the transformation of the apriltag in gazebo with respect to the world frame using the Gazebo GetModel service.
4) Multiply the transformations: (3)(2)(1)^-1 to obtain the transformation of the robot in the world environment, specifically the realsense_gazebo link. Note that the camera is static with respect to the baselink, so you can apply a transformation from the camera to the baselink.
Check out the feedback_controller.py node to see how this transformation is implemented.
