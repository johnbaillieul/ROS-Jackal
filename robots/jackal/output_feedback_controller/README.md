This package contains the implementation of a robust output feedback controller for a robotic system as described in Robust Sample-Based Output-Feedback Path Planning (https://arxiv.org/pdf/2105.14118.pdf) in Gazebo simulation. 

To work with the package:

1) Get the K_gains for the controller form the matlab code found in https://bitbucket.org/tronroberto/codecontrol/.
2) Use the K_gains 


Before getting the K_gains from the matlab code you  will need to first Create a convex Gazebo environment with apriltags.

To create a Gazebo environment with apriltags:
1) Install apriltag-ros package found in https://github.com/AprilRobotics/apriltag_ros
2) Using Blender to create a .dae model of a cube with the image of the April tag on one of the faces using the wrap texture technique. The models folder already conatins models of the 20 apriltags that can be used. You will need to create your own if you want different sizes for the boxes or different apriltags.
once you have the dae file add the new folder to the models file with the name you want and make sure to follow the same as format as the available models.
3)In the apriltag_ros package you need to modify the config/settings.yaml file and config/tags.yaml file.
In the settings.yaml file you need to identify the tag_family you are using and you can set publish_tf  to true to receive transformation messages.
In the tags.yaml file you need to add the tag ids that you used along with the size. You get the size by measuring the length of the side of the black square of the April tag. Adding a name is optional.

example: 
```
standalone_tags:
  [{id: 1, size: 0.81 , name: tag_1},
  {id: 2, size: 0.81 , name: tag_2}
  ]
```
A PDF of the April tags can be found here https://www.dotproduct3d.com/uploads/8/5/1/1/85115558/apriltags1-20.pdf

4) Create the enviroment from either boxes or walls but make sure its a convex enviroment and add the apriltags preferable at the verticies. 

More detailed instructions can be found in https://wiki.bu.edu/robotics/index.php?title=Jackal.

To get the K-gains from the Matlab code:
1) Get the location of the apriltags from Gazebo. Note: you can use the gazebo GetModel service to get the locations.
2) Get the location of the vertecies in the environment wrt the center frame.
3) Contains an example of how to create the enviroemnt in Matlab ~/codemeta/control/RRTstar_CBFCLF/homograph/test_journal.m 
