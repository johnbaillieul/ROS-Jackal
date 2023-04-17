This package provides a robust output feedback controller implementation for a robotic system, as described in the paper titled "Robust Sample-Based Output-Feedback Path Planning" (https://arxiv.org/pdf/2105.14118.pdf), which can be used in Gazebo simulation.

To use the package, follow these steps:

Obtain the K_gains for the controller by running the MATLAB code available at https://bitbucket.org/tronroberto/codecontrol/.
Once you have obtained the K_gains, you can use them for your application.
Before obtaining the K_gains from the MATLAB code, you must first create a convex Gazebo environment with apriltags. Here's how you can do it:

Install the apriltag-ros package available at https://github.com/AprilRobotics/apriltag_ros.
Use Blender to create a .dae model of a cube with the image of the April tag on one of the faces, using the wrap texture technique. The models folder already contains models of the 20 apriltags that can be used. If you want different sizes for the boxes or different apriltags, you will need to create your own. Once you have the dae file, add the new folder to the models file with the name you want, and make sure to follow the same format as the available models.
Modify the config/settings.yaml and config/tags.yaml files in the apriltag_ros package. In the settings.yaml file, identify the tag_family you are using and set publish_tf to true to receive transformation messages. In the tags.yaml file, add the tag ids that you used along with the size. You can get the size by measuring the length of the side of the black square of the April tag. Adding a name is optional. For example:
yaml
Copy code
standalone_tags:
  [{id: 1, size: 0.81 , name: tag_1},
  {id: 2, size: 0.81 , name: tag_2}
  ]
A PDF of the April tags can be found here https://www.dotproduct3d.com/uploads/8/5/1/1/85115558/apriltags1-20.pdf
4. Create the environment from either boxes or walls, but make sure it's a convex environment, and add the apriltags preferably at the vertices. More detailed instructions can be found at https://wiki.bu.edu/robotics/index.php?title=Jackal.

To obtain the K-gains from the MATLAB code, follow these steps:

Get the location of the apriltags from Gazebo. Note: you can use the Gazebo GetModel service to get the locations.
Get the location of the vertices in the environment with respect to the center frame.
An example of how to create the environment in MATLAB can be found at ~/codemeta/control/RRTstar_CBFCLF/homograph/test_journal.m.
