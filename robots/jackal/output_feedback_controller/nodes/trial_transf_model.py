#!/usr/bin/env python3

#import the dependencies
 
import rospy
from geometry_msgs.msg import Twist, Pose, TwistStamped
from apriltag_ros.msg import AprilTagDetectionArray
import transformation_utilities as tu
import numpy as np
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf2_geometry_msgs
from std_msgs.msg import Float64
from gazebo_msgs.srv import GetModelState, GetModelStateRequest


rospy.wait_for_service('/gazebo/get_model_state')
get_model_srv = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
model = GetModelStateRequest()
model.model_name = 'unit_sphere'
model.relative_entity_name = 'unit_box'
result = get_model_srv(model)
result_trans = tu.msg_to_se3(result.pose)
# result_trans[3,2] = result_trans[3,2] + 0.5
# print('id',id,result.pose.position)
print('getmodel',result_trans[:3,:3])
# print(self.position_landmark_inworld_matrix)