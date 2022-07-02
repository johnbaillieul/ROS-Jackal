from marc_demo.main_driver import Jackal
class Mahroo():
    def __init__(self) -> None:
        pass

    def activate(self):
        ZERO_TAG_OFFSET = np.array([0,-0.327]) 

        rospy.init_node("jackal_iros_node")
 ##TODO: cHANGE THE PATH  
        shared_path = os.environ["HOME"]+"/jackal_ws/src/jackal_iros_2022/csv/"
        K_dir = shared_path + "K_gains.csv"
        landmark_positions_dir = shared_path + "landmark_positions.csv"
        landmark_orientations_dir = shared_path + "landmark_orientations_deformed.csv"
        transform_matrix_dir = shared_path + "transform_matrix.csv"
        obstacle_coordinates_dir = shared_path + "global_obstacle_coordinates.csv"

        visual_landmarks_dir = shared_path + "landmark_visual_positions.csv"

        try:
            visual_landmarks = read_matrix(visual_landmarks_dir)
        except Exception as e:
            raise e
            raise NotImplementedError("We don't have coordinates for visual landmarks yet!")

        K_gains = read_matrix(K_dir)
        landmark_positions = read_matrix(landmark_positions_dir)
        landmark_orientations = read_matrix(landmark_orientations_dir)
        transform_matrix = read_matrix(transform_matrix_dir) # From local to global
        obstacle_coordinates = read_matrix(obstacle_coordinates_dir)

        ## Transform obstacle and landmark coordinates to Mahroo's reference frame:
        obstacle_coordinates = (GLOBAL_ROTATION_MATRIX.dot(obstacle_coordinates.T) + GLOBAL_TRANSFORM_MATRIX[:3,3][:,None]).T
        landmark_positions = np.concatenate((landmark_positions[:,0][:,None],
                    (GLOBAL_ROTATION_MATRIX.dot(landmark_positions[:,1:-1].T) + GLOBAL_TRANSFORM_MATRIX[:3,3][:,None]).T,
                    landmark_positions[:,-1][:,None]),
                    axis=1
        )

        # Transform the rotation types as well:
        print("TANSFORM MATRIX:\n",transform_matrix)
        for i in range(len(ROTATION_TYPES)):
            ROTATION_TYPES[i] = GLOBAL_ROTATION_MATRIX.dot(ROTATION_TYPES[i])

        obstacle_list = make_my_obstacles(obstacle_coordinates)
        landmarks_dict = make_my_landmarks(landmark_positions)

        # Save the coordinates in Mahroo's reference frames
        write_matrix(shared_path + 'obstacle_coordinates_transformed.csv',obstacle_coordinates)
        write_matrix(shared_path + 'landmark_coordinates_transformed.csv',landmark_positions)

        
        if SIMULATION:
            start_point = np.array([1.,4.,0.]).T
        else:
            start_point = None

        jackal = Jackal(K_gains,landmarks_dict,obstacle_list=obstacle_list,start_point=start_point,sequence_idx=2,orientation_=[0,-1,0])

        rospy.on_shutdown(jackal.on_rospy_shutdown)

        rospy.loginfo("jackal Jackald.")

        jackal.navigate()

        # if SIMULATION:
        #     rospy.spin()

    def deactivate(self):
        pass