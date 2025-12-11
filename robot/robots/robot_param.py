import numpy as np
import cv2

class RobotParam:
    def __init__(self, to_camera_rvec: np.ndarray = None, to_camera_tvec: np.ndarray = None):
        self.to_camera_rvec = to_camera_rvec
        self.to_camera_tvec = to_camera_tvec
        if to_camera_rvec is not None and to_camera_tvec is not None:
            self.to_camera_matrix = self._compute_extrinsic_matrix(to_camera_rvec, to_camera_tvec)
    
    def _compute_extrinsic_matrix(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        R, _ = cv2.Rodrigues(rvec)
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = tvec.flatten()
        return extrinsic_matrix
    
    def transform_to_world(self, points: np.ndarray) -> np.ndarray:
        if points.ndim == 1:
            points = points[np.newaxis, :]
        assert points.shape[1] == 3, "Points should be of shape (N, 3)"
        
        return (self.to_camera_matrix @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T[:, :3]
    
    def transform_from_world(self, points: np.ndarray) -> np.ndarray:
        if points.ndim == 1:
            points = points[np.newaxis, :]
        assert points.shape[1] == 3, "Points should be of shape (N, 3)"
        
        to_robot_matrix = np.linalg.inv(self.to_camera_matrix)
        return (to_robot_matrix @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T[:, :3]

    def save_to_file(self, file_dir: str):
        import json
        import pathlib
        file_path = pathlib.Path(file_dir) / "robot_param.json"
        data = {
            "to_camera_rvec": self.to_camera_rvec.tolist(),
            "to_camera_tvec": self.to_camera_tvec.tolist()
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def load_from_file(self, file_dir: str):
        import json
        import pathlib
        file_path = pathlib.Path(file_dir) / "robot_param.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.to_camera_rvec = np.array(data["to_camera_rvec"])
        self.to_camera_tvec = np.array(data["to_camera_tvec"])
        self.to_camera_matrix = self._compute_extrinsic_matrix(self.to_camera_rvec, self.to_camera_tvec)
    
if __name__ == "__main__":
    import math
    import time
    
    from cameras.realsense_env import RealSenseEnv
    from cameras.camera_param import CameraParam

    ################## Calibration Procedure ##################
    # 1. Calib on xy plane
    # point = robot.robot_param.transform_to_world(ee_xyz)
    # point[2] = 0.0
    # observe the point in camera frame, making sure the gripper eef projection on the xy plane is at the target point
    # 2. Calib on z axis
    # point = robot.robot_param.transform_to_world(ee_xyz)
    # move the gripper up and down, making sure the z value is correct
    ################## Calibration Procedure ##################

    rvec_robot_to_cam = np.array([ 0.0, 0.0, -math.pi / 2])
    tvec_robot_to_cam = np.array([ 0.53433071, 0.52905707, 0.00440881])
    robot_param = RobotParam(rvec_robot_to_cam, tvec_robot_to_cam)
    main_camera_matrix = np.array([[908.1308, 0, 655.7268], [0, 910.0818, 395.8856], [0, 0, 1]], dtype=np.float32)
    main_camera_dist_coeffs = np.array([0.1068, -0.2123, -0.0092, 0.0000, 0.0000], dtype=np.float32)
    camera = RealSenseEnv(camera_name="main_image", serial_number="339322073638", camera_param=CameraParam(main_camera_matrix, main_camera_dist_coeffs))
    camera.calib_camera()
    camera.start_monitoring()
    from robots.franky_env import FrankyEnv
    from common.constants import ActionSpace
    robot = FrankyEnv(robot_param=robot_param, action_space=ActionSpace.EEF_VELOCITY)
    robot.reset()
    from controllers.spacemouse_env import SpaceMouseEnv
    controller = SpaceMouseEnv(robot_env=robot)
    controller.start_controlling()
    while True:
        print("Capturing frame and robot pose...")
        frame = None
        while frame is None:
            time.sleep(0.05)
            frame = camera.get_latest_frame()
        image = frame['bgr']
        ee_pose = robot.get_position(action_space=ActionSpace.EEF_POSE)
        ee_xyz = ee_pose[:3]
        print("End-Effector Position (Robot Frame):", ee_xyz)
        point = robot.robot_param.transform_to_world(ee_xyz)
        point[0][2] = 0
        camera.camera_param.draw_trajectory_on_image(image, point)

        cv2.imshow("Image with Trajectory", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break