import numpy as np
import cv2
from aprilgrid import Detector as AprilDetector

class AprilGridBoard:
    def __init__(self, size, square_size, gap_size):
        self.size: list[int, int] = size
        self.square_size: float = square_size
        self.gap_size: float = gap_size
        self.board_corners = np.empty((0, 4, 3), dtype=np.float32)
        self.board_corners = self._get_board_corners() 
        self.board_points: np.ndarray = np.array([
            [0, 0, 0],
            [0, 0.4000, 0],
            [0.871, 0.411, 0],
            [0.871, 0, 0],
            [0.437, -0.1215, 0],
        ], dtype=np.float32)

    def _get_board_corners(self):
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                base_point = np.array([j * (self.square_size + self.gap_size), i * (self.square_size + self.gap_size), 0])
                base_point[0] += self.gap_size
                base_point[1] += self.gap_size
                corners = np.array([
                    base_point,
                    base_point + np.array([self.square_size, 0, 0]),
                    base_point + np.array([self.square_size, self.square_size, 0]),
                    base_point + np.array([0, self.square_size, 0])
                ])
                self.board_corners = np.append(self.board_corners, corners[np.newaxis, :, :], axis=0)
        return self.board_corners
    
class CameraParam:
    def __init__(self, intrinsic_matrix: np.ndarray = None, distortion_coeffs: np.ndarray = None, extrinsic_rvec: np.ndarray = None, extrinsic_tvec: np.ndarray = None):
        self.intrinsic_matrix = intrinsic_matrix
        self.distortion_coeffs = distortion_coeffs
        self.extrinsic_tvec = extrinsic_tvec
        self.extrinsic_rvec = extrinsic_rvec
        if extrinsic_tvec is not None and extrinsic_rvec is not None:
            self.extrinsic_matrix = self._compute_extrinsic_matrix(extrinsic_rvec, extrinsic_tvec)
        else:
            self.extrinsic_matrix = None

    def update_extrinsic(self, rvec: np.ndarray, tvec: np.ndarray):
        self.extrinsic_rvec = rvec
        self.extrinsic_tvec = tvec
        self.extrinsic_matrix = self._compute_extrinsic_matrix(rvec, tvec)

    def _compute_extrinsic_matrix(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Compute extrinsic matrix"""
        R, _ = cv2.Rodrigues(rvec)
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = R
        extrinsic_matrix[:3, 3] = tvec.flatten()
        return extrinsic_matrix
    
    def save_to_file(self, file_dir: str, camera_name: str = "default_image"):
        """Save camera parameters to json file"""
        import json
        import pathlib
        file_path = pathlib.Path(file_dir) / f"{camera_name}_camera_param.json"
        data = {
            "intrinsic_matrix": self.intrinsic_matrix.tolist(),
            "distortion_coeffs": self.distortion_coeffs.tolist(),
            "extrinsic_tvec": self.extrinsic_tvec.tolist() if self.extrinsic_tvec is not None else None,
            "extrinsic_rvec": self.extrinsic_rvec.tolist() if self.extrinsic_rvec is not None else None,
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
    
    def load_from_file(self, file_dir: str, camera_name: str="default_image"):
        """Load camera parameters from json file"""
        import json
        import pathlib

        file_path = pathlib.Path(file_dir) / f"{camera_name}_camera_param.json"
        if pathlib.Path.exists(file_path) is False:
            file_path = pathlib.Path(file_dir) / f"camera_param.json"
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.intrinsic_matrix = np.array(data["intrinsic_matrix"], dtype=np.float32)
        self.distortion_coeffs = np.array(data["distortion_coeffs"], dtype=np.float32)
        self.extrinsic_tvec = np.array(data["extrinsic_tvec"], dtype=np.float32) if data["extrinsic_tvec"] is not None else None
        self.extrinsic_rvec = np.array(data["extrinsic_rvec"], dtype=np.float32) if data["extrinsic_rvec"] is not None else None
        if self.extrinsic_tvec is not None and self.extrinsic_rvec is not None:
            self.extrinsic_matrix = self._compute_extrinsic_matrix(self.extrinsic_rvec, self.extrinsic_tvec)
        else:
            self.extrinsic_matrix = None

    def estimate_camera_poses(self, image, 
                              board: AprilGridBoard = AprilGridBoard(size=[2, 2], square_size=0.066, gap_size=0.020), 
                              detector: AprilDetector = AprilDetector('t36h11')):
        """
        Estimate camera pose using ArUco calibration board
        :param image: Input image
        :param board: ArUco calibration board object, containing calibration board information
        :param camera_matrix: Camera intrinsic matrix
        :param dist_coeffs: Camera distortion coefficients
        :param detector: ArUco marker detector
        :return: rvec, tvec rotation vector and translation vector
        """
        rvec, tvec = None, None
        board_corners = board.board_corners

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        tags = detector.detect(gray)
        coners_2d_all = []
        corners_3d_all = []
        for tag in tags:
            id = tag.tag_id
            corners = tag.corners
            corners_2d = corners.reshape(-1, 2)
            corners_3d = board_corners[id % 4].reshape(-1, 3) + board.board_points[id // 4]
            coners_2d_all.append(corners_2d)
            corners_3d_all.append(corners_3d)
            cnt = 0
            for corner in corners_2d:
                cnt+=1
                cv2.circle(image, (int(corner[0]), int(corner[1])), 5, (255, 0, 0), -1)
                cv2.putText(image, str(cnt), (int(corner[0]), int(corner[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            center = np.mean(corners_2d, axis=0)
            cv2.putText(image, str(id), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        
        # pnp
        if len(coners_2d_all) > 0:
            coners_2d_all = np.concatenate(coners_2d_all, axis=0)
            corners_3d_all = np.concatenate(corners_3d_all, axis=0)
            success, rvec, tvec = cv2.solvePnP(corners_3d_all, coners_2d_all, self.intrinsic_matrix, self.distortion_coeffs)
            if not success:
                print("PnP failed")
        self.extrinsic_rvec = rvec
        self.extrinsic_tvec = tvec
        self.extrinsic_matrix = self._compute_extrinsic_matrix(rvec, tvec)
        return rvec, tvec
    
    def draw_cube_and_axis_on_image(self, image, cube_size=0.1):
        """
        Draw cube on image
        :param image: Input image
        :param cube_size: Cube side length
        :return: Image after drawing cube
        """
        if self.extrinsic_rvec is None or self.extrinsic_tvec is None:
            raise ValueError("Extrinsic parameters are not set. Please run estimate_camera_poses first.")
        
        # Define the 8 vertices of the cube
        half_size = cube_size / 2
        cube_points_3d = np.array([
            [-half_size, -half_size, 0],
            [ half_size, -half_size, 0],
            [ half_size,  half_size, 0],
            [-half_size,  half_size, 0],
            [-half_size, -half_size, cube_size],
            [ half_size, -half_size, cube_size],
            [ half_size,  half_size, cube_size],
            [-half_size,  half_size, cube_size],
        ], dtype=np.float32)

        cube_points_3d += np.array([half_size, half_size, 0])  # Place the cube at the origin of the camera coordinate system

        # Project 3D points to 2D image plane
        img_points, _ = cv2.projectPoints(cube_points_3d, self.extrinsic_rvec, self.extrinsic_tvec, self.intrinsic_matrix, self.distortion_coeffs)
        img_points = img_points.reshape(-1, 2).astype(int)

        # Draw the 12 edges of the cube
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        for edge in edges:
            pt1 = tuple(img_points[edge[0]])
            pt2 = tuple(img_points[edge[1]])
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        # Draw coordinate axes
        axis_length = cube_size * 1.5
        axis_points_3d = np.array([
            [0, 0, 0],
            [axis_length, 0, 0],
            [0, axis_length, 0],
            [0, 0, axis_length]
        ], dtype=np.float32)
        axis_img_points, _ = cv2.projectPoints(axis_points_3d, self.extrinsic_rvec, self.extrinsic_tvec, self.intrinsic_matrix, self.distortion_coeffs)
        axis_img_points = axis_img_points.reshape(-1, 2).astype(int)
        cv2.line(image, tuple(axis_img_points[0]), tuple(axis_img_points[1]), (0, 0, 255), 2)  # X-axis red
        cv2.line(image, tuple(axis_img_points[0]), tuple(axis_img_points[2]), (0, 255, 0), 2)  # Y-axis green
        cv2.line(image, tuple(axis_img_points[0]), tuple(axis_img_points[3]), (255, 0, 0), 2)  # Z-axis blue
        return image
    def draw_trajectory_on_image(self, image, trajectory):
        """
        Draw hand trajectory on image
        :param image: Input image
        :param trajectory: Hand trajectory list, each element is a (x, y, z) coordinate tuple
        :param CameraParam: CameraParam object, containing camera intrinsic and extrinsic parameters
        :return: Image after drawing trajectory
        """
        if self.extrinsic_rvec is None or self.extrinsic_tvec is None:
            raise ValueError("Extrinsic parameters are not set. Please run estimate_camera_poses first.")
        for i in range(len(trajectory)):
            point1 = trajectory[i]

            # Project 3D points to 2D image plane
            img_point1, _ = cv2.projectPoints(
                np.array([point1], dtype=np.float32), 
                self.extrinsic_rvec, 
                self.extrinsic_tvec, 
                self.intrinsic_matrix, 
                self.distortion_coeffs
            )
            x, y = img_point1[0][0]
            if x > image.shape[1] or x < 0 or y > image.shape[0] or y < 0:
                continue
            color_ratio = i / max(1, len(trajectory) - 1)  # Avoid division by zero error
            green_intensity = int(255 * (1 - color_ratio))  # Green component decreases
            red_intensity = int(255 * color_ratio)  # Red component increases
            color = (0, green_intensity, red_intensity)  # BGR format: blue, green, red
            cv2.circle(image, tuple(img_point1[0][0].astype(int)), 5, color, -1)  # Draw point
        return image

    def transform_to_camera(self, points_in_world: np.ndarray) -> np.ndarray:
        """
        Transform points from world coordinate system to camera coordinate system
        :param points_in_world: Points in world coordinate system, shape (N, 3)
        :return: Points in camera coordinate system, shape (N, 3)
        """
        if self.extrinsic_matrix is None:
            raise ValueError("Extrinsic parameters are not set. Please run estimate_camera_poses first.")
        
        img_points, _ = cv2.projectPoints(points_in_world, self.extrinsic_rvec, self.extrinsic_tvec, self.intrinsic_matrix, self.distortion_coeffs)
        img_points = img_points.reshape(-1, 2).astype(int)
        return img_points

if __name__ == "__main__":
    main_camera_matrix = np.array([[908.1308, 0, 655.7268], [0, 910.0818, 395.8856], [0, 0, 1]], dtype=np.float32)
    main_camera_dist_coeffs = np.array([0.1068, -0.2123, -0.0092, 0.0000, 0.0000], dtype=np.float32)
    from cameras.realsense_env import RealSenseEnv
    import time
    env = RealSenseEnv(camera_name="main", serial_number="339322073638", camera_param = CameraParam(main_camera_matrix, main_camera_dist_coeffs))
    frame = None
    env.start_monitoring()

    while frame is None:
        time.sleep(0.11)
        frame = env.get_latest_frame()
    cv2.imshow("Camera Frame", frame['bgr'])
    cv2.waitKey
    image = frame['bgr']
    camera_param = CameraParam(main_camera_matrix, main_camera_dist_coeffs)
    rvec, tvec = camera_param.estimate_camera_poses(image)
    print("rvec:", rvec)
    print("tvec:", tvec)
    image_with_cube = camera_param.draw_cube_and_axis_on_image(image.copy(), cube_size=0.1)
    cv2.imshow("Image with Cube", image_with_cube)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
