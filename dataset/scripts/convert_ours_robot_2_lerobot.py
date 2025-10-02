import cv2
from pathlib import Path
import os
import sys
from policy.lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import logging
import numpy as np
import cv2


#################################################################################
RAW_DATA_DIR_PATH = "SET_YOUR_RAW_DATA_DIR_PATH_DOWNLOADED_FROM_HUGGINGFACE_HERE"
OUTPUT_BASE = "SET_YOUR_OUTPUT_BASE_PATH_HERE"
REPO_NAME = "robot_dataset"
#################################################################################


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


ROBOT_STATS = "FrankaEmika_states.json"
IMAGE_TIMESTAMPS = [
    "main_image_timestamps.json",
    "wrist_image_timestamps.json"
]
IMAGE_VIDEO = [
    "main_image.mp4",
    "wrist_image.mp4"
]
CONTROLLER_STATS = "SpaceMouseController_states.json"
TASK_INFO = "task_info.json"
ROBOT_PARAM = "robot_param.json"
CAMERA_PARAM = [
    "camera_param.json",
]


class StatesIterator:
    def __init__(self, states: list):
        self.states = states
        self.index = 0
        self.length = len(states)
    
    def reset(self):
        self.index = 0
    
    def get_next_state(self, timestamp: float):
        while self.index < self.length and self.states[self.index]['timestamp'] < timestamp:
            self.index += 1
        # cal diff
        time_diff = abs(self.states[self.index - 1]['timestamp'] - timestamp) if self.index > 0 else float('inf')
        if self.index == 0:
            return self.states[0]
        if self.index >= self.length:
            return self.states[-1]
        return self.states[self.index - 1]
    def sample_state(self, num: int, fps: float = 30, current_time: float = None):
        sampled_states = []
        interval = 1.0 / fps
        index = self.index
        current_time = self.states[self.index]['timestamp'] if current_time is None else current_time
        while len(sampled_states) < num:
            while index < self.length and self.states[index]['timestamp'] < current_time:
                index += 1
            if index == 0:
                state = self.states[0]
            elif index >= self.length:
                state = self.states[-1]
            else:
                state = self.states[index - 1]
            sampled_states.append(state)
            current_time += interval
        return sampled_states

output_path = Path(OUTPUT_BASE) / REPO_NAME

dataset = LeRobotDataset.create(
                repo_id=output_path,
                robot_type="franka",
                fps=10,
                features={
                    "image": {
                        "dtype": "image",
                        "shape": (720, 1280, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "wrist_image": {
                        "dtype": "image",
                        "shape": (720, 1280, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "state": {
                        "dtype": "float64",
                        "shape": (8,), 
                        "names": ["state"],
                    },
                    "actions": {
                        "dtype": "float64",
                        "shape": (7,),
                        "names": ["actions"],
                    },
                    "trajectory": {
                        "dtype": "float64",
                        "shape": (3,),  
                        "names": ["trajectory"],
                    },
                    "state_trajectory": {
                        "dtype": "float64",
                        "shape": (3,), 
                        "names": ["state_trajectory"],
                    },
                },
                image_writer_threads=20,
                image_writer_processes=20,
            )
class DataReplayer:
    def __init__(self, data_path: str, fps: float = 30):
        self.data_path = Path(data_path)
        self.fps = fps
        self.robot_states = []
        self.controller_states = []
        self.task_info = {}
        self.robot_param = None
        self.camera_params = {
            "main_image": None,
        }
        self.cams = {
            "main_image": None,
            "wrist_image": None
        }
        self.image_timestamps = {
            "main_image": [],
            "wrist_image": []
        }
        self.load_data()
        self.robot_states_iterator = StatesIterator(self.robot_states)
        self.controller_states_iterator = StatesIterator(self.controller_states)
        self.start_timestamp = self.check_max_timestamp()

    def load_data(self):
        import json
        # Load robot states
        robot_stats_path = self.data_path / ROBOT_STATS
        if robot_stats_path.exists():
            with open(robot_stats_path, 'r') as f:
                self.robot_states = json.load(f)
        else:
            raise FileNotFoundError(f"Robot stats file not found at {robot_stats_path}")
        
        # Load controller states
        controller_stats_path = self.data_path / CONTROLLER_STATS
        if controller_stats_path.exists():
            with open(controller_stats_path, 'r') as f:
                self.controller_states = json.load(f)
        else:
            raise FileNotFoundError(f"Controller stats file not found at {controller_stats_path}")
        # Load task info
        task_info_path = self.data_path / TASK_INFO
        if task_info_path.exists():
            with open(task_info_path, 'r') as f:
                self.task_info = json.load(f)
        else:
            raise FileNotFoundError(f"Task info file not found at {task_info_path}")
        
        # Load robot param
        robot_param_path = self.data_path / ROBOT_PARAM
        if robot_param_path.exists():
            self.robot_param = RobotParam()
            self.robot_param.load_from_file(str(self.data_path))
        else:
            raise FileNotFoundError(f"Robot param file not found at {robot_param_path}")
        
        for ts_file in IMAGE_TIMESTAMPS:
            ts_path = self.data_path / ts_file
            if ts_path.exists():
                with open(ts_path, 'r') as f:
                    self.image_timestamps[str(ts_file).replace("_timestamps.json", "")] = json.load(f)
            else:
                raise FileNotFoundError(f"Image timestamps file not found at {ts_path}")
        
        for cam_file in IMAGE_VIDEO:
            cam_path = self.data_path / cam_file
            if cam_path.exists():
                cap = cv2.VideoCapture(str(cam_path))
                if not cap.isOpened():
                    raise IOError(f"Cannot open video file {cam_path}")
                self.cams[cam_file.replace(".mp4", "")] = cap
            else:
                raise FileNotFoundError(f"Camera video file not found at {cam_path}")
        # check if timestamps length match video frames
        for cam_name in self.cams:
            if self.cams[cam_name] is not None:
                frame_count = int(self.cams[cam_name].get(cv2.CAP_PROP_FRAME_COUNT))
                ts_count = len(self.image_timestamps[cam_name])
                if frame_count != ts_count:
                    raise ValueError(f"Frame count {frame_count} does not match timestamp count {ts_count} for camera {cam_name}")
                
    def check_max_timestamp(self):
        max_timestamp = 0
        
        if self.robot_states[0]['timestamp'] > max_timestamp:
            max_timestamp = self.robot_states[0]['timestamp']
        if self.controller_states[0]['timestamp'] > max_timestamp:
            max_timestamp = self.controller_states[0]['timestamp']
        for cam in self.image_timestamps:
            if len(self.image_timestamps[cam]) > 0 and self.image_timestamps[cam][0] > max_timestamp:
                max_timestamp = self.image_timestamps[cam][0]
        return max_timestamp

    def replay(self):
        frame_index = 0
        last_point = None
        if "yellow" in self.task_info['name']:
            print("skip yellow tray")
            return
        while True:
            frame = {
                "image": None,
                "wrist_image": None,
                "state": None,
                "actions": None,
                "trajectory": None,
                "state_trajectory": None
            }
            if frame_index >= len(self.image_timestamps["main_image"]) or frame_index >= len(self.image_timestamps["wrist_image"]):
                break

            frame_time_diff = self.image_timestamps["main_image"][frame_index] - self.image_timestamps["wrist_image"][frame_index]
            frame_average_time = self.image_timestamps["main_image"][frame_index] 
            robot_states = self.robot_states_iterator.get_next_state(frame_average_time)
            controller_states = self.controller_states_iterator.get_next_state(frame_average_time)

            frame["state"] = np.array(robot_states['eef_pose'] + [robot_states['gripper_width']])
            robot_states_world = robot_states['eef_pose'][:3]
            robot_states_world = self.robot_param.transform_to_world(np.array(robot_states_world)).flatten()
            frame["state_trajectory"] = np.array(robot_states_world[:3])
            frame["actions"] = np.array(controller_states['action'] + [controller_states['gripper']['target_position']])
            frame["trajectory"] = np.array(robot_states_world[:3]) 

            if abs(frame_time_diff) > 0.060:
                logging.error(f"Frame time difference too large: {frame_time_diff}")
                break
            frame_index += 1
            for cam_name, cam in self.cams.items():
                if cam is None:
                    raise ValueError(f"Camera {cam_name} is not loaded properly.")
                ret, image = cam.read()
                if not ret:
                    print("End of video stream")
                    return
                if cam_name == "main_image":
                    frame["image"] = image
                elif cam_name == "wrist_image":
                    frame["wrist_image"] = image
                else:
                    raise ValueError(f"Unknown camera name {cam_name}")
            if frame_index % 3 != 0:
                continue
            last_point = robot_states_world[:3]

            dataset.add_frame(
                task= self.task_info['name'],
                frame=frame
            )
        dataset.save_episode()
        

            
                    
if __name__ == "__main__":
    dir = "RAW_DATA_DIR_PATH"
    import tqdm
    files = os.listdir(dir)
    total_num = len(files)
    import random
    random.seed(0)
    selected_files = random.sample(files, total_num)
    for subdir in tqdm.tqdm(selected_files):
        try:
            data_replay = DataReplayer(Path(dir) / subdir)
            data_replay.replay()
        except Exception as e:
            print(f"Error processing {subdir}: {e}")

        
