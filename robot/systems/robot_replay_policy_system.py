from robots.franky_env import FrankyEnv
from controllers.gello_env import GelloEnv
from controllers.spacemouse_env import SpaceMouseEnv
from cameras.realsense_env import RealSenseEnv
from common.constants import ActionSpace
import time
from pathlib import Path
import logging
from systems.robot_policy_utils import WebsocketClientPolicy
import cv2
import numpy as np
from cameras.camera_param import CameraParam
from robots.robot_param import RobotParam
import math
from scripts.replay_robot_data import DataReplayer


class RobotReplayPolicySystem(DataReplayer):
    def __init__(self, data_path: str, action_space: ActionSpace = ActionSpace.EEF_POSE, ip: str = "10.21.40.5", port: str = "8003"):
        super().__init__(data_path)
        # 初始化机器人环境
        self.action_space = action_space
        logging.info(f"Trying to connect to policy server at {ip}:{port}...")
        self.client = WebsocketClientPolicy(
            host= ip,
            port= port
        )
    def replay(self):
        frame_index = 0
        while True:
            if frame_index >= len(self.image_timestamps["main_image"]) or frame_index >= len(self.image_timestamps["wrist_image"]):
                break

            frame_time_diff = self.image_timestamps["main_image"][frame_index] - self.image_timestamps["wrist_image"][frame_index]
            frame_average_time = self.image_timestamps["main_image"][frame_index] 
            robot_states = self.robot_states_iterator.get_next_state(frame_average_time)
            task_info = self.task_info
            state = robot_states['eef_pose'] + [robot_states['gripper_width']]
            state_trajectory = robot_states['eef_pose'][:3]
            prompt = task_info['name']
            element = {
                "observation/image": None,
                "observation/wrist_image": None,
                "observation/state": state,
                "observation/state_trajectory": state_trajectory,
                "prompt": prompt,
            }
            traj_show_fps = 10
           
            if abs(frame_time_diff) > 0.060:
                logging.error(f"Frame time difference too large: {frame_time_diff}")
                break
            frame_index += 1
            main_image_to_show = None
            wrist_image_to_show = None
            
            for cam_name, cam in self.cams.items():
                if cam is None:
                    raise ValueError(f"Camera {cam_name} is not loaded properly.")
                ret, frame = cam.read()
                if not ret:
                    print("End of video stream")
                    return
                eef_pose = np.array(robot_states['eef_pose'])
                if cam_name == "main_image":
                    element["observation/image"] = frame
                    main_image_to_show = frame.copy()
                elif cam_name == "wrist_image":
                    element["observation/wrist_image"] = frame
                    wrist_image_to_show = frame.copy()    
            if frame_index % 10 != 0:
                continue    
            inference_results=self.client.infer(element)
            trajectory = []
            last_point = None
            print(np.array(inference_results['actions'])[:7][:,-1,])
            print(state[-1])
            for state in inference_results['actions']:
                last_point = None
                if len(trajectory) > 0:
                    last_point = trajectory[-1]
                else:
                    last_point = robot_states['eef_pose'][:3]
                trajectory.append(
                    np.array(state[:3] ) * (1/traj_show_fps) + last_point
                    )
            trajectory = np.array(trajectory)
            trajectory = self.robot_param.transform_to_world(trajectory)
            self.camera_params["main_image"].draw_trajectory_on_image(main_image_to_show, trajectory)
            cv2.imshow("Main Camera", main_image_to_show)
            cv2.imshow("Wrist Camera", wrist_image_to_show)
            cv2.waitKey(1)

        
        
if __name__ == "__main__":
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # 输出到控制台
            ]
        )
    system = RobotReplayPolicySystem("./data/20250910_stack_cups/20250910_105056", action_space=ActionSpace.EEF_VELOCITY)
    system.replay()