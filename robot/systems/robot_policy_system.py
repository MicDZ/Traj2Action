from robots.franky_env import FrankyEnv
from controllers.gello_env import GelloEnv
from controllers.spacemouse_env import SpaceMouseEnv
from cameras.realsense_env import RealSenseEnv
from cameras.usb_env import USBEnv

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
import threading

class RobotPolicySystem:
    def __init__(self, action_space: ActionSpace = ActionSpace.JOINT_ANGLES, ip: str = "10.21.40.5", port: str = "8003", 
                 action_only_mode: bool = False, calibration: bool=True):
        # 初始化机器人环境
        self.action_space = action_space
        self.action_only_mode = action_only_mode

        self.robot_env = FrankyEnv(action_space=action_space, inference_mode=True, robot_param=RobotParam(np.array([ 0.0, 0.0, -math.pi / 2]), np.array([ 0.53433071, 0.52905707, 0.00440881])))
        if self.action_space not in [ActionSpace.EEF_VELOCITY, ActionSpace.JOINT_ANGLES]:
            raise NotImplementedError(f"Action space '{self.action_space}' is not supported.")
        logging.info(f"Trying to connect to policy server at {ip}:{port}...")
        self.client = WebsocketClientPolicy(
            host= ip,
            port= port
        )
        logging.info(f"Connected to policy server at {ip}:{port}.")
        
        self.main_camera = RealSenseEnv(camera_name="main_image", serial_number="339322073638", width=1280, height=720,
                                        camera_param=CameraParam(intrinsic_matrix = np.array([[908.1308, 0, 655.7268], [0, 910.0818, 395.8856], [0, 0, 1]], dtype=np.float32),
                                                                 distortion_coeffs = np.array([0.1068, -0.2123, -0.0092, 0.0000, 0.0000], dtype=np.float32)))
        self.wrist_camera = RealSenseEnv(camera_name="wrist_image", serial_number="342222072092", width=1280, height=720)
        
        # 只在非action_only模式下初始化top_camera
        
        self.top_camera = USBEnv(camera_name="top_image", serial_number="12", width=1920, height=1080, exposure=100,
                        camera_param=CameraParam(np.array([[1158.0, 0, 999.9484], [0, 1159.9, 584.2338], [0, 0, 1]], dtype=np.float32), np.array([0.0412, -0.0509, 0.0000, 0.0000, 0.0000], dtype=np.float32))
                    )
        if calibration:
            self.main_camera.calib_camera()
            self.top_camera.calib_camera()

        self.gripper_status = {
            "current_state": 0,
            "target_state": 0 
        }
        self.stop_evaluation = threading.Event()
        self.all_action_and_traj = []
        self.all_action_and_traj_lock = threading.Lock()

    def reset_for_collection(self):
        """Reset robot to random position for data collection"""
        success = False



    def run(self, show_image: bool = False, task_name: str = "default_task"):
        self.main_camera.start_monitoring()
        self.wrist_camera.start_monitoring()
        self.top_camera.start_monitoring()
        self.robot_env.step(np.array([0.01,0.01, -0.02,0,0,0]), asynchronous=False)

        self.gripper_status = {
            "current_state": 0,
            "target_state": 0 
        }
        self.stop_evaluation.clear()
        all_action_and_traj = []
        while not self.stop_evaluation.is_set():
            main_image = self.main_camera.get_latest_frame()['bgr']
            wrist_image = self.wrist_camera.get_latest_frame()['bgr']
            top_image = self.top_camera.get_latest_frame()['bgr']

            if main_image is None or wrist_image is None:
                time.sleep(0.05)
                continue
                
            joint_angles = self.robot_env.get_position(action_space=ActionSpace.JOINT_ANGLES)
            gripper_width = self.robot_env.get_gripper_width()
            eef_pose = self.robot_env.get_position(action_space=ActionSpace.EEF_POSE)
            state = np.concatenate([eef_pose, [gripper_width]])
            # 根据模式选择不同的处理逻辑
            if self.action_only_mode:
                state_trajectory = eef_pose[:3]
                element = {
                    "observation/image": main_image,
                    "observation/wrist_image": wrist_image,
                    "observation/state": state,
                    "prompt": task_name,
                }
            else:
                state_trajectory = self.robot_env.robot_param.transform_to_world(np.array([eef_pose[:3]]))[0]
                element = {
                    "observation/image": main_image,
                    "observation/wrist_image": wrist_image,
                    "observation/state": state,
                    "observation/state_trajectory": state_trajectory,
                    "prompt": task_name,
                }

            inference_results = self.client.infer(element)
            actions_chunk = np.array(inference_results["actions"])
            
            if not self.action_only_mode:
                trajectory_chunk = np.array(inference_results["trajectory"])
            all_action_and_traj.append({
                'actions': actions_chunk.tolist(),
                'trajectory': trajectory_chunk.tolist() if not self.action_only_mode else None,
                'timestamp': time.time(),
                'state': state.tolist(),
                'state_trajectory': state_trajectory.tolist() if not self.action_only_mode else None
            }.copy())
            with self.all_action_and_traj_lock:
                self.all_action_and_traj = all_action_and_traj

            cnt = 0
            
            if show_image:
                draw_main_image = main_image.copy()
                
                
                draw_top_image = top_image.copy()
                
                action_trajectory = 0.1 * np.cumsum(actions_chunk,axis=0)
                action_trajectory_in_world = self.robot_env.robot_param.transform_to_world(action_trajectory[:,:3] + eef_pose[:3])
                if not self.action_only_mode:
                    draw_main_image = self.main_camera.camera_param.draw_trajectory_on_image(draw_main_image, trajectory_chunk)
                    draw_top_image = self.top_camera.camera_param.draw_trajectory_on_image(draw_top_image, trajectory_chunk)

                draw_main_image = self.main_camera.camera_param.draw_trajectory_on_image(draw_main_image, action_trajectory_in_world)
                draw_top_image = self.top_camera.camera_param.draw_trajectory_on_image(draw_top_image, action_trajectory_in_world)
                cv2.imshow("Top Camera", draw_top_image)
                
                cv2.imshow("Main Camera", draw_main_image)
                cv2.imshow("Wrist Camera", wrist_image)
                cv2.waitKey(1)

            for action in actions_chunk:
                self.robot_env.step(action[:-1], asynchronous=True)
                time.sleep(0.1)

                cnt += 1
                gripper_action = action[-1]
                

                if gripper_action > 0.95:
                    self.gripper_status["target_state"] = 1
                elif gripper_action < -0.95:
                    self.gripper_status["target_state"] = -1
                
                if self.gripper_status["current_state"] != self.gripper_status["target_state"]:
                    if self.gripper_status["target_state"] == -1:
                        self.robot_env.open_gripper(asynchronous=True)
                    else:
                        self.robot_env.close_gripper(asynchronous=True)
                    self.gripper_status["current_state"] = self.gripper_status["target_state"]
                
                max_cnt = 10

                if cnt == max_cnt:
                    self.robot_env.step(np.array([0,0,0,0,0,0]), asynchronous=False)
                    break
    def stop(self):
        self.stop_evaluation.set()
        time.sleep(0.5)

        self.robot_env.stop_saving_state()
        logging.info("Robot policy system stopped.")

if __name__ == "__main__":
    system = RobotPolicySystem(action_space=ActionSpace.EEF_VELOCITY, action_only_mode=True, prompt="pick up the water bottle", calibration=True)
    system.run(show_image=True)