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

class RobotPolicySystem:
    def __init__(self, action_space: ActionSpace = ActionSpace.JOINT_ANGLES, ip: str = "10.21.40.5", port: str = "8003", camera_calib_file: str = "./calib"):
        # 初始化机器人环境
        self.action_space = action_space

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
        # self.main_camera.calib_camera()
        if camera_calib_file:
            self.main_camera.camera_param.load_from_file(camera_calib_file)
        else:
            self.main_camera.calib_camera()
        self.gripper_status = {
            "current_state": 0,
            "target_state": 0 
        }
        # self.reset_for_collection()
    def reset_for_collection(self):
        success = False
        while not success:
            success = self.robot_env.reset()

        success = False
        while not success:
            action = np.concatenate((np.random.rand(2) * 0.3 - 0.3, -np.random.rand(1) * 0.2 , np.random.rand(3) * 0.3 - 0.3))
            success = self.robot_env.step(action, asynchronous=False)
            eef_pose = self.robot_env.get_position(action_space=ActionSpace.EEF_POSE)
            if not success:
                logging.warning("Reset action failed, retrying...")
                self.robot_env.reset()
                continue

        gripper_action = np.random.rand(1)[0] > 0.5
        if gripper_action:
            self.robot_env.close_gripper(asynchronous=False)
        else:
            self.robot_env.open_gripper(asynchronous=False)
    def run(self, show_image: bool = False):
        self.main_camera.start_monitoring()
        self.wrist_camera.start_monitoring()
        self.robot_env.step(np.array([0.0,0.0, -0.00,0,0,0]), asynchronous=False)
        self.gripper_status = {
            "current_state": 0,
            "target_state": 0 
        }
        while True:
            main_image = self.main_camera.get_latest_frame()['bgr']
            wrist_image = self.wrist_camera.get_latest_frame()['bgr']
            # bgr 2 rgb
            
            if main_image is None or wrist_image is None:
                time.sleep(0.05)
                continue
            joint_angles = self.robot_env.get_position(action_space=ActionSpace.JOINT_ANGLES)
            gripper_width = self.robot_env.get_gripper_width()
            print("gripper_width:", gripper_width)
            eef_pose = self.robot_env.get_position(action_space=ActionSpace.EEF_POSE)
            state = np.concatenate([eef_pose, [gripper_width]])

            state_trajectory = eef_pose[:3]
            # prompt = "pick up the water bottle"
            prompt = "pick up the tomato and put it in the yellow tray"
            element = {
                "observation/image": main_image,
                "observation/wrist_image": wrist_image,
                "observation/state": state,
                "prompt": prompt,
            }

            inference_results = self.client.infer(element)
            actions_chunk = np.array(inference_results["actions"])
            cnt = 0
            draw_main_image = main_image.copy()
            
            cv2.imshow("Main Camera", draw_main_image)
            cv2.imshow("Wrist Camera", wrist_image)
            cv2.waitKey(1)
            print(actions_chunk[:7][:,-1,])
            for action in actions_chunk:
                self.robot_env.step(action[:-1], asynchronous=True)
                time.sleep(0.1)

                cnt += 1
                gripper_action = action[-1]
                if gripper_action > 0.95:
                    self.gripper_status["target_state"] = 1
                elif gripper_action < -0.95:
                    self.gripper_status["target_state"] = -1
                print(self.gripper_status)
                if self.gripper_status["current_state"] != self.gripper_status["target_state"]:
                    if self.gripper_status["target_state"] == -1:
                        self.robot_env.open_gripper(asynchronous=True)
                    else:
                        # self.robot_env.stop()
                        self.robot_env.close_gripper(asynchronous=True)
                        time.sleep(0.5)
                    self.gripper_status["current_state"] = self.gripper_status["target_state"]
                if cnt == 6:
                    # self.robot_env.stop()
                    self.robot_env.step(np.array([0,0,0,0,0,0]), asynchronous=False)
                    break

if __name__ == "__main__":
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # 输出到控制台
            ]
        )
    system = RobotPolicySystem(action_space=ActionSpace.EEF_VELOCITY)
    system.run(show_image=True)