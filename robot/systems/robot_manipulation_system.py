from robots.franky_env import FrankyEnv
from controllers.gello_env import GelloEnv
from controllers.spacemouse_env import SpaceMouseEnv
from cameras.realsense_env import RealSenseEnv
from cameras.usb_env import USBEnv
from common.constants import ActionSpace
import time
from pathlib import Path
import logging
from cameras.camera_param import CameraParam
from robots.robot_param import RobotParam
import numpy as np

class RobotManipulationSystem:
    def __init__(self, control_type: str = "spacemouse", save_dir: str = "./data/", camera_calib_file: str = "./calib"):
        # Initialize robot environment
        if control_type == "spacemouse":
            self.robot_env = FrankyEnv(action_space=ActionSpace.EEF_VELOCITY,
                                       robot_param=RobotParam(np.array([ 0.0, 0.0, -np.pi / 2]), np.array([ 0.53433071, 0.52905707, 0.00440881])))
            self.controller_env = SpaceMouseEnv(robot_env=self.robot_env)
        elif control_type == "gello":
            self.robot_env = FrankyEnv(action_space=ActionSpace.JOINT_ANGLES,
                                       robot_param=RobotParam(np.array([ 0.0, 0.0, -np.pi / 2]), np.array([ 0.53433071, 0.52905707, 0.00440881])))
            self.controller_env = GelloEnv(robot_env=self.robot_env)
        else:
            NotImplementedError(f"Control type '{control_type}' is not supported.")
        self.reset_for_collection()

        self.save_dir = save_dir
        self.save_path = None
        self.camera_envs = [
            RealSenseEnv(camera_name="main_image", serial_number="339322073638", width=1280, height=720, 
                         camera_param=CameraParam(np.array([[908.1308, 0, 655.7268], [0, 910.0818, 395.8856], [0, 0, 1]], dtype=np.float32), np.array([0.1068, -0.2123, -0.0092, 0.0000, 0.0000], dtype=np.float32))),
            USBEnv(camera_name="top_image", serial_number="12", width=1920, height=1080, exposure=100,
                        camera_param=CameraParam(np.array([[1158.0, 0, 999.9484], [0, 1159.9, 584.2338], [0, 0, 1]], dtype=np.float32), np.array([0.0412, -0.0509, 0.0000, 0.0000, 0.0000], dtype=np.float32))
                     ),
            RealSenseEnv(camera_name="wrist_image", serial_number="342222072092", width=1280, height=720),
        ]
        
        # self.camera_envs[0].calib_camera()
        # self.camera_envs[1].calib_camera()
        self.camera_envs[0].camera_param.load_from_file(camera_name="main_image", file_dir="./calib")
        self.camera_envs[1].camera_param.load_from_file(camera_name="top_image", file_dir="./calib")
        self.task_info = {
            "name": "default_task",
            "success": False,
            "start_time": time.time(),
            "end_time": None,
        }
    def reset_for_collection(self):
        success = False
        while not success:
            success = self.robot_env.reset()

        success = False
        while not success:
            action = np.concatenate((np.random.rand(2) * 0.3 - 0.15, -np.random.rand(1) * 0.2 , np.random.rand(3) * 0.3 - 0.15))
            success = self.robot_env.step(action, asynchronous=False)
            eef_pose = self.robot_env.get_position(action_space=ActionSpace.EEF_POSE)
            print(f"random action: {action}")
            if not success:
                logging.warning("Reset action failed, retrying...")
                self.robot_env.reset()
                continue
            print("eef_pose after reset:", eef_pose)
            # delta_z = eef_pose[2] - 0.05 * np.random.rand(1)[0] - 0.02
            # action = np.array([0.0, 0.0, -delta_z, 0.0, 0.0, 0.0])
            # success = self.robot_env.step(action, asynchronous=False)
            # if not success:
            #     logging.warning("Reset action failed, retrying...")
            #     self.robot_env.reset()
        gripper_action = np.random.rand(1)[0] > 0.5
        if gripper_action:
            self.robot_env.close_gripper(asynchronous=False)
        else:
            self.robot_env.open_gripper(asynchronous=False)
    def _run_all_cameras(self):
        """Start monitoring and saving for all cameras"""
        for camera_env in self.camera_envs:
            camera_env.start_monitoring()
            camera_env.start_saving_frames(str(self.save_path))
    def _stop_all_cameras(self):
        """Stop monitoring and saving for all cameras"""
        for camera_env in self.camera_envs:
            camera_env.stop_saving_frames()
            camera_env.stop_monitoring()
    def run(self, task_name: str = "default_task"):
        # 启动相机监控和保存
        self.task_info["name"] = task_name
        self.task_info["start_time"] = time.time()
        self.save_path = Path(self.save_dir) / time.strftime("%Y%m%d_%H%M%S")
        self.save_path.mkdir(parents=True, exist_ok=True)
    
        self.robot_env.robot_param.save_to_file(str(self.save_path))
        self.camera_envs[0].camera_param.save_to_file(str(self.save_path))


        self.controller_env.start_controlling()
        while not self.controller_env.get_state()["movement_enabled"]:
            time.sleep(0.11)
            logging.warning("Waiting for controller to enable movement...")
        self._run_all_cameras()
        self.robot_env.start_saving_state(str(self.save_path))
        self.controller_env.start_saving_state(str(self.save_path))
        self.task_name = task_name


    def stop(self, success: bool = True):
        self.task_info["success"] = success
        self.controller_env.stop_controlling()
        self.robot_env.stop_saving_state()
        self.controller_env.stop_saving_state()
        self._stop_all_cameras()
        self.task_info["end_time"] = time.time()
        # Save task information
        task_info_path = self.save_path / "task_info.json"
        import json
        with open(task_info_path, 'w') as f:
            json.dump(self.task_info, f, indent=4)
        logging.info(f"Task info saved to {task_info_path}")
        if not success:
            logging.error("Task ended with failure.")
            fail_path = self.save_path.parent / "failed_tasks" 
            if not fail_path.exists():
                fail_path.mkdir(parents=True, exist_ok=True)
            # move the whole dir to faile_path
            import shutil
            shutil.move(str(self.save_path), str(fail_path))
            logging.info(f"Task data moved to {fail_path}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Output to console
        ]
    )
    system = RobotManipulationSystem(control_type="spacemouse", save_dir="./data/fruits_and_plates/")
    key = input("Press Enter to start...")
    
    if key == 'q':
        system.run("pick up the tomato and put it in the blue tray")
    else:
        system.run("pick up the tomato and put it in the blue tray")

    
    key = input("Press Enter to stop...")
    if key == "f":
        system.stop(success=False)
    else:
        system.stop()
    input()
    system.controller_env.stop_controlling()
