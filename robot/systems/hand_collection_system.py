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
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts.hand_detection import hand_detect
from scripts.hand_pose_estimation import HandPoseEstimator
class HandCollectionSystem:
    def __init__(self, save_dir: str = "./data/", calibration: bool = True):
        # Initialize robot environment
        self.save_dir = save_dir
        self.save_path = None
        self.camera_envs = [
            RealSenseEnv(camera_name="main_image", serial_number="339322073638", width=1280, height=720, 
                         camera_param=CameraParam(np.array([[908.1308, 0, 655.7268], [0, 910.0818, 395.8856], [0, 0, 1]], dtype=np.float32), np.array([0.1068, -0.2123, -0.0092, 0.0000, 0.0000], dtype=np.float32))),
            RealSenseEnv(camera_name="wrist_image", serial_number="342222072092", width=1280, height=720,
                        camera_param=CameraParam(np.array([[909.3397, 0, 633.1042], [0, 909.0326, 360.8437], [0, 0, 1]], dtype=np.float32), np.array([0.0852, -0.1976, 0.0000, 0.0000, 0.0000], dtype=np.float32))),
            USBEnv(camera_name="top_image", serial_number="12", width=1920, height=1080, exposure=100,
                        camera_param=CameraParam(np.array([[1158.0, 0, 999.9484], [0, 1159.9, 584.2338], [0, 0, 1]], dtype=np.float32), np.array([0.0412, -0.0509, 0.0000, 0.0000, 0.0000], dtype=np.float32))
                   ),
            USBEnv(camera_name="human_image", serial_number="14", width=1920, height=1080, exposure=400)
        ]
        # None means to calibrate, file path means to load from file
        self.calib_way = {
            "main_image": None,
            "top_image": None,
            "wrist_image": None,
        }
        
        self.task_info = {
            "name": "default_task",
            "success": False,
            "start_time": time.time(),
            "end_time": None,
        }
        # use thread pool to process hand data
        self.hand_data_process_pipline = ThreadPoolExecutor(max_workers=4)
        self.calibration = calibration
        self.calibrate_camera()

    def process_hand_data(self, save_path: str):
        """处理手部数据，返回手部关键点"""
        hand_detect(save_path)
        hand_estimator = HandPoseEstimator(save_path)
        hand_estimator.replay()

    def calibrate_camera(self):
        """Calibrate specified camera"""
        for camera_env in self.camera_envs:
            if camera_env.camera_name in self.calib_way and self.calibration and self.calib_way[camera_env.camera_name] is None:
                camera_env.calib_camera()
                logging.info(f"Camera {camera_env.camera_name} calibrated.")
            elif camera_env.camera_name in self.calib_way and not self.calibration and type(self.calib_way[camera_env.camera_name]) is str:
                camera_env.camera_param.load_from_file(self.calib_way[camera_env.camera_name], camera_env.camera_name)
                logging.info(f"Camera {camera_env.camera_name} loaded calibration from file {self}.")
            elif camera_env.camera_name in self.calib_way and self.calibration and self.calib_way[camera_env.camera_name] is None:
                raise ValueError(f"Camera {camera_env.camera_name} calibration file not provided, but required.")

    def _save_all_camera_calib(self, save_path: Path):
        """Save calibration parameters for all cameras"""
        for camera_env in self.camera_envs:
            if camera_env.camera_param is not None and self.save_path is not None:
                camera_env.camera_param.save_to_file(save_path, camera_env.camera_name)
                logging.info(f"Camera {camera_env.camera_name} calibration saved to {save_path}")
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
        self._save_all_camera_calib(self.save_path)

        self._run_all_cameras()
   
        self.task_name = task_name


    def stop(self, success: bool = True):
        self.task_info["success"] = success
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
        
        self.hand_data_process_pipline.submit(self.process_hand_data, str(self.save_path))
        logging.info("Hand data processing started in background.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Output to console
        ]
    )
    system = HandCollectionSystem(save_dir="./data/test/", calibration=False)
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
