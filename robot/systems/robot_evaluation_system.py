from re import S
from tkinter import NO

from sympy import im
from systems.robot_policy_system import RobotPolicySystem
from pathlib import Path
from common.constants import ActionSpace
import time
import threading
import logging
import numpy as np
import random
import cv2
import hex
class RobotEvaluationSystem:
    def __init__(self, save_dir: str = "./data/evaluation/0906_pickup_bottle_200epi_160k", action_space: ActionSpace = ActionSpace.EEF_VELOCITY, action_only_mode: bool = False, calibration: bool=True, port: int = 8765, host: str = "10.21.40.5"):
        self.policy_system = RobotPolicySystem(action_space=action_space, action_only_mode=action_only_mode, calibration=calibration, port=port, host=host)
        self.save_dir = save_dir
        self._monitoring_thread = None
        self._placer_thread = None
        if Path(save_dir).exists() is False:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.eval_info = {
            "name": "default_task",
            "success": False,
            "start_time": time.time()
        }
    def run(self, task_name: str = "default_task"):
        self.eval_info = {
            "name": task_name,
            "success": False,
            "start_time": time.time()
        }
        self.save_path = Path(self.save_dir) / time.strftime("%Y%m%d_%H%M%S")
        self.save_path.mkdir(parents=True, exist_ok=True)
        print("="*40)
        print(f"Saving evaluation task {self.eval_info['name']} to {self.save_path}")
        print("="*40)
        self.policy_system.main_camera.start_saving_frames(str(self.save_path))
        self.policy_system.wrist_camera.start_saving_frames(str(self.save_path))
        self.policy_system.top_camera.start_saving_frames(str(self.save_path))
        self.policy_system.robot_env.start_saving_state(str(self.save_path))
        self._monitoring_thread = threading.Thread(
                target=self._run_evaluation, 
                daemon=True,
                args=(task_name,)
            )

        self._monitoring_thread.start()

    def _run_evaluation(self, task_name: str):
        self.policy_system.run(task_name=task_name, show_image=False)



    def stop(self, success: bool = True):
        self.policy_system.stop()
        time.sleep(0.5)
        self._monitoring_thread.join(timeout=2.0)
        self.policy_system.main_camera.stop_saving_frames()
        self.policy_system.wrist_camera.stop_saving_frames()
        self.policy_system.top_camera.stop_saving_frames()
        self.policy_system.robot_env.stop_saving_state()
        self.eval_info["success"] = success
        self.eval_info["end_time"] = time.time()
        import json
        with open(self.save_path / "eval_info.json", "w") as f:
            json.dump(self.eval_info, f, indent=4)
        with self.policy_system.all_action_and_traj_lock:
            with open(self.save_path / "eval_data.json", "w") as f:
                json.dump(self.policy_system.all_action_and_traj, f, indent=4)
    
    def reset_for_collection(self):
        success = False
        while not success:
            success = self.policy_system.reset_for_collection()

        

if __name__ == "__main__":
    system = RobotEvaluationSystem()
    # system.run(task_name="pick up the water bottle")
    system._run_evaluation_placer()
    key = input("Press Enter to stop...\n")
    if key == "f":
        system.stop(success=False)
    else:
        system.stop(success=True)