import numpy as np
from abc import abstractmethod
from enum import Enum
from common.constants import ActionSpace
import threading
import json
import logging
import time
from robots.robot_param import RobotParam
class RobotEnv:
    def __init__(self, initial_position: np.ndarray, action_space: ActionSpace, robot_name: str = "FrankaEmika", robot_param: RobotParam = None):
        self.initial_position = initial_position
        self.action_space = action_space
        self.robot_name = robot_name
        self.robot_param = robot_param
        self.movement_enabled = False
        self._saving_thread = None
        self._stop_saving = threading.Event()
        self.gripper_state = 1.0
        self._gripper_state_lock = threading.RLock()
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action: np.ndarray, action_space: ActionSpace):
        pass

    @abstractmethod
    def get_position(self, action_space: ActionSpace) -> np.ndarray:
        pass
    
    @abstractmethod
    def stop(self):
        pass
    
    @abstractmethod
    def get_gripper_width(self) -> float:
        pass
    
    def get_gripper_state(self) -> float:
        return self.gripper_state

    def start_saving_state(self, file_path: str):
        """Start a thread to save the robot state periodically."""
        if self._saving_thread is None or not self._saving_thread.is_alive():
            self._stop_saving.clear()
            self._saving_thread = threading.Thread(
                target=self._save_state_periodically, 
                args=(file_path,),
                daemon=True
            )
            self._saving_thread.start()
    def stop_saving_state(self):
        """Stop the state saving thread."""
        self._stop_saving.set()
        time.sleep(0.11)
        if self._saving_thread and self._saving_thread.is_alive():
            self._saving_thread.join(timeout=1.0)
            self._saving_thread = None

    @abstractmethod
    def _save_state_periodically(self, file_path: str, fps: float = 120):
        states = []
        import os
        from pathlib import Path
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        json_path = Path(file_path) / f"{self.robot_name}_states.json"
        logging.info(f"Starting to save robot state to {json_path} at {fps} FPS.")
        while not self._stop_saving.is_set():
            state = self._saving_state().copy()
            state = {k: np.round(v, 5) if isinstance(v, np.ndarray) else v for k, v in state.items()}
            states.append(state)

            time.sleep(1 / fps)
        
        with open(json_path, 'w') as f:
            json.dump(states, f, separators=(',', ':'))
        logging.info(f"Robot state saved to {json_path}")
            

    @abstractmethod
    def _saving_state(self) -> dict:
        pass
    
    @abstractmethod
    def stop(self):
        pass

    def __del__(self):
        """Ensure the saving thread is stopped when the object is deleted."""
        self.stop_saving_state()

    def __str__(self):
        return f"RobotEnv(type={self.robot_type}, action_space={self.action_space.name})"
    

if __name__ == "__main__":
    env = RobotEnv(initial_position=np.zeros(3), action_space=ActionSpace.EEF_POSE)
    print(env)