import numpy as np
from abc import abstractmethod
from enum import Enum
from common.constants import ActionSpace
from robots.robot_env import RobotEnv
import threading
import logging
import time
import json
import copy

class ControllerEnv:
    def __init__(self, robot_env: RobotEnv, controller_name: str):
        self.initial_position = robot_env.initial_position
        self.controller_name = controller_name
        self.action_space = robot_env.action_space
        self.robot_name = robot_env.robot_name
        self.robot_env = robot_env
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._stop_controlling = threading.Event()  
        self._control_thread = None
        self._state = {
            "movement_enabled": False,
            "controller_on": False,
        }
        self._state_lock = threading.RLock()  
        self._saving_thread = None
        self._stop_saving = threading.Event()

    @abstractmethod
    def reset(self):
        pass

    def start_monitoring(self):
        # Implement monitoring logic specific to GelloEnv
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(
                target=self._update_internal_state, 
                args=(200,),
                daemon=True
            )
            self._monitoring_thread.start()

    def start_controlling(self):
        """Start control thread"""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self.start_monitoring()
        
        with self._state_lock:
            self._state["movement_enabled"] = False
            self._state["controller_on"] = False
            self._state["gripper"] = {
                "target_position": self.robot_env.get_gripper_state(),  # Target position 1=closed, -1=open
                "current_position": self.robot_env.get_gripper_state(),  # Current gripper position
            }
        if self._control_thread is None or not self._control_thread.is_alive():
            self._stop_controlling.clear()  # Use separate event
            self._control_thread = threading.Thread(
                target=self._update_robot_state, 
                args=(200,),
                daemon=True
            )
            self._control_thread.start()

    def stop_controlling(self):
        """Stop control thread"""
        self._stop_controlling.set()  # Use separate event
        self.stop_monitoring()
        
        if self._control_thread and self._control_thread.is_alive():
            try:
                self._control_thread.join(timeout=2.0)
                if self._control_thread.is_alive():
                    logging.warning("Control thread did not terminate properly")
            except Exception as e:
                logging.error(f"Error stopping control thread: {e}")
            
        with self._state_lock:
            self._state["movement_enabled"] = False
            self._state["controller_on"] = False

    def stop_monitoring(self):
        """Stop monitoring thread"""
        self._stop_monitoring.set()
        time.sleep(0.11)
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=1.0)

    def start_saving_state(self, file_path: str):
        """Start a thread to save the robot state periodically."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            logging.warning("Saving state thread is not started because monitoring thread is not running.")
            return
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
        if self._saving_thread and self._saving_thread.is_alive():
            logging.info("Stopping controller state saving thread...")
            self._stop_saving.set()
            try:
                self._saving_thread.join(timeout=2.0)
                if self._saving_thread.is_alive():
                    logging.warning("Saving thread did not terminate properly")
            except Exception as e:
                logging.error(f"Error stopping saving thread: {e}")
            self._saving_thread = None
    def _round_nested_values(self, obj, decimals=5):
        """Recursively round all float values in nested dictionaries and lists."""
        if isinstance(obj, float):
            return round(obj, decimals)
        elif isinstance(obj, np.ndarray):
            return np.round(obj, decimals).tolist()
        elif isinstance(obj, dict):
            return {k: self._round_nested_values(v, decimals) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._round_nested_values(item, decimals) for item in obj]
        else:
            return obj
    def _save_state_periodically(self, file_path: str, fps: float = 120):
        states = []
        import os
        from pathlib import Path
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        json_path = Path(file_path) / f"{self.controller_name}_states.json"
        logging.info(f"Starting to save controller state to {json_path} at {fps} FPS.")
        while not self._stop_saving.is_set():
            state = self.get_state()
            # Keep all values to 5 decimal places
            state = self._round_nested_values(state, decimals=5)
            states.append(state)
            time.sleep(1 / fps)

        with open(json_path, 'w') as f:
            json.dump(states, f, separators=(',', ':'))
        logging.info(f"Controller state saved to {json_path}")

    def get_state(self):
        """Thread-safe get state"""
        with self._state_lock:
            # Return deep copy of state to avoid external modification
            return copy.deepcopy(self._state)
    
    def set_movement_enabled(self, enabled: bool):
        """Thread-safely set motion enable"""
        with self._state_lock:
            self._state["movement_enabled"] = enabled
    
    def set_controller_on(self, on: bool):
        """Thread-safely set controller status"""
        with self._state_lock:
            self._state["controller_on"] = on

    @abstractmethod
    def _update_internal_state(self, num_wait_sec=5, hz=1000):
        """Update internal state"""
        pass

    @abstractmethod
    def _update_robot_state(self, hz=400):
        """Update robot state"""
        pass
    
    def __str__(self):
        return f"ControllerEnv(type={self.robot_name}, action_space={self.action_space.name})"
    


if __name__ == "__main__":
    # Example usage
    robot_env = RobotEnv(initial_position=np.zeros(3), action_space=ActionSpace.EEF_POSE)
    env = ControllerEnv(robot_env=robot_env)
    print(env)