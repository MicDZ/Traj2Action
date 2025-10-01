from controllers.controller_env import ControllerEnv
from robots.franky_env import FrankyEnv
from robots.robot_env import RobotEnv
import numpy as np
from gello.agents.gello_agent import GelloAgent
from common.constants import ActionSpace
import time
import threading
import logging

class GelloEnv(ControllerEnv):
    def __init__(self, robot_env: RobotEnv, port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAC91FI-if00-port0"):
        super().__init__(robot_env, "GelloController")
        # Additional initialization for GelloEnv can be added here
        try:
            self.gello_agent = GelloAgent(port=port)
        except Exception as e:
            logging.error(f"Failed to initialize GelloAgent: {e}, check if the port is correct and the GelloAgent is properly configured.")
            raise e
        self._state = {
            "joint_angles": np.zeros(7),
            "gripper": {
                    "target_position": 1.0,  # target position 0=closed, 1=open
                    "current_position": 1.0,  # current gripper position
                },
            "movement_enabled": False,
            "controller_on": True,
        }
        # Add thread lock
        self._state_lock = threading.RLock()  # Use reentrant lock
        
    def reset(self):
        # Implement reset logic specific to GelloEnv
        with self._state_lock:
            self._state["joint_angles"] = np.zeros(7)
            self._state["gripper"]["target_position"] = 1.0
            self._state["gripper"]["current_position"] = 1.0
            self._state["movement_enabled"] = False
            self._state["controller_on"] = False

    
    def _update_robot_state(self, hz=400):
        try:
            """Update robot state"""
            # Wait for joint angles to stabilize before starting control
            start_time = time.time()
            
            while not self._stop_monitoring.is_set():
                with self._state_lock:
                    delta_action = self.robot_env.get_position() - self._state["joint_angles"]
                    if any(abs(delta) > 0.4 for delta in delta_action):
                        over_threshold_joints = [i for i, delta in enumerate(delta_action) if abs(delta) > 0.4]
                        logging.warning(f"Huge delta action detected on joints {over_threshold_joints}: {delta_action[over_threshold_joints]}, retrying...")
                        time.sleep(0.5)  # Wait before retrying
                    else:
                        logging.info(f"Starting to control robot with gello...")
                        break

            # Main control loop
            while not self._stop_monitoring.is_set():
                try:
                    # Adjust reading frequency
                    time.sleep(1 / hz)
                    gripper_action = 0
                    with self._state_lock:
                        self._state["movement_enabled"] = True
                        action = self._state["joint_angles"]
                        if self._state["gripper"]["target_position"] != self._state["gripper"]["current_position"]:
                            # Update gripper position
                            self._state["gripper"]["current_position"] = self._state["gripper"]["target_position"]
                            if self._state["gripper"]["target_position"] == 1.0:
                                # self.robot_env.open_gripper(asynchronous=True)
                                gripper_action = 1.0
                            else:
                                # self.robot_env.close_gripper(asynchronous=True)
                                gripper_action = -1.0
                    if not self.robot_env.movement_enabled:
                        logging.warning("Movement is disabled. Wait for reset() to enable.")
                        continue
                    if gripper_action == 1.0:
                        logging.info("Opening gripper...")
                        self.robot_env.open_gripper(asynchronous=True)
                    elif gripper_action == -1.0:
                        logging.info("Closing gripper...")
                        self.robot_env.close_gripper(asynchronous=True)
                    self.robot_env.step(action, asynchronous=True)
                except Exception as e:
                    logging.error(f"Error in _update_robot_state control loop: {e}")
                    time.sleep(0.1)  # Add a small delay to avoid CPU thrashing on errors
        except Exception as e:
            logging.error(f"Fatal error in _update_robot_state: {e}")
        finally:
            with self._state_lock:
                self._state["movement_enabled"] = False
                self._state["controller_on"] = False
                self.robot_env.stop()
                logging.info("Robot control thread stopped.")
                
    
    def _update_internal_state(self, hz=200):
        """Update Gello internal state"""        
        while not self._stop_monitoring.is_set():
            try:
                # Adjust reading frequency
                time.sleep(1 / hz)
                action = self.gello_agent.act(obs=None)
                
                # Use lock to protect state update
                with self._state_lock:
                    self._state["joint_angles"] = action[:7]  # First 7 joint angles
                    self._state["gripper_position"] = action[7]  # The last one
                    self._state["controller_on"] = True 
                    self._state["timestamp"] = time.time()
                    if action[-1] > 0.5:
                        self._state["gripper"]["target_position"] = 0.0  # Close gripper
                    else:
                        self._state["gripper"]["target_position"] = 1.0  
                    print(self._state)                    
            except Exception as e:
                print(f"Error in _update_internal_state: {e}")
                # Can choose to continue or exit the loop
                continue

                

if __name__ == "__main__":
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # Output to console
            ]
        )

    # Example usage
    robot_env = FrankyEnv(action_space=ActionSpace.JOINT_ANGLES)
    robot_env.start_saving_state("./data")
    env = GelloEnv(robot_env=robot_env)
    time.sleep(2)  # Wait for environment initialization
    robot_env.stop_saving_state()
    print(env)
    env.start_controlling()
    time.sleep(5000)  # Let the monitoring thread run for a while
    env.stop_controlling()
    time.sleep(2)  # Wait for control thread to stop