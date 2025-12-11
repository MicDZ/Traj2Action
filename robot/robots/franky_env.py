import numpy as np
from robots.robot_env import RobotEnv, ActionSpace
import franky
import scipy.spatial.transform as transform
import logging
import time
import multiprocessing as mp


class FrankyEnv(RobotEnv):
    def __init__(self, robot_ip: str = "172.16.0.2",
                 action_space: ActionSpace = ActionSpace.JOINT_ANGLES, 
                 initial_position: np.ndarray = np.array([-0.0, -0.0, 0.0, -1.57, 0.0, 1.57, 0.785037]),
                 robot_param = None,
                 inference_mode: bool = False
                 ):
        super().__init__(initial_position, action_space, robot_name="FrankaEmika",
                         robot_param=robot_param)

        self._robot = franky.Robot(robot_ip)
        self._robot.recover_from_errors()
        self._gripper = franky.Gripper(robot_ip)
        self._gripper_width_queue = mp.Queue(maxsize=1)
        self._gripper_process = mp.Process(target=self._gripper_state_worker, daemon=True)
        self._gripper_process.start()
        self._robot.relative_dynamics_factor = 0.15

        self.reset()
                
        if self.action_space == ActionSpace.JOINT_ANGLES:
            self._robot.relative_dynamics_factor = 0.25
        if self.action_space == ActionSpace.EEF_VELOCITY:
            self._robot.relative_dynamics_factor = 0.15
        if inference_mode:
            self._robot.relative_dynamics_factor = 0.05

    # https://github.com/TimSchneider42/franky/issues/35
    def _gripper_state_worker(self):
        while True:
            try:
                width = self._gripper.width
                if self._gripper_width_queue.full():
                    self._gripper_width_queue.get()
                self._gripper_width_queue.put(width)
                time.sleep(0.01)  # 50ms
            except Exception as e:
                print(f"Error in gripper state worker: {e}")
                time.sleep(0.11)

    def get_gripper_width(self):
        try:
            self._gripper_last_width = self._gripper_width_queue.get_nowait()
            return self._gripper_last_width
        except:
            return self._gripper_last_width

    def check_action(self, action: np.ndarray):
        if self.action_space == ActionSpace.EEF_POSE:
            assert action.shape == (6,), "Action must be a 6D vector for end-effector pose."
        elif self.action_space == ActionSpace.EEF_VELOCITY:
            assert action.shape == (6,), "Action must be a 6D vector for end-effector velocity."
        elif self.action_space == ActionSpace.JOINT_ANGLES:
            assert action.shape == (7,), "Action must be a 7D vector for joint angles."
        elif self.action_space == ActionSpace.JOINT_VELOCITIES:
            assert action.shape == (7,), "Action must be a 7D vector for joint velocities."
        else:
            raise ValueError("Invalid action space.")

    def reset(self, asynchronous: bool = False, homing : bool = False):
        logging.info("Resetting Franky to initial position...")
        assert self.initial_position.shape == (7,), "Initial position must be a 7D vector for joint angles."
        try:
            motion = franky.JointMotion(self.initial_position)
            self.open_gripper(asynchronous=asynchronous)
            logging.info("Gripper opened.")

            self.movement_enabled = True
            if homing:
                self._gripper.homing()
            logging.info("Robot Reset complete.")
            self._robot.stop()
            self._robot.recover_from_errors()
            self._robot.relative_dynamics_factor = 0.25
            self._robot.move(motion=motion, asynchronous=asynchronous)
            self._robot.relative_dynamics_factor = 0.15
            time.sleep(0.1)
        except Exception as e:
            logging.error(f"Error during reset: {e}")
            self._robot.recover_from_errors()
            return False
        return True

    def step(self, action: np.ndarray, asynchronous: bool = False, duration: float = 1000, relative: bool = True):
        if not self.movement_enabled:
            logging.warning("Movement is disabled. Wait for reset() to enable.")
            return
        self.check_action(action)
        if self.action_space == ActionSpace.EEF_POSE:
            NotImplementedError("EEF_POSE action space not implemented yet.")
        elif self.action_space == ActionSpace.EEF_VELOCITY:
            motion = franky.CartesianVelocityMotion(
                franky.Twist(linear_velocity=action[:3], angular_velocity=action[3:6]), 
                duration=franky.Duration(int(duration))
            )
        elif self.action_space == ActionSpace.JOINT_ANGLES:
            motion = franky.JointMotion(action)
        elif self.action_space == ActionSpace.JOINT_VELOCITIES:
            motion = franky.JointVelocityMotion(action)
        else:
            raise ValueError("Invalid action space.")
        try:
            self._robot.move(motion=motion, asynchronous=asynchronous)
        except Exception as e:
            logging.error(f"Error executing step: {e}")
            self._robot.recover_from_errors()
            return False
        return True
    def stop(self):
        # logging.info("Stopping robot...")
        self._robot.stop()
        # logging.info("Robot stopped.")    
    def close_gripper(self, asynchronous: bool = False):
        speed = 0.1  # [m/s]
        force = 35.0  # [N]
        logging.info("Closing gripper...")
        if asynchronous:
            self._gripper.grasp_async(0.02, speed, force, epsilon_inner=0.05, epsilon_outer=0.15)
        else:
            self._gripper.grasp(0.0, speed, force, epsilon_inner=0.05, epsilon_outer=0.15)
        with self._gripper_state_lock:
            self.gripper_state = 1.0

    def open_gripper(self, asynchronous: bool = False):
        speed = 0.22
        logging.info("Opening gripper...")
        if asynchronous:
            self._gripper.open_async(speed)
        else:
            self._gripper.open(speed)
        with self._gripper_state_lock:
            self.gripper_state = -1.0
    def get_position(self, action_space: ActionSpace = ActionSpace.JOINT_ANGLES) -> np.ndarray:
        if action_space == ActionSpace.JOINT_ANGLES:
            pose = self._robot.current_joint_state.position
            return pose
        elif action_space == ActionSpace.EEF_POSE:
            pose = self._robot.current_cartesian_state.pose.end_effector_pose
            return np.concatenate([pose.translation, pose.quaternion])
        elif action_space == ActionSpace.EEF_VELOCITY:
            velocity = self._robot.current_cartesian_state.velocity.end_effector_twist
            return np.concatenate((velocity.linear, velocity.angular))
        elif action_space == ActionSpace.JOINT_VELOCITIES:
            velocity = self._robot.current_joint_state.velocity
            return velocity
        
    def _saving_state(self) -> dict:
        joint_state = self._robot.current_joint_state
        cartesian_state = self._robot.current_cartesian_state
        state = {
            "joint_angles": joint_state.position.tolist(),
            "eef_pose": np.concatenate([cartesian_state.pose.end_effector_pose.translation,
                                        cartesian_state.pose.end_effector_pose.quaternion]).tolist(),
            "eef_velocity": np.concatenate([cartesian_state.velocity.end_effector_twist.linear, 
                                           cartesian_state.velocity.end_effector_twist.angular]).tolist(),
            "joint_velocities": joint_state.velocity.tolist(),
            "gripper_width": self.get_gripper_width(),
            "timestamp": time.time()
        }
        return state
    
    def __str__(self):
        return f"FrankyEnv(type={self._robot_type}, action_space={self.action_space.name})"
    
    def __del__(self):
        try:
            self.stop()
            time.sleep(3)
            logging.info("FrankyEnv instance deleted and robot stopped.")
        except Exception as e:
            logging.error(f"Error during FrankyEnv deletion: {e}")
    
if __name__ == "__main__":
    env = FrankyEnv(action_space=ActionSpace.EEF_POSE)
    print(env)
    env.step(np.array([0.03, 0.0, 0.03, 0.0, 0, 0.0]))
