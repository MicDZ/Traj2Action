from enum import Enum

class ActionSpace(Enum):
    EEF_POSE = 1
    EEF_VELOCITY = 2
    JOINT_ANGLES = 3
    JOINT_VELOCITIES = 4