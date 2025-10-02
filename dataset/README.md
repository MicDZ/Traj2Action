# Traj2Action: Dataset Conversion and Preparation
This directory contains scripts and instructions for converting and preparing datasets in the LeRobot format for training robot policies. It includes tools to convert raw data into the required format, preprocess it, and organize it for efficient loading during training.

## Dataset Structure
We follow the LeRobot dataset structure, which can be easily curated using the following LeRobot code:

* Our Robot Dataset:
```python
dataset = LeRobotDataset.create(
                repo_id=output_path,
                robot_type="franka",
                fps=10,
                features={
                    "image": {
                        "dtype": "image",
                        "shape": (720, 1280, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "wrist_image": {
                        "dtype": "image",
                        "shape": (720, 1280, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "state": {
                        "dtype": "float64",
                        "shape": (8,),      # xyz + qpos + gripper width
                        "names": ["state"],
                    },
                    "actions": {
                        "dtype": "float64",
                        "shape": (7,),      # xyz + ypr + gripper width
                        "names": ["actions"],
                    },
                    "trajectory": {
                        "dtype": "float64",
                        "shape": (3,),      # xyz
                        "names": ["trajectory"],
                    },
                    "state_trajectory": {
                        "dtype": "float64",
                        "shape": (3,),      # xyz
                        "names": ["state_trajectory"],
                    },
                },
                image_writer_threads=20,
                image_writer_processes=20,
            )
```

* Our Hand Dataset:
```python
dataset = LeRobotDataset.create(
                repo_id=output_path,
                robot_type="franka",
                fps=30,
                features={
                    "main_image": {
                        "dtype": "image",
                        "shape": (720, 1280, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "human_image": {
                        "dtype": "image",
                        "shape": (1080, 1920, 3),
                        "names": ["height", "width", "channel"],
                    },
                    "state": {
                        "dtype": "float64",
                        "shape": (3,),    # xyz
                        "names": ["state"],
                    },
                    "actions": {
                        "dtype": "float64",
                        "shape": (3,),     # xyz, not used in training
                        "names": ["actions"],
                    },
                    "trajectory": {
                        "dtype": "float64",
                        "shape": (3,),     # xyz
                        "names": ["trajectory"],
                    },
                    "state_trajectory": {
                        "dtype": "float64",
                        "shape": (3,),     # xyz
                        "names": ["state_trajectory"],
                    },
                },
                image_writer_threads=50,
                image_writer_processes=50,
)
```

Please notice that the dataset keys (e.g., `image`, `state`, `actions`, `trajectory`, etc.) should match those expected by the policy training scripts. You can modify the keys based on your dataset and ensure they are consistent throughout the pipeline. The keys mapping can be configured in the `policy/lerobot/common/constants.py`.