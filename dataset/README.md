# Traj2Action: Dataset Conversion and Preparation
This directory contains scripts and instructions for converting and preparing datasets in the LeRobot format for training robot policies. It includes tools to convert raw data into the required format, preprocess it, and organize it for efficient loading during training.


## Quick Start

1. Download our collected datasets or prepare your own dataset.
We have uploaded our collected datasets to Hugging Face for easy access:
* [Robot Dataset](https://huggingface.co/datasets/Traj2Action/Traj2Action_Pick_up_Tomato_Robot)
* [Hand Dataset](https://huggingface.co/datasets/Traj2Action/Traj2Action_Pick_up_Tomato_Hand)

These two dataset required a minimum of 60GB free disk space. Considering unzip operation, please ensure you have at least 150GB free disk space.

```sh
mkdir data
huggingface-cli download --resume-download Traj2Action/Traj2Action_Pick_up_Tomato_Robot --local-dir ./data/
huggingface-cli download --resume-download Traj2Action/Traj2Action_Pick_up_Tomato_Hand --local-dir ./data/
unzip ./data/*.zip -d ./data/
```

These datasets we uploaded to Hugging Face are structured in our custom format, which stores synchronized multi-view images, robot states, actions, and trajectories in a highly efficient manner. You can use the provided conversion scripts to adapt your own datasets to this format.

2. Convert the raw data to LeRobot format.

We provide scripts to convert raw data into the LeRobot format. You can find these scripts in the `dataset/scripts` directory. Open the script and set the variables `RAW_DATA_DIR_PATH`, `OUTPUT_BASE`, and `REPO_NAME` to specify the input data directory, output base directory, and repository name respectively.

For example, to convert our robot dataset, you can run:
```sh
python -m dataset.scripts.convert_ours_robot_2_lerobot
```
Similarly, to convert our hand dataset, you can run:
```sh
python -m dataset.scripts.convert_ours_hand_2_lerobot
```




## Dataset Structure

For the dataset we used for training, we follow the LeRobot dataset structure, which can be easily curated using the following LeRobot code:

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

```python
DATA_KEYS_MAPPING_HAND = {
    "left_image":"main_image",
    "ego_image":"human_image",
    "right_image":"wrist_image",
    "top_image":"top_image"
}

DATA_KEYS_MAPPING_ROBOT = {
    "left_image":"image",
    "ego_image":"wrist_image",
}
```

In our training code, we only use `left_image` and `ego_image` for the input images. However, we still save other camera views in the dataset for potential future use.
