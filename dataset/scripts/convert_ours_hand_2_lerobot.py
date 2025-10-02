
import numpy as np
import os
import shutil
from pathlib import Path
from policy.lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import json
import cv2

#################################################################################
RAW_DATA_DIR_PATH = "SET_YOUR_RAW_DATA_DIR_PATH_DOWNLOADED_FROM_HUGGINGFACE_HERE"
OUTPUT_BASE = "SET_YOUR_OUTPUT_BASE_PATH_HERE"
REPO_NAME = "hand_dataset"
#################################################################################

OUTPUT_PATH = Path(OUTPUT_BASE) / REPO_NAME
if OUTPUT_PATH.exists():
    shutil.rmtree(OUTPUT_PATH)

dataset = LeRobotDataset.create(
    repo_id=OUTPUT_PATH,
    robot_type="franka",
    fps=30,
    features={
        "top_image": {
            "dtype": "image",
            "shape": (1080, 1920, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (720, 1280, 3),
            "names": ["height", "width", "channel"],
        },
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
            "shape": (3,), 
            "names": ["state"],
        },
        "actions": {
            "dtype": "float64",
            "shape": (3,),
            "names": ["actions"],
        },
        "trajectory": {
            "dtype": "float64",
            "shape": (3,),  
            "names": ["trajectory"],
        },
        "state_trajectory": {
            "dtype": "float64",
            "shape": (3,), 
            "names": ["state_trajectory"],
        },
    },
    image_writer_threads=50,
    image_writer_processes=50,
)

dirs = [d for d in os.listdir(RAW_DATA_DIR_PATH) if os.path.isdir(os.path.join(RAW_DATA_DIR_PATH, d))]
import random
random.seed(0)
sample_len = len(dirs)
dirs = random.sample(dirs, sample_len)
for subdir in dirs:
    subdir_path = os.path.join(RAW_DATA_DIR_PATH, subdir)
    pose_json_path = os.path.join(subdir_path, "hand_mano_pose_estimation_results.json")
    if not os.path.exists(pose_json_path):
        print(f"JSON file {pose_json_path} does not exist, skipping {subdir_path}.")
        continue
    
    with open(pose_json_path, 'r') as f:
        pose_data = json.load(f)
    pose_data = pose_data['joints_3d_batch']

    task_info_path = os.path.join(subdir_path, "task_info.json")
    with open(task_info_path, 'r') as f:
        task_data = json.load(f)
    if 'task_name' in task_data:
        task_name = task_data['task_name']
    elif 'name' in task_data:
        task_name = task_data['name']
    else:
        raise ValueError(f"Cannot find task name in {task_info_path}")
    top_video_path = os.path.join(subdir_path, "top_image.mp4")
    wrist_video_path = os.path.join(subdir_path, "wrist_image.mp4")
    main_video_path = os.path.join(subdir_path, "main_image.mp4")
    human_video_path = os.path.join(subdir_path, "human_image.mp4")

    top_cap = cv2.VideoCapture(top_video_path)
    wrist_cap = cv2.VideoCapture(wrist_video_path)
    main_cap = cv2.VideoCapture(main_video_path)
    human_cap = cv2.VideoCapture(human_video_path)

    if not (top_cap.isOpened() and wrist_cap.isOpened() and main_cap.isOpened() and human_cap.isOpened()):
        print(f"One or more video files in {subdir_path} could not be opened, skipping.")
        continue
    frame_count = 0
    last_target_point = None
    while True:
        frame = {}
        ret_top, top_frame = top_cap.read()
        ret_wrist, wrist_frame = wrist_cap.read()
        ret_main, main_frame = main_cap.read()
        ret_human, human_frame = human_cap.read()
        if not (ret_top and ret_wrist and ret_main and ret_human):
            break
        top_frame = cv2.cvtColor(top_frame, cv2.COLOR_BGR2RGB)
        wrist_frame = cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2RGB)
        main_frame = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
        human_frame = cv2.cvtColor(human_frame, cv2.COLOR_BGR2RGB)
        if frame_count < len(pose_data):
            pose = pose_data[frame_count]
            if pose is None or len(pose) != 21:
                frame_count += 1
                continue
            target_point = ((np.array(pose[16]) + np.array(pose[17]))/2)
            if last_target_point is None:
                last_target_point = target_point
            else:
                actions = np.array(target_point) - np.array(last_target_point)
                frame['actions'] = actions
                frame['state'] = last_target_point
                frame['top_image'] = top_frame
                frame['wrist_image'] = wrist_frame
                frame['main_image'] = main_frame
                frame['human_image'] = human_frame
                frame['trajectory'] = last_target_point
                frame['state_trajectory'] = last_target_point 
                dataset.add_frame(
                    task= task_name,
                    frame=frame
                )
                last_target_point = target_point
        else:
            print(f"Frame count {frame_count} exceeds pose data length, skipping.")
            break
        frame_count += 1
    dataset.save_episode()


    print(f"Processing {subdir_path}...")
