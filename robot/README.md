# Traj2Action: Robot Control Library

This directory contains a robot control library for the Franka Emika Panda robot, including components for robot control, camera management, robot/hand data collection, and task execution.

## System Architecture

The project follows a modular design with several key components:

1. **Robot Environment** - Interfaces with the physical robot
2. **Controller Environment** - Handles user input devices for robot control
3. **Camera Environment** - Manages camera feeds for monitoring and recording
4. **Robot Manipulation System** - Orchestrates all components during task execution
5. **Robot Evaluation System** - Evaluates robot performance using trained policies

## Core Components

### Robots

- `FrankyEnv`: Implementation for Franka Emika robot control
- `robot_env.py`: Base class defining common robot interface
- `robot_param.py`: Robot-specific parameters and configurations

### Controllers

- `SpaceMouseEnv`: Implementation for 3D SpaceMouse controller
- `GelloEnv`: Implementation for Gello controller
- `controller_env.py`: Base class for all controllers

### Cameras

- `RealSenseEnv`: Implementation for Intel RealSense cameras
- `camera_env.py`: Base class for camera interfaces
- `camera_param.py`: Camera parameters and configurations

### Common

- `constants.py`: Shared constants including action space definitions

### Systems

- `manipulation_system.py`: Main system coordinating robot, controllers, and cameras
- `hand_collection_system.py`: System for collecting hand data
- `robot_policy_system`: System for robot policy execution

## Key Features

- Multiple control methods (SpaceMouse, Gello)
- Multi-camera support with synchronized recording
- State recording for robot and controllers
- Timestamped data collection
- Task success/failure tracking
- Organized data storage structure

## Usage


### Installation
Since the hardware setup can vary, you can choose the components that fit your setup. 

Our hardware setup includes:
- Franka Emika Panda robot
- Gello Controller
- Spacemouse Controller
- USB Cameras
- RealSense Cameras
  
You can easily adapt this codebase to your own setup by modifying the drive code.

#### Franky Setup
This codebase uses the [Franky Control](https://github.com/Franky-Emika/Franky) library for high-level robot control. Please follow the installation instructions in the Franky repository to set it up.

#### Camera Calibration

Use Matlab or OpenCV to calibrate your cameras and obtain intrinsic parameters. Save these parameters in a YAML file and provide the path when initializing the camera environment.

Set up the extrinsic calibration between the camera and robot coordinate systems. This is crucial for accurate manipulation tasks.
```python
main_camera_matrix = np.array([[908.1308, 0, 655.7268], [0, 910.0818, 395.8856], [0, 0, 1]], dtype=np.float32)
main_camera_dist_coeffs = np.array([0.1068, -0.2123, -0.0092, 0.0000, 0.0000], dtype=np.float32)
camera = RealSenseEnv(camera_name="main_image", serial_number="339322073638", camera_param=CameraParam(main_camera_matrix, main_camera_dist_coeffs))
```
#### Robot Calibration
Calibrate the transformation between the robot base and world coordinate systems. 


Run the following script to visualize and adjust the calibration:
```sh
python -m robots.robot_param
```
the calibration procedure includes two steps:

1. Calib on xy plane, project the end-effector position to the xy plane
```python
point = robot.robot_param.transform_to_world(ee_xyz)
point[2] = 0.0
```
observe the point in camera frame, making sure the gripper eef projection on the xy plane is at the target point

2. Calib on z axis
```python
point = robot.robot_param.transform_to_world(ee_xyz)
```
move the gripper up and down, making sure the z value is correct



### Basic Usage

#### Robot Data Collection

Run the user interface script to start controlling the robot:
```sh
python -m user_server.manipulation_interface
```

Than open the address in your browser:
```
http://127.0.0.1:5001
```

You can also open it in another computer in the same network.

#### Hand Data Collection

Run the user interface script with hand data collection enabled:
```sh
python -m user_server.hand_interface
```

Than open the address in your browser:
```
http://127.0.0.1:5001
```

You can also open it in another computer in the same network.

#### Robot Policy Execution

Run the robot policy execution script:
```sh
python -m user_server.evaluation_interface
```

Before running, make sure the inference server is up and running, and the port and host are correctly set in `robot/systems/robot_policy_system.py`. You can check `policy/README.md` for instructions on how to start the inference server.

Then open the address in your browser:
```
http://127.0.0.1:5001