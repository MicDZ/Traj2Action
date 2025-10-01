from flask import Flask, render_template, request, jsonify
import threading
import time
import os
import json
import numpy as np
import cv2
import sys
import logging
# Add the parent directory to sys.path to import from systems
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from systems.robot_evaluation_system import RobotEvaluationSystem

#####################################################
DATA_SAVE_PATH = "./data/evaluation/20250912_fps5-10"
ACTION_ONLY_MODE = False  # Whether to use only action space for control
CALIBRATION =True
#####################################################

# Import necessary functions and classes from your original script
TASKS = [
    # "stack the paper cups",
    # "stack the rings on the pillar"
    # "clean up the table",

    # "pick up the water bottle",

    "pick up the tomato and put it in the yellow tray",
    "pick up the tomato and put it in the blue tray",

    # "pick up the tomato and put it in the basket",
    # "pick up the pepper and put it in the yellow tray",
    # "pick up the pepper and put it in the blue tray",
    # "pick up the pepper and put it in the basket",
    # "pick up the broccoli and put it in the yellow tray",
    # "pick up the broccoli and put it in the blue tray",
    # "pick up the broccoli and put it in the basket",
]

app = Flask(__name__)

# Global variables for robot system state
robot_system = None
is_running = False
has_calibration = False
current_task = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def initialize_robot_system():
    """Initialize the robot manipulation system"""
    global robot_system, has_calibration
    try:
        if robot_system is None:
            robot_system = RobotEvaluationSystem(save_dir=DATA_SAVE_PATH,action_only_mode=ACTION_ONLY_MODE, calibration=CALIBRATION)
        else:
            robot_system.reset_for_collection()
        has_calibration = True
        logging.info("Robot manipulation system initialized successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize robot system: {e}")
        has_calibration = False
        return False

@app.route('/')
def index():
    """Render main page"""
    return render_template('index_eval.html', tasks=TASKS)

@app.route('/status', methods=['GET'])
def get_status():
    """Get current status"""
    global is_running, has_calibration
    return jsonify({
        'status': 'success',
        'recording': is_running,
        'calibrating': False,
        'has_calibration': has_calibration
    })

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Start robot task"""
    global robot_system, is_running, current_task
    
    if is_running:
        return jsonify({'status': 'error', 'message': 'Robot is already running'})
    
    if not has_calibration or robot_system is None:
        return jsonify({'status': 'error', 'message': 'Please initialize robot system first'})
    
    data = request.get_json()
    task = data.get('task', '').strip()
    custom_task = data.get('custom_task', '').strip()
    
    # Use custom task if provided, otherwise use selected task
    task_name = custom_task if custom_task else task
    
    if not task_name:
        return jsonify({'status': 'error', 'message': 'Please select or enter a task'})
    
    try:
        current_task = task_name
        robot_system.run(task_name=task_name)
        is_running = True
        logging.info(f"Started robot task: {task_name}")
        return jsonify({
            'status': 'success',
            'message': f'Robot task started: {task_name}',
            'task': task_name
        })
    except Exception as e:
        logging.error(f"Failed to start robot task: {e}")
        return jsonify({'status': 'error', 'message': f'Start failed: {str(e)}'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Stop robot task (successful completion)"""
    global robot_system, is_running, current_task
    
    if not is_running:
        return jsonify({'status': 'error', 'message': 'Robot is not running'})
    
    try:
        robot_system.stop(success=True)
        is_running = False
        task_name = current_task
        current_task = None
        logging.info(f"Stopped robot task successfully: {task_name}")
        return jsonify({
            'status': 'success',
            'message': f'Task completed successfully: {task_name}'
        })
    except Exception as e:
        logging.error(f"Failed to stop robot task: {e}")
        return jsonify({'status': 'error', 'message': f'Stop failed: {str(e)}'})

@app.route('/remove_recording', methods=['POST'])
def remove_recording():
    """Stop robot task (mark as failed)"""
    global robot_system, is_running, current_task
    
    if not is_running:
        return jsonify({'status': 'error', 'message': 'Robot is not running'})
    
    try:
        robot_system.stop(success=False)
        is_running = False
        task_name = current_task
        current_task = None
        logging.info(f"Stopped robot task as failed: {task_name}")
        return jsonify({
            'status': 'success',
            'message': f'Task marked as failed: {task_name}'
        })
    except Exception as e:
        logging.error(f"Failed to remove robot task: {e}")
        return jsonify({'status': 'error', 'message': f'Removal failed: {str(e)}'})

@app.route('/initialize_system', methods=['POST'])
def initialize_system():
    """Reset robot to collection position"""
    global robot_system, is_running, has_calibration

    if is_running:
        return jsonify({'status': 'error', 'message': 'Robot is running, cannot reset'})

    success = initialize_robot_system()
    if not success:
        return jsonify({'status': 'error', 'message': 'Please initialize robot system first'})
    else:
        return jsonify({'status': 'success', 'message': 'Robot reset to collection position'})

@app.route('/start_placer', methods=['POST'])
def start_placer():
    pass

if __name__ == '__main__':
    # Initialize robot system on startup

    app.run(host='0.0.0.0', port=5001, debug=False) # host='0.0.0.0' allows external access