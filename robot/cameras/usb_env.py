from cameras.camera_env import CameraEnv
import threading
import time
import numpy as np
import pyrealsense2 as rs
import logging
import cv2

class USBEnv(CameraEnv):
    def __init__(self, camera_name: str, serial_number: str, exposure: float = 250, fps: float = 30, width: int = 1280, height: int = 720, camera_param = None):
        super().__init__(camera_name=camera_name, exposure=exposure, fps=fps, camera_type="USB", width=width, height=height, camera_param=camera_param)
        # assert serial_number is a num
        assert serial_number.isdigit(), "Please provide the USB camera index as a string."
        try:  
            self.cap = cv2.VideoCapture(int(serial_number))
        except Exception as e:
            logging.error(f"Failed to open USB camera with index {serial_number}: {e}")
            raise e
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))

    def _capture_image(self):
        """Capture image"""
        ret, frames = self.cap.read()
        timestamp = time.time()
        if not ret:
            logging.error(f"Failed to capture color frame: {self.camera_name}")
            return None
        # Convert image from BGR to RGB format
        return {
            "bgr": frames,
            "timestamp": timestamp
            }
    
if __name__ == "__main__":
    import cv2
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Output to console
        ]
    )
    # Example usage
    env = USBEnv(camera_name="top_image", serial_number="12", width=1920, height=1080, exposure=100)
    env.start_monitoring()
    env.start_saving_frames("./data/videotest")

    while True:
        frame = env.get_latest_frame()
        if frame is not None:
            cv2.imshow("USB Camera", frame['bgr'])
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    env.stop_monitoring()
    env.stop_saving_frames()
    cv2.destroyAllWindows()
