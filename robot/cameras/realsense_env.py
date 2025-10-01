from cameras.camera_env import CameraEnv
import threading
import time
import numpy as np
import pyrealsense2 as rs
import logging
import cv2

class RealSenseEnv(CameraEnv):
    def __init__(self, camera_name: str, serial_number: str, exposure: float = 250, fps: float = 30, width: int = 1280, height: int = 720, camera_param = None):
        super().__init__(camera_name=camera_name, exposure=exposure, fps=fps, camera_type="RealSense", width=width, height=height, camera_param=camera_param)
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial_number is None:
            raise ValueError("Please provide the RealSense camera serial number.")
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.pipeline.start(config)
        sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]  # Color sensor
        sensor.set_option(rs.option.exposure, exposure)  # Set exposure time
    def _capture_image(self):
        """Capture image"""
        frames = self.pipeline.wait_for_frames()
        timestamp = frames.get_timestamp()
        color_frame = frames.get_color_frame()
        if not color_frame:
            logging.error(f"Failed to capture color frame: {self.camera_name}")
            return None
        color_image = np.asanyarray(color_frame.get_data())
        # Convert image from BGR to RGB format
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        return {
            "bgr": color_image,
            "timestamp": time.time()
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
    env = RealSenseEnv(camera_name="main_image", serial_number="339322073638", width=640, height=480)
    env.start_monitoring()
    env.start_saving_frames("./data/videotest")

    while True:
        frame = env.get_latest_frame()
        if frame is not None:
            cv2.imshow("RealSense Camera", frame['bgr'])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    env.stop_monitoring()
    env.stop_saving_frames()
    cv2.destroyAllWindows()
    env.pipeline.stop()