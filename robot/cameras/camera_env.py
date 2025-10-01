import threading
import time
import numpy as np
from abc import abstractmethod
import logging
import cv2 
from cameras.camera_param import CameraParam

class CameraEnv:
    def __init__(self, camera_name: str, exposure: float = 250, fps: float = 30, camera_type: str = "RealSense", width: int = 1280, height: int = 720, camera_param: CameraParam = None):
        self.fps = fps
        self.camera_name = camera_name
        self.exposure = exposure
        self.width = width
        self.height = height
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._frame = None
        self._frame_lock = threading.RLock()  # Use reentrant lock
        self.camera_type = camera_type
        self._saving_thread = None
        self._stop_saving = threading.Event()
        self.camera_param = camera_param

    def start_monitoring(self):
        """Start monitoring thread"""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(
                target=self._update_camera_frame, 
                daemon=True
            )
            self._monitoring_thread.start()

    def stop_monitoring(self):
        """Stop monitoring thread"""
        self._stop_monitoring.set()
        time.sleep(0.11)
        self.stop_saving_frames()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=1.0)
            self._monitoring_thread = None

    def start_saving_frames(self, file_path: str):
        """Start a thread to save camera frames periodically."""
        if self._saving_thread is None or not self._saving_thread.is_alive():
            self._stop_saving.clear()
            self._saving_thread = threading.Thread(
                target=self._save_frames_periodically, 
                args=(file_path,),
                daemon=True
            )
            self._saving_thread.start()

    def stop_saving_frames(self):
        """Stop the frame saving thread."""
        self._stop_saving.set()
        time.sleep(0.11)
        if self._saving_thread and self._saving_thread.is_alive():
            self._saving_thread.join(timeout=1.0)
            self._saving_thread = None

    def _save_frames_periodically(self, file_path: str):
        """Save camera frames to file using OpenCV"""
        video_writer = None
        import os
        from pathlib import Path
        if not os.path.exists(file_path):
            os.makedirs(file_path, exist_ok=True)
        
        last_timestamp = None

        try:
            # Use OpenCV's VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = Path(file_path) / f"{self.camera_name}.mp4"
            timestamps_path = Path(file_path) / f"{self.camera_name}_timestamps.json"
            logging.info(f"Saving camera frames to {video_path}")
            video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (self.width, self.height))

            if not video_writer.isOpened():
                logging.error(f"Failed to open video writer for {video_path}")
                return

            logging.info(f"Starting to save camera data to {file_path} at {self.fps} FPS.")
            timestamps = []
            while not self._stop_monitoring.is_set() and not self._stop_saving.is_set():
                frame = self.get_latest_frame()

                if frame is None:
                    logging.warning("No frame captured, skipping...")
                    time.sleep(1 / self.fps)
                    continue
                if frame.get("bgr") is None or frame.get("timestamp") is None:
                    logging.error("Captured frame is missing 'bgr' or 'timestamp' keys.")
                    time.sleep(1 / self.fps)
                    continue
                # Check frame shape
                frame_shape = frame["bgr"].shape
                expected_shape = (self.height, self.width, 3)

                if frame_shape != expected_shape:
                    logging.error(f"Frame shape {frame_shape} does not match expected {expected_shape}")
                    time.sleep(1 / self.fps)
                    continue
                if last_timestamp is not None and frame["timestamp"] <= last_timestamp:
                    time.sleep(0.001)
                    continue
                last_timestamp = frame["timestamp"]
                # OpenCV uses BGR format, convert if input is RGB
                frame_bgr = frame["bgr"]
                timestamps.append(frame["timestamp"])
                video_writer.write(frame_bgr)

        except Exception as e:
            logging.error(f"Error in _save_frames_periodically: {e}")
        finally:
            if video_writer is not None:
                video_writer.release()
            import json
            with open(timestamps_path, 'w') as f:
                json.dump(timestamps, f, separators=(',', ':'))
            logging.info(f"Timestamps saved to {timestamps_path}")
            logging.info(f"Camera frames saved to {video_path}")

    def _update_camera_frame(self):
        """Update camera status"""
        while not self._stop_monitoring.is_set():
            try:
                frame = self._capture_image()
                with self._frame_lock:
                    if frame is not None:
                        self._frame = frame
            except Exception as e:
                logging.error(f"Camera state update error: {e}")
                time.sleep(0.1)  # Brief wait on error
                continue

    def get_latest_frame(self) -> np.ndarray:
        """Get the latest camera frame"""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None
        
    @abstractmethod
    def _capture_image(self):
        """Capture image blocking"""
        pass

    def __del__(self):
        """Ensure thread closes properly"""
        self.stop_monitoring()
        self.stop_saving_frames()

    def __str__(self):
        return f"CameraEnv(camera_name={self.camera_name}, width={self.width}, height={self.height}, camera_type={self.camera_type}, exposure={self.exposure}, fps={self.fps})"
    
    def calib_camera(self):
        frame = self._capture_image()
        if frame is None or frame.get("bgr") is None:
            raise ValueError("No valid frame captured for calibration.")
            return
        rvec, tvec = self.camera_param.estimate_camera_poses(frame["bgr"])
        self.camera_param.extrinsic_rvec = rvec
        self.camera_param.extrinsic_tvec = tvec
        self.camera_param.update_extrinsic(rvec, tvec)
        
