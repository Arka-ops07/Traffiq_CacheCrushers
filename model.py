import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from model_base import BaseModel

class Model(BaseModel):
    """
    Refactored TraffIQ model following the organizer's BaseModel.
    """

    def load(self) -> None:
        """
        Initializes the model. 
        Note: We use TFLite instead of Keras (.h5) for real-time performance 
        on the Raspberry Pi 4B hardware.
        """
        # Place your .tflite file in the 'participant/' folder
        model_path = "participant/traffiq_model.tflite"
        
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            # Pre-fetch input/output details to maintain the <100ms requirement
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
        except Exception as e:
            # Raise exception to signal initialization failure as per template
            raise RuntimeError(f"Could not load TFLite model: {e}")

    def _red_signal_detected(self, frame: np.ndarray) -> bool:
        """
        Helper to detect the Traffic Light (Obstacle 1) or Stop Sign (Completion).
        """
        # Focus on the upper 60% of the image where signals are likely to be
        height, width, _ = frame.shape
        roi = frame[0:int(height * 0.6), :]

        # Convert RGB to HSV (Brochure states input frame is RGB)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

        # Mask for red (accounting for the wrap-around in HSV space)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # If a significant cluster of red is detected, we stop
        return cv2.countNonZero(red_mask) > 1500

    def predict(self, frame: np.ndarray) -> tuple:
        """
        Called every frame. Returns (speed, direction) within 100ms.
        """
        # 1. Safety Check: Stop for Red Light or Stop Sign
        # This handles Obstacle 1 and the Completion Sign in the schematic
        if self._red_signal_detected(frame):
            return 0.0, 0.0

        # 2. Navigation Pre-processing
        # To avoid the white wall and ceiling glare noted in the track map,
        # we only feed the bottom 60% of the frame (the floor) to the AI.
        h, w, _ = frame.shape
        floor_roi = frame[int(h * 0.4):h, :]

        # Resize for your model (e.g., 224x224 as in your original snippet)
        input_shape = self.input_details[0]['shape']  # [1, 224, 224, 3]
        target_size = (input_shape[2], input_shape[1])
        resized = cv2.resize(floor_roi, target_size)

        # Normalize 
        input_data = np.expand_dims(resized, axis=0).astype(np.float32)
        input_data /= 255.0

        # 3. TFLite Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Extract speed and direction
        speed = float(output_data[0])
        direction = float(output_data[1])

        # Clamp values to [-1.0, 1.0] as required by the organizer
        return np.clip(speed, -1.0, 1.0), np.clip(direction, -1.0, 1.0)