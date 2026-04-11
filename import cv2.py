import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from tensorflow.keras.models import load_model
import RPi.GPIO as GPIO

model = load_model("traffiq_model.h5")
camera = cv2.VideoCapture(0)

STEER_PIN = 18
THROTTLE_PIN = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(STEER_PIN, GPIO.OUT)
GPIO.setup(THROTTLE_PIN, GPIO.OUT)

steer_pwm = GPIO.PWM(STEER_PIN, 50)
throttle_pwm = GPIO.PWM(THROTTLE_PIN, 50)

steer_pwm.start(0)
throttle_pwm.start(0)

def preprocess(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

def control_vehicle(speed, direction):
    steer_val = np.interp(direction, [-1, 1], [5, 10])
    throttle_val = np.interp(speed, [-1, 1], [0, 100])
    steer_pwm.ChangeDutyCycle(steer_val)
    throttle_pwm.ChangeDutyCycle(throttle_val)

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            control_vehicle(0, 0)
            break
        input_data = preprocess(frame)
        prediction = model.predict(input_data)[0]
        speed, direction = prediction[0], prediction[1]
        control_vehicle(speed, direction)
except KeyboardInterrupt:
    pass
finally:
    steer_pwm.stop()
    throttle_pwm.stop()
    GPIO.cleanup()
    camera.release()
    cv2.destroyAllWindows()