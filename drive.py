import argparse
import base64
import json

import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)
    # image_array_pro =  cv2.cvtColor(image_array[50:140, :, :], cv2.COLOR_RGB2YUV)

    ##GRAYSCALE, DOWNSAMPLE
    # image_array_pro =  cv2.cvtColor(cv2.resize(image_array[30:150,:,:], (0,0), fx=0.6, fy=0.6), cv2.COLOR_RGB2GRAY)
    # image_array_pro2= image_array_pro[..., np.newaxis]
    #transformed_image_array = image_array_pro2[None, :, :, :]


    ##DOWNSAMPPLE
    image_array_pro = cv2.resize(image_array[ :, :, :],(0,0), fx=0.4, fy=0.4)
    transformed_image_array = image_array_pro[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))

    #Basic P control on throttle
    speed_setpoint = 25
    kP = 1.0/5.0
    throttle = (speed_setpoint-float(speed))*kP
    #should put a np.min here but something funny is going on when i do
    # throttle = 0.15
    print(steering_angle, throttle, speed)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)