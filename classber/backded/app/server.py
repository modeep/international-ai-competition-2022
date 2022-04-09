from tkinter import Y
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import serial
import cv2

from threading import Thread
import time

from camera.oak_d_poe import oak_read


scroll_status = 0

def create_app(app, socket_io:SocketIO, arduino:serial.Serial):
    @app.route('/')
    def index():
        return render_template('index.html')

    def run_model(frame):
        print("running model!")
        for i in range(5000): print(i)
        print("model run end!")
        arduino.write('1'.encode('utf-8'))
        # arduino serial control
        return


    def gen():
        oak_device = oak_read()

        prev_time = time.time()
        with oak_device as device:
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            while True:
                frame = cv2.imencode('.jpg', qRgb.get().getCvFrame())[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.05)
                end_time = time.time()
                if end_time - prev_time > 10:
                    thread = Thread(target=run_model, args=(frame,))
                    thread.daemon = True
                    prev_time = time.time()
                    thread.start()

    @app.route('/video_feed')
    def video_feed():
        return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @socket_io.on("move")
    def request(message):
        message = str(message)
        print(f"message : {message}")
        arduino.write(message.encode('utf-8'))

    @socket_io.on("scroll")
    def scroll(message):
        print(message)
        scroll_status += int(message["scroll"])
        socket_io.emit('scroll', {"scroll_status" : scroll_status})

if __name__ == '__main__':
    app = Flask(__name__)
    app.secret_key = "FlaskSecret"
    socket_io = SocketIO(app, cors_allowed_origins="*")

    arduino = serial.Serial('COM9', 9600)

    create_app(app, socket_io, arduino)
    socket_io.run(app, host='0.0.0.0', port=5000)