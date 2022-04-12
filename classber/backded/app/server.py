from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import serial
import cv2

from threading import Thread
import time

from classber.detect_ai.classber_bottle import ClassBar

scale_status = 1
scale_flag = 0


def create_app(app, socket_io:SocketIO, arduino:serial.Serial, classber):
    @app.route('/')
    def index():
        return render_template('index.html', scale=scale_status**2)

    def run_model(frame):
        print("running model!")
        _, result_location = classber.run_model(frame)
        print("model run end!")

        if result_location is None:
            return

        print("Detect!@!!!!!")

        x1, _, x2, _ = result_location

        object_mid = int(((x2 - x1) / 2) + x1)
        frame_width = int(frame.shape[0] / 2)
        motor_step_value = object_mid - frame_width

        if motor_step_value > 5:
            arduino.write(f'-1'.encode('utf-8'))
        elif motor_step_value < -5:
            arduino.write(f'1'.encode('utf-8'))
        time.sleep(0.05)
        arduino.write(f'0'.encode('utf-8'))
        return

    def gen():
        cap = cv2.VideoCapture(1)
        print("Hello")
        prev_time = time.time()
        while True:
            suc, image = cap.read()
            if suc:
                frame = cv2.imencode('.jpg', image)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                end_time = time.time()
                if end_time - prev_time > 0.05:
                    run_model(image)
            else:
                break

    @app.route('/video_feed')
    def video_feed():
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @socket_io.on("startmove")
    def start_move(message):
        message = str(message)
        arduino.write(message.encode('utf-8'))

    @socket_io.on("endmove")
    def end_move():
        arduino.write('0'.encode('utf-8'))

    @socket_io.on("startcloseup")
    def start_closeup(message):
        global scale_status, scale_flag

        scale_flag = 1

        print("Start CloseUP")

        while scale_flag == 1:
            scale_status += 0.01 * message

            if scale_status > 1: scale_status = 1
            elif scale_status < 0: scale_status = 0

            socket_io.emit('scroll', {"scale_status" : scale_status})
            time.sleep(0.1)

    @socket_io.on("endcloseup")
    def end_closeup():
        global scale_flag
        scale_flag = 0
        print("End CloseUP")


if __name__ == '__main__':
    app = Flask(__name__)
    app.secret_key = "FlaskSecret"
    socket_io = SocketIO(app, cors_allowed_origins="*")

    classber = ClassBar()

    arduino = serial.Serial('/dev/ttyACM1', 9600)

    create_app(app, socket_io, arduino, classber)
    socket_io.run(app, host='0.0.0.0', port=5000, debug=False)
