from flask import Flask, render_template, Response
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
from sklearn.model_selection import train_test_split
import numpy as np

#cargar y preprocesar las imagenes
path = 'Dataset/'
categories = os.listdir(path)

# Cargar el modelo entrenado
model = tf.keras.models.load_model('fruit_classifier.h5')

cap = cv2.VideoCapture(0)

redBajo = np.array([0, 0, 0], np.uint8)
redAlto = np.array([179, 255, 116], np.uint8)
orangeBajo = np.array([10, 100, 80], np.uint8)
orangeAlto = np.array([20, 255, 255], np.uint8)
yellowBajo = np.array([25, 100, 100], np.uint8)
yellowAlto = np.array([45, 255, 255], np.uint8)
greenBajo = np.array([35, 100, 0], np.uint8)
greenAlto = np.array([75, 255, 255], np.uint8)


#funcion de generacion de frames
def gen_frame():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
                # Preprocesar imagen
            image = cv2.resize(frame, (50,50))
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

            # Detectar el color de la fruta
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            maskRed = cv2.inRange(hsv, redBajo, redAlto)
            maskOrange = cv2.inRange(hsv, orangeBajo, orangeAlto)
            maskYellow = cv2.inRange(hsv, yellowBajo, yellowAlto)
            maskGreen = cv2.inRange(hsv, greenBajo, greenAlto)

            # Evaluar la madurez de la fruta según el color detectado
            if (np.count_nonzero(maskRed) > 1000): # si hay suficiente color rojo en la imagen
                madurez = "Madura"
            elif(np.count_nonzero(maskOrange) > 1000): # si hay suficiente color naranja en la imagen
                madurez = "Madura"
            elif(np.count_nonzero(maskYellow) > 1000): # si hay suficiente color amarillo en la imagen
                madurez = "Madura"
            elif(np.count_nonzero(maskGreen) > 1000): # si hay suficiente color verde en la imagen
                madurez = "No madura"
            else:
                madurez = ""

            # Hacer prediccion
            pred = model.predict(np.expand_dims(image, axis=0))
            #Clasificar la imagen
            class_idx = np.argmax(pred, axis=1)[0]
            class_name = categories[class_idx]
            #Mostrar el resultado de la clasificacion y la madurez en pantalla
            cv2.putText(frame, "Fruta: " + class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, "Madurez: " + madurez, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            suc, encode = cv2.imencode('.jpg', frame)
            frame = encode.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Index.html')

@app.route('/video')
def video():
    return Response(gen_frame(),  mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)


