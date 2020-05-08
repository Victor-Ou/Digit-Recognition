from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageFilter
import os
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import keras
from keras import backend as K
import tensorflow as tf

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

app = Flask(__name__, template_folder='Templates')
model = tf.keras.models.load_model("Digit_Rec.h5")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['Upload_folder'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template('Home.html')


@app.route('/predict', methods=['POST'])
def post():
    # Posting the image
    if request.method == 'POST':
        img = Image.open(request.files['myfile'].stream).convert("L")
        img = img.resize((28, 28))
        img = img_to_array(img)
        img = img.reshape(1, 28, 28, 1)
        img = img.astype('float32')
        prediction = model.predict(img)
        output = prediction.argmax()

        return render_template("Home.html", prediction_text='The number in the picture is a {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
