# flask for REST backend
from flask import Flask, request, url_for, redirect, render_template

# keras and tensorflow for loading pretrained model
from keras.models import *
from keras.preprocessing.image import img_to_array, load_img
from keras.applications import imagenet_utils

# resize will resize the image
import matplotlib.pyplot as plt 
import tensorflow as tf 

# saving file
from werkzeug.utils import secure_filename

# numpy for converting images to array
import numpy as np

import os
import io
from PIL import Image

# load json and create model
json_file = open('../Model/model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# load weights into new model
model.load_weights('../Model/model.h5')
print("Loaded model from disk")

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path ,'static','img')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def prepare_image(image, target):

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)

	# return the processed image
	return image


@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    if request.method == 'POST':

        Image_file = request.files["file"]

        # save the image to the upload folder, for display on the webpage.
        save_image = Image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], Image_file.filename))

        with open(os.path.join(app.config['UPLOAD_FOLDER'], Image_file.filename), 'rb') as f:
            read_image = Image.open(io.BytesIO(f.read()))

        processed_image = prepare_image(read_image, target=(224, 224))

        # Image_file = img_to_array(save_image)
        # Image_file = np.expand_dims(Image_file,axis=0)

        prediction = model.predict_classes(processed_image)
            
    return render_template('index.html', pred = 'SARS-COV-19: POSITIVE\nProbability of SARS-COV-19 is {}:'.format(prediction))

    # if results > str(0.5):
    #     return render_template('index.html', pred = 'SARS-COV-19: POSITIVE\nProbability of SARS-COV-19 is {}:'.format(results))
    # else:
    #     return render_template('index.html', pred = 'SARS-COV-19: NEGATIVE\nProbability of SARS-COV-19 is {}:'.format(results))



if __name__ == "__main__":
    app.run(debug = True)