#
from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys
import os 
import base64
from load import*
global graph, model

model, graph = init()
app = Flask(__name__)


sys.path.append(os.path.abspath())
from werkzeug.wrappers import response

@app.route('/')
def homepage_view():
    return render_template("index.html")
    
# Predict route for our saved model
@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    imageData = request.get_data()
    convertImage(imageData)
    x = imread('output.png', model='L')
    x = np.invert(x)
    x = imresize(x,(224,224))
    x = x.reshape(1,224,224,1)

    return response

    
if __name__ == '__main__':
    app.run(debug=True)


