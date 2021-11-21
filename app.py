
from flask import Flask, render_template, request
from scripy.msc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import sys
import os 
import base64
from load import*

sys.path.append(os.path.abspath())
from werkzeug.wrappers import response
app = Flask(__name__)

@app.route('/')
def homepage_view():
    return render_template("index.html")
    
# Predict route for our saved model
@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    response = "ML Prediction"
    return response
    
if __name__ == '__main__':
    app.run(debug=True)

