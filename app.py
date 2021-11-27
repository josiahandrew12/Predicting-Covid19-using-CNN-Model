# %%
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from numpy import *

app = Flask(__name__)
dic = {0:"Covid-19", 1: "non-Covid-19"}
model = load_model('model.h5')
model.make_predict_function()

@app.route('/', methods = ['GET', 'POST'])
def main():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)



