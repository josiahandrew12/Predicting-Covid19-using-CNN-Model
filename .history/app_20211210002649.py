# %%
from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from os.path import join, dirname, realpath
import numpy 

app = Flask(__name__)



# %%
@app.route('/submit', methods = ['GET', 'POST'])
def get_files():
    if request.method == 'POST':
        img = request.files['image_input']
        path_upload = join(dirname(realpath(__file__)), 'static/..')
        image_path = path_upload + img.filename
        img.save(image_path)
        final_prediction = predict_image(image_path)
        final_accuracy = accuracy_image(image_path)

        return render_template("index.html", prediction = final_prediction,  accuracy = final_accuracy,  image = img.filename)



# %%

if __name__ == '__main__':
    app.run(debug=True)