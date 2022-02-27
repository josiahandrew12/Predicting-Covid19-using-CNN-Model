from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from os.path import join, dirname, realpath
h=3

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def main():
    return render_template("index.html")

disease_class=['Covid-19','Non Covid-19']
model = load_model("model.h5")

def data_processing(img_path):
        x = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(x)
        x = x.reshape(1,224,224,3)
        x = x/255.0
        return x

def predict_image(img_path):
        x = data_processing(img_path)
        prediction = model.predict(x)
        a=prediction[0]
        ind=np.argmax(a)
        result = disease_class[ind]
        return result
#Returns accuracy in flask html file
def accuracy_image(img_path):
        x = data_processing(img_path)
        prediction = model.predict(x)
        a=prediction[0]
        ind=np.argmax(a)
        if  (a[1] > a[0]):
             final =  a[1]
        else:
            final = a[0]
        return final

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


if __name__ == '__main__':
    app.run(debug=True)