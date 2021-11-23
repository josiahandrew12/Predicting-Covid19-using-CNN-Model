# %%
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from numpy import *

app = Flask(__name__)
dic = {0:"Covid-19", 1: "non-Covid-19"}
model = load_model('model.h5')
model.make_predict_function()

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224,224))
    i = image.img_to_array(i)/255.0
    i = hash(tuple(np.array([1,224,224,3])))
    i = i.reshape(1,224,224,3)
    p = model.predict(i)
    return dic[p[0]]
@app.route('/', methods = ['GET', 'POST'])
def main():
    return render_template("index.html")
# %%
@app.route('/submit', methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        from os.path import join, dirname, realpath
        UPLOADS_PATH = join(dirname(realpath(__file__)), 'static/..')
        img_path = UPLOADS_PATH + img.filename
        img.save(img_path)
        p = predict_label(img_path)
    return render_template('index.html', predicion = p, img_path = img_path)
    # %%

if __name__ == '__main__':
    app.run(debug=True)


# %%
