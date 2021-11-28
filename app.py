# %%
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from os.path import join, dirname, realpath



app = Flask(__name__)

loaded_model = load_model("model.h5")


@app.route('/', methods = ['GET', 'POST'])
def main():
    return render_template("index.html")
# %%
# %%
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from os.path import join, dirname, realpath
def predict_image(img_path): 
        disease_class=['Covid-19','Non Covid-19']
        model = load_model("model.h5")
        x = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(x)
        x = x.reshape(1,224,224,3)
        x = x/255.0         
        disease_class=['Covid-19','Non Covid-19']               
        prediction = model.predict(x)
        a=prediction[0]
        ind=np.argmax(a)     
        result = disease_class[ind] 
        return result

# %%
@app.route('/submit', methods = ['GET', 'POST'])
def get_files():
    if request.method == 'POST':
        img = request.files['image_input']
        path_upload = join(dirname(realpath(__file__)), 'static/..')
        image_path = path_upload + img.filename
        img.save(image_path)
        final_prediction = predict_image(image_path)
        return render_template("index.html", prediction = final_prediction, img_path = image_path)

if __name__ == '__main__':
    app.run(debug=True)
