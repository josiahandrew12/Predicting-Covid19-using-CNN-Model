# %%
from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from os.path import join, dirname, realpath
import numpy 
from connection import s3_connection
from config import AWS_ACCESS_KEY, BUCKET_NAME, AWS_SECRET_KEY
import boto
from boto3.s3.key import Key
app = Flask(__name__)

s3 = s3_connection()

# s3.put_object(
#     Bucket = BUCKET_NAME,
#     body = profile_image,
#     key = s3_path,
#     contentType = profile_image.content_type
# )
# client_s3 = boto3.client("s3")
# result = client_s3.download_file("h5data", "model.h5", "/tmp/model.h5")
# model = load_model("/tmp/model.h5")

# location = s3.get_bucket_location(Bucket=BUKET_NAME)['LocationConstraint']
# image_url = f'https://{BUCKET_NAME}.s3.{location}.amazonaws.com/{s3_path}'
srcFileName = "model.h5"
destFileName = "model1.h5"

bucketName=BUCKET_NAME
bucket = s3.get_bucket(bucketName)

k = Key(bucket, srcFileName)
k.get_contents_to_filename(destFileName)

@app.route('/', methods = ['GET', 'POST'])
def main():
    return render_template("index.html")

disease_class=['Covid-19','Non Covid-19']
model = load_model("model")    

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