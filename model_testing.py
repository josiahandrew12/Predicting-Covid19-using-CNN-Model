
# %%
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
model = load_model('model.h5')


# %%
img = image.load_img('/Users/joe/Desktop/test/predictingcovid/images.png', grayscale=False, target_size=(224, 224))
show_img=image.load_img('/Users/joe/Desktop/test/predictingcovid/images.png', grayscale=False, target_size=(200, 200))
disease_class=['Covid-19','Non Covid-19']
x = image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
x /= 255

custom = model.predict(x)
print(custom[0])

plt.imshow(show_img)
plt.show()

a=custom[0]
ind=np.argmax(a)
        
print('Diagnois:',disease_class[ind])



# %%
