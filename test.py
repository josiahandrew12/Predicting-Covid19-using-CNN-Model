# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Model,Sequential, Input, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications.densenet import DenseNet121
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import cv2
import os
from tqdm import tqdm

scan_types = ['COVID','non-COVID']
covid_scans = 'data_set'
train_set = os.path.join(covid_scans)

def data_cleaning(scan_types):
    training_scans = []
    for defects_id, sp in enumerate(scan_types):
        for file in os.listdir(os.path.join(train_set, sp)):
            training_scans.append(['{}/{}'.format(sp, file), defects_id, sp])
    train_var = pd.DataFrame(training_scans, columns=['file', 'scanid','scan_types'])
    return train_var
    
data_cleaned = data_cleaning(scan_types)

def data_randomization(data_cleaned):
    data_cleaned = data_cleaned.sample(frac=1, random_state=42) 
    data_cleaned.index = np.arange(len(data_cleaned)) 
    return data_cleaned

data_random = data_randomization(data_cleaned)

# %%
IMAGE_SIZE = 224
def input_image(filepath):
    return cv2.imread(os.path.join(train_set, filepath)) 
    
def resize_scan_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)
#Training data
X_train_data = np.zeros((data_random.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(data_random['file'].values)):
    image = input_image(file)
    if image is not None:
        X_train_data[i] = resize_scan_image(image, (IMAGE_SIZE, IMAGE_SIZE))
X_Train_data = X_train_data / 255.
Y_train = data_random['scanid'].values
Y_train = to_categorical(Y_train, num_classes=2)
BATCH_SIZE = 32
X_train_data, X_val, Y_train, Y_val = train_test_split(X_Train_data, Y_train, test_size=0.2, random_state=42)




# %%
from keras.applications.densenet import DenseNet201
EPOCHS = 7
SIZE=224
N_ch=3
def build_resnet50():
    densetnet201 = DenseNet201(weights='imagenet', include_top=False)

    input = Input(shape=(224, 224, 3))
    x = Conv2D(3, (3, 3), padding='same')(input)
    
    x = densetnet201(x)
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)

    # multi output
    output = Dense(2,activation = 'softmax', name='root')(x)
 

    model = Model(input,output)
    
    optimizer = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    
    return model


# %%
model = build_resnet50()
annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.70, patience=10, verbose=1, min_lr=1e-6)
checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)
datagen = ImageDataGenerator(rotation_range=20, 
                        width_shift_range=0.2, 
                        height_shift_range=0.2, 
                        horizontal_flip=True) 
datagen.fit(X_train_data)
hist = model.fit_generator(datagen.flow(X_train_data, Y_train, batch_size=32),
               steps_per_epoch=X_train_data.shape[0] // BATCH_SIZE,
               epochs=EPOCHS,
               verbose=1,
               callbacks=[annealer, checkpoint],
               validation_data=(X_val, Y_val))


# %%
model = load_model('model.h5')
final_loss, final_accuracy = model.evaluate(X_val, Y_val)
print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))


# %%
from skimage import io
from keras.preprocessing import image
#path='imbalanced/Scratch/Scratch_400.jpg'
img = image.load_img('/Users/josiahcornelius/Desktop/CT_COVID_SCANS/chest-ct-lungs.png', grayscale=False, target_size=(224, 224))
show_img=image.load_img('/Users/josiahcornelius/Desktop/CT_COVID_SCANS/chest-ct-lungs.png', grayscale=False, target_size=(200, 200))
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


