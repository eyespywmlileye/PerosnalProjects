from zipfile import ZipFile 
import pandas as pd 
import pickle 
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 

file = "/content/data0.pickle"

#Install kaggle 
! pip install -q kaggle 
from google.colab import files 
files.upload()

#Cretae a kaggle folder
! mkdir ~/.kaggle

#Copy the kaggle json to folder created 
! cp kaggle.json ~/.kaggle/

#Permison for the json to act 
! chmod 600 ~/.kaggle/kaggle.json

#to list all datasets in kaggle 
! kaggle datasets list

#Download dataset 
#! kaggle competitions download -c amex-default-prediction
! kaggle datasets download -d valentynsichkar/traffic-signs-preprocessed

with ZipFile("/content/traffic-signs-preprocessed.zip" , "r") as f: 
  f.extractall(), 
  f.close()

data_0 = pd.read_pickle(file)

y_test = data_0["y_test"].astype(np.float32)
X_test = data_0["x_test"]

X_train = data_0["x_train"].astype(np.float32)
y_train = data_0["y_train"]

X_val= data_0["x_validation"].astype(np.float32)
y_val = data_0["y_validation"]

def load_rgb_data(file):
    # Opening 'pickle' file and getting images
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3
        # At the same time method 'astype()' used for converting ndarray from int to float
        # It is needed to divide float by float when applying Normalization
        x = d['features'].astype(np.float32)   # 4D numpy.ndarray type, for train = (34799, 32, 32, 3)
        y = d['labels']                        # 1D numpy.ndarray type, for train = (34799,)
        s = d['sizes']                         # 2D numpy.ndarray type, for train = (34799, 2)
        c = d['coords']                        # 2D numpy.ndarray type, for train = (34799, 4)
        """
        Data is a dictionary with four keys:
            'features' - is a 4D array with raw pixel data of the traffic sign images,
                         (number of examples, width, height, channels).
            'labels'   - is a 1D array containing the label id of the traffic sign image,
                         file label_names.csv contains id -> name mappings.
            'sizes'    - is a 2D array containing arrays (width, height),
                         representing the original width and height of the image.
            'coords'   - is a 2D array containing arrays (x1, y1, x2, y2),
                         representing coordinates of a bounding frame around the image.
        """

    # Returning ready data
    return x, y, s, c

y_train = tf.keras.utils.to_categorical(y_train , num_classes = 43) 
y_test =  tf.keras.utils.to_categorical(y_test , num_classes = 43) 
y_val =  tf.keras.utils.to_categorical(y_val , num_classes = 43) 

X_train= X_train.transpose (0,2,3,1)
X_test = X_test.transpose(0,2,3,1)
X_val= X_val.transpose( 0,2,3,1)

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, AvgPool2D, BatchNormalization, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler


model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(43, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history= model.fit(X_train , y_train , 
                    
                   epochs = 10 , batch_size = 32)


model.evaluate(X_train , y_train)

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """ 
  loss = history.history['loss']
  # val_loss = history.history['val_loss']


  accuracy = history.history['accuracy']
  # val_accuracy = history.history['val_accuracy']


  epochs = range(len(history.history['loss']))


  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  # plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()


  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  # plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();

plot_loss_curves(history)