import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np 

!wget https://raw.githubusercontent.com/eyespywmlileye/Helper_functions-/main/helper_function.py?token=GHSAT0AAAAAABXK7T4H6NPYLE5TAVBQMJSGYZIN6YA

from helper_function import create_tensorboard_callback, plot_loss_curves , confusion_matrix , load_and_prep_image, compare_historys , walk_through_dir , unzip_data 

#Get tensorflow datasets 

import tensorflow_datasets as tfds 

datasets_list = tfds.list_builders() # get all avaiable datasets in TFDS datasets 

#Load in the data (takes 5-6 min in Google colab)
(train_data,test_data) , ds_info = tfds.load(name = "food101", 
                                             split = ["train", "validation"], 
                                             shuffle_files = True, 
                                             as_supervised = True, #data gets returned in tuple format (data,label)
                                             with_info = True)

ds_info.features 
class_names = ds_info.features["label"].names

#Make a fucntion from preprcoessing images 
def preprocess_img (image, label , img_shape = 224): 
  """ 
  converts image data type from `unit8` -> `float32` and reshapes 
  image ti [img_shape, img_shape,colour_channels]
  """

  image = tf.image.resize (image, [img_shape,img_shape]) # reshape image 
  #image = image/255. #scale image values ( not required with EfficientNetBX models from tf.keras.applications)
  return tf.cast(image, tf.float32), label # return (float32_image , label) tuple

# Map preprocessing function to training data (and paralellize)
train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Shuffle train_data and turn it into batches and prefetch it (load it faster)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Map prepreprocessing function to test data
test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Turn test data into batches (don't need to shuffle)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

#Create tenorboard callback (import from helper_functions.py)

 
from tensorflow.keras.callbacks import ModelCheckpoint 

#Create modelcheckpoint callback to save a models progress during training 

checkpoint_path = "model_checkpoints/cp.ckpt"
model_checkpoint =  ModelCheckpoint(checkpoint_path , 
                                            monitor = "val_acc", 
                                            save_best_only = True, #Save the best model with the highest validation accuacy  
                                            save_weights_only = True, 
                                            verbose = 0 )


#Turn on mixed precsion traning 

# Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=3) # if val loss decreases for 3 epochs in a row, stop training

# Creating learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=2,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)


# Create ModelCheckpoint callback to save best model during fine-tuning
checkpoint_path = "fine_tune_checkpoints/"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_best_only=True,
                                                      monitor="val_loss")


from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16") #Set global data policy to mixed precision  

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing 

#Create base model 

input_shape = ( 224,224,3)
base_model = tf.keras.applications.EfficientNetB4(include_top = False) #Download all pre-trained weights
base_model.trainable = False 

#Create fucntional model 
input= layers.Input(shape = input_shape , name = "input_layer")
#NOTE: EfficientNetBx models have rescaling built in but if your model does not it could be a problem, here is the soloution: 
# x = preprocessing.Rescaling (1/255.)(x)
x = base_model (input, training = False)  #Makes sure layers which should be in inference traning (not training) stay like that 
x = layers.GlobalAveragePooling2D ()(x) # Feature label 
x = layers.Dense (len(class_names)) (x)
output = layers.Activation ("softmax", dtype = tf.float32 , name ="softmax_flaot_32")(x)
model = tf.keras.Model(input,output) 

#Compile model 

model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(), #use sparse cause labels is not one-hot incoded , it is intergers 
             optimizer = tf.keras.optimizers.Adam(), 
             metrics = ["accuracy"])


# fit the feature extraction modl with call backs 

history_101_classes_feature_extract = model.fit (train_data,
                                                        epochs=100, # fine-tune for a maximum of 100 epochs
                                                        steps_per_epoch=len(train_data),
                                                        validation_data=test_data,
                                                        validation_steps=(test_data), # validation during training on 15% of test data
                                                        callbacks=[create_tensorboard_callback("training_logs", "efficientb0_101_classes_all_data_fine_tuning"), # track the model training logs
                                                                   model_checkpoint, # save only the best model during training
                                                                   early_stopping, # stop model after X epochs of no improvements
                                                                   reduce_lr]) # reduce the learning rate after X epochs of no improvements

results_loaded_fine_tuned = model.evaluate(test_data) 
results_loaded_fine_tuned