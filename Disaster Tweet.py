 
 !wget https://raw.githubusercontent.com/eyespywmlileye/Helper_functions-/main/helper_function.py?token=GHSAT0AAAAAABXK7T4H6NPYLE5TAVBQMJSGYZIN6YA

#Import a series of helper funtions for the notebook 
from helper_functions import unzip_data ,walk_through_dir, create_tensorboard_callback , plot_loss_curves , compare_historys
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random 


#Reading in data using pandas 
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df_shuffled = train_df.sample(frac = 1, random_state = 42)
 

random_index = random.randint(0,len(train_df)-5) #Create random indexes not higher than the total number of samples 
for row in train_df_shuffled [["text", "target"]][random_index:random_index+5].itertuples(): 
  _,text,tagret =row 
  print(f"{tagret}", "(real_disater)" if tagret > 0 else "(not real disaster)")
  print(f"Text:\n{text}\n")
  print("---\n")

# Split data into train and val sets 

from sklearn.model_selection import train_test_split 

#Use train_test_split to spilt training data into trainng and validation sets 
train_sentences , val_sentences , train_labels , val_labels = train_test_split(train_df_shuffled["text"].to_numpy(), # use .numpy() when spilting text data
                                                                               train_df_shuffled["target"].to_numpy(), 
                                                                               test_size = 0.1 , #Use 10% of training data for validation split 
                                                                               random_state=42)
                                                        
from tensorflow.keras.layers import TextVectorization

#Use deafult TextVectorizaiton parameters 
text_vectorizer = TextVectorization (max_tokens = None ,  #How many words in the vocabulary  (automatically add ,00v) which stands for "out of vocabulary"
                                     standardize = "lower_and_strip_punctuation", 
                                     split = "whitespace", 
                                     ngrams = None, #Create groups of n-words  
                                     output_mode = "int", #how to map tokens to numbers  
                                     output_sequence_length = None) # how long do you want your sequences to be
                                     # pad_to_max_tokens=True) # Not valid if using max_tokens=None


#Find the average number of tokens (words) in the training tweets 
round(sum([len(i.split()) for i in train_sentences])/len(train_sentences))

#Setup text vectorization variables 
max_vocab_length = 10000 #max number of words to have in our vocabulary 
max_length = 15 #max lenght our sequences will be ( e.g how many words from a tweet does the model see)

text_vectorizer = TextVectorization(max_tokens = max_vocab_length, 
                                    output_mode = "int", 
                                    output_sequence_length = max_length)

#Fit the text vectorizer to the training set 
text_vectorizer.adapt(train_sentences)

#Get the unique words in the vocabulary 
words_in_vocab= text_vectorizer.get_vocabulary()#Get all of the unique words in our training data
top_5_words = words_in_vocab[:5] # get the most common words 
bottom_5_words = words_in_vocab [-5:] #Get the least common words 

#Creating an embedding layer 

from tensorflow.keras import layers

embedding = layers.Embedding(input_dim = max_vocab_length, # set input shape  
                             output_dim = 128, #output shape
                             input_length = max_length #how long is each input
                             )

embedding

import tensorflow_hub as hub 
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv1D , Dense 
from tensorflow.keras import Sequential

SAVE_DIR = "model_logs"


inputs = layers.Input ( shape = (1,), dtype = tf.string)
x = text_vectorizer(inputs)
x = embedding (x)

x = layers.Bidirectional(layers.LSTM(64, return_sequences = True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
outputs = layers.Dense (1 , activation = "sigmoid")(x)

model_4 =tf.keras.Model(inputs,outputs)

#Compile model 

model_4.compile(loss ="binary_crossentropy", 
                optimizer = tf.keras.optimizers.Adam(), 
                metrics =["accuracy"])

#Fit the model  
model_4_history =model_4.fit(train_sentences, 
                              train_labels ,
                              epochs = 5, 
                              validation_data = (val_sentences , val_labels), 
                              callbacks =[create_tensorboard_callback(SAVE_DIR, 
                                                                      "model_4_bidirectional")]) 

#Predict using model_4 

model_4_pred_probs = model_4.predict(val_sentences)
#Squueze pred_probs 
model_4_preds = tf.squeeze(tf.round(model_4_pred_probs))

#Show results 
model_4_results = metric_function(val_labels, 
                                  model_4_preds)
model_4_results



#Testing model on input data 

def model_predict( tweet): 
  prediction = model_4.predict([tweet])

  if tf.round(prediction) == 0: 
    print("Not a disaster")
  else: 
    print ("Disaster")

  print(prediction)
