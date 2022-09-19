import pandas as pd 
import tensorflow as tf 
import numpy as np 
from keras import layers , Model 
from keras.layers import LSTM ,Bidirectional, GlobalAveragePooling1D , Dense
from keras.optimizers import Adam , SGD  
from keras.losses import CategoricalCrossentropy

from keras.layers import TextVectorization 
from keras.layers import Embedding

# New libraries 
import nltk 
from nltk.corpus import stopwords
from textblob import Word 
from textblob import TextBlob

# Visualize 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots




#Import data 

spotify_data = pd.read_csv("C:\\Users\\thibe\\OneDrive\\Documents\\Spotify review\\archive\\reviews.csv" , usecols=  ["Review" , "Rating"])
 

#categorise labels into positive , neutral and negative 
spotify_data["Rating"].replace(1 , value ="negative" , inplace = True)
spotify_data["Rating"].replace(2, value = "negative" , inplace = True)
spotify_data["Rating"].replace(3, value = "average" , inplace = True)
spotify_data["Rating"].replace (4 , value = "postive" , inplace = True)
spotify_data["Rating"].replace(5, value ="postive" , inplace = True)
 

# Remove punctuation 

import re 
spotify_data["Review"] = spotify_data["Review"].apply(lambda x: re.sub('[^"" a-z A-Z 0-9-]+', '',x))
 

# Remove URL and Tags

spotify_data["Review"] = spotify_data["Review"].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , str(x)))

#make text lower case ( can also do this in tokenization layer )
spotify_data["Review"] = spotify_data["Review"].apply(lambda x: " ".join(x.lower() for x in x.split()))

#Remove numbers 
spotify_data["Review"] = spotify_data["Review"].str.replace("\d", "")

#Removing extra space 
spotify_data["Review"] = spotify_data["Review"].apply(lambda x: " ".join(x.split()))

#Stop words 

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

sw = stopwords.words("english")
spotify_data["Review"] =spotify_data["Review"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))

#Lemmatization 
nltk.download("wordnet")
nltk.download('omw-1.4')
spotify_data["Review"]=spotify_data["Review"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
 

#Split data 


from sklearn.model_selection import train_test_split 

X_train, X_test , y_train , y_test = train_test_split(spotify_data["Review"].tolist(), spotify_data["Rating"], random_state= 42 , test_size= 0.15)
# X_train, X_val , y_train , y_val = train_test_split(X_train, y_train, random_state= 42 , test_size= 0.10)


#One hot encode target labels


from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)

train_labels_one_hot = one_hot_encoder.fit_transform(y_train.to_numpy().reshape(-1, 1))
# val_labels_one_hot = one_hot_encoder.transform(y_val.to_numpy().reshape(-1, 1))
test_labels_one_hot = one_hot_encoder.transform(y_test.to_numpy().reshape(-1, 1))

 

#Label encode labels 

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(y_train.to_numpy())
# val_labels_encoded = label_encoder.transform(y_val.to_numpy())
test_labels_encoded = label_encoder.transform(y_test.to_numpy())

#get class names and number from label encoder instance 
num_classes = len(label_encoder.classes_)
class_names = label_encoder.classes_


#Create text vectorizer 

max_token = 60760
output_seq_len = 41

max_vocab_length = max_token

text_vectorizer = TextVectorization(max_tokens=max_token,
standardize= "lower_and_strip_punctuation", 
split = "whitespace", 
ngrams = None , 
output_mode= "int", 
output_sequence_length= output_seq_len, 
pad_to_max_tokens= True )

text_vectorizer.adapt(X_train)

#Token embedding 

 

embedding = Embedding (input_dim = max_token , 
output_dim= 128 ,  
mask_zero= True, 
name = "token_embedding")

#Create fast dataset 

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, train_labels_one_hot))
# valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, val_labels_one_hot))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, test_labels_one_hot))

# Take the TensorSliceDataset's and turn them into prefetched batches
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
# valid_dataset = valid_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

#Create character -level embedding 

def split_chars(text): 
    """
    splits text into characters 
    """
    return " ".join(list(text))

# Split sequence-level data splits into character-level data splits
train_chars = [split_chars(sentence) for sentence in X_train]
# val_chars = [split_chars(sentence) for sentence in X_val]
test_chars = [split_chars(sentence) for sentence in X_test]
 


# Find what character length covers 95% of sequences
char_lens = [len(sentence) for sentence in X_train]
output_seq_char_len = int(np.percentile(char_lens, 95))
 

# Create char-level token vectorizer instance
# Get all keyboard characters for char-level embedding
import string
alphabet = string.ascii_lowercase + string.digits + string.punctuation


NUM_CHAR_TOKENS = len(alphabet) + 2 # num characters in alphabet + space + OOV token
char_vectorizer = TextVectorization(max_tokens=NUM_CHAR_TOKENS,  
                                    output_sequence_length=output_seq_char_len,
                                    standardize="lower_and_strip_punctuation",
                                    name="char_vectorizer")

# Adapt character vectorizer to training characters
char_vectorizer.adapt(train_chars)
 

#Create char embedding layer
char_embed = layers.Embedding(input_dim=NUM_CHAR_TOKENS, # number of different characters
                              output_dim=25, #
                              mask_zero=False, # don't use masks (this messes up model_5 if set to True)
                              name="char_embed")

# Combine chars and tokens into a dataset
train_char_token_data = tf.data.Dataset.from_tensor_slices((X_train, train_chars)) # make data
train_char_token_labels = tf.data.Dataset.from_tensor_slices(train_labels_one_hot) # make labels
train_char_token_dataset = tf.data.Dataset.zip((train_char_token_data, train_char_token_labels)) # combine data and labels

# Prefetch and batch train data
train_char_token_dataset = train_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE) 

# Repeat same steps validation data
# val_char_token_data = tf.data.Dataset.from_tensor_slices((X_val, val_chars))
# val_char_token_labels = tf.data.Dataset.from_tensor_slices(val_labels_one_hot)
# val_char_token_dataset = tf.data.Dataset.zip((val_char_token_data, val_char_token_labels))
# val_char_token_dataset = val_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Repeat same steps validation data
test_char_token_data = tf.data.Dataset.from_tensor_slices((X_test, test_chars))
test_char_token_labels = tf.data.Dataset.from_tensor_slices(test_labels_one_hot)
test_char_token_dataset = tf.data.Dataset.zip((test_char_token_data, test_char_token_labels))
test_char_token_dataset = test_char_token_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
 
 


# Download pretrained TensorFlow Hub USE
import tensorflow_hub as hub
tf_hub_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                        trainable=False,
                                        name="universal_sentence_encoder")


## Set up training model 

#Create early stopping callback 
early_stopping = tf.keras.callbacks.EarlyStopping(patience= 4)
# Creating learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.2, # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=2,
                                                 verbose=1, # print out when learning rate goes down 
                                                 min_lr=1e-7)

# 1. Setup token inputs/model
token_inputs = layers.Input(shape=[], dtype=tf.string, name="token_input")
token_embeddings = tf_hub_embedding_layer(token_inputs)
token_output = layers.Dense(512, activation="relu")(token_embeddings)
token_model = tf.keras.Model(inputs=token_inputs,
                             outputs=token_output)

# 2. Setup char inputs/model
char_inputs = layers.Input(shape=(1,), dtype=tf.string, name="char_input")
char_vectors = char_vectorizer(char_inputs)
char_embeddings = char_embed(char_vectors)
char_lstm = layers.LSTM(64 , return_sequences= True)(char_embeddings)
char_bi_lstm = layers.Bidirectional(layers.LSTM(32))(char_lstm)  
char_dense = layers.Dense(32 , activation= "relu")(char_bi_lstm)
char_model = tf.keras.Model(inputs=char_inputs,
                            outputs=char_dense)

# 3. Concatenate token and char inputs (create hybrid token embedding)
token_char_concat = layers.Concatenate(name="token_char_hybrid")([token_model.output, 
                                                                  char_model.output])

# 4. Create output layers - 
combined_dropout = layers.Dropout(0.5)(token_char_concat)
combined_dense = layers.Dense(264, activation="relu")(combined_dropout) 
x =  layers.Dense(128, activation="relu")(combined_dense)
final_dropout = layers.Dropout(0.5)(x)
output_layer = layers.Dense(num_classes, activation="softmax")(final_dropout)

# 5. Construct model with char and token inputs
model_5 = tf.keras.Model(inputs=[token_model.input, char_model.input],
                         outputs=output_layer,
                         name="model_4_token_and_char_embeddings")

# Compile the model
model_5.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# Fit the model on tokens and chars
model_5_history = model_5.fit(train_char_token_dataset, # train on dataset of token and characters
                              steps_per_epoch=len(train_char_token_dataset),
                              epochs=10)
                              # validation_data=val_char_token_dataset,
                              # validation_steps= int(0.1* len(val_char_token_dataset)), 
                              # callbacks = [early_stopping , 
                              #reduce_lr])


#Evaluate model
# model_5.evaluate(val_char_token_dataset)

# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy_score = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision_score, model_recall_score, model_f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy_score,
		  "precision_score": model_precision_score,
		  "recall score": model_recall_score,
		  "f1 score": model_f1_score}
  return model_results
