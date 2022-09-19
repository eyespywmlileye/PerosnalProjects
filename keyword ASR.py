import tensorflow as tf 
import numpy as np 
import librosa 
import tensorflow_datasets as tfds 
import pathlib 

from tensorflow.keras import layers 
from tensorflow import keras 
from tensorflow.keras import models 

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

path = "C:\\Users\\thibe\\OneDrive\Documents\\My Personal M.L Projects\\Keyword ASR\\data\\mini_speech_commands"

# Get data 

data_dir = pathlib.Path(path)
if not data_dir.exists(): 
    tf.keras.utils.get_file(
              'mini_speech_commands.zip',
      origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
      extract=True,
      cache_dir='.', cache_subdir='data')

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']



filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)

test_file = tf.io.read_file(path+'/down/0a9f9af7_nohash_0.wav')
test_audio, _ = tf.audio.decode_wav(contents=test_file)

train_files = filenames[:6400]
val_files = filenames[6400: 6400 + 800]
test_files = filenames[-800:]

#####Preprocess Data 

# function to decode all audio 

def decode_audio(audio_binary):
  # Decode WAV-encoded audio files to `float32` tensors, normalized
  # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  # Since all the data is single channel (mono), drop the `channels`
  # axis from the array.
  return tf.squeeze(audio, axis=-1)
    

    #Create a funtion that creates labels using the parent directory for each file 

def get_label(file_path):
  parts = tf.strings.split(
      input=file_path,
      sep=os.path.sep)
  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-2]


    # Create a function that decodes the wav file and labels it 
    
def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

AUTOTUNE = tf.data.AUTOTUNE

files_ds = tf.data.Dataset.from_tensor_slices(train_files)

waveform_ds = files_ds.map(map_func = get_waveform_and_label, 
num_parallel_calls = AUTOTUNE) 

def get_spectrogram(waveform):
  # Zero-padding for an audio waveform with less than 16,000 samples.
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  # Cast the waveform tensors' dtype to float32.
  waveform = tf.cast(waveform, dtype=tf.float32)
  # Concatenate the waveform with `zero_padding`, which ensures all audio
  # clips are of the same length.
  equal_length = tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  label_id = tf.math.argmax(label == commands)
  return spectrogram, label_id

spectrogram_ds = waveform_ds.map(map_func= get_spectrogram_and_label_id, 
num_parallel_calls= AUTOTUNE) 


def preprocess_dataset(files): 
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(map_func= get_waveform_and_label , num_parallel_calls= AUTOTUNE)
    output_ds = output_ds.map(map_func= get_spectrogram_and_label_id , num_parallel_calls= AUTOTUNE)
    return output_ds 

train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

BATCH_SIZE = 64 

train_ds = train_ds.batch(BATCH_SIZE) 
val_ds = val_ds.batch(BATCH_SIZE) 

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape

norm_layer = layers.Normalization()
# Fit the state of the layer to the spectrograms
# with `Normalization.adapt`.
norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

early_stopping = tf.keras.callbacks.EarlyStopping(patience= 5 )
checkpnt = "C:\\Users\\thibe\\OneDrive\\Documents\\My Personal M.L Projects\\Keyword ASR\\model_checpoints"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = checkpnt , save_best_only= True )

num_labels = len(commands)


model_0 =models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32 , 3, activation='relu'),
    layers.Conv2D(64 ,3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels , activation = "softmax"),
])

model_0.summary()

model_0.compile( loss = tf.keras.losses.SparseCategoricalCrossentropy(), 
optimizer = tf.keras.optimizers.Adam(), 
metrics = ['accuracy'],)

history = model_0.fit( train_ds , 
epochs = 20 ,
validation_data = val_ds)

model_0.evaluate(val_ds)