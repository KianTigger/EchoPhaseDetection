from processing import Predict
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import load_model
# import tensorflow.python.keras.backend as tfback
# HPC doesn't need the .python
import tensorflow.keras.backend as tfback
import tensorflow as tf

# Initialise GPU session
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=True))

def _get_available_gpus():
    if tfback._LOCAL_DEVICES == None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus
tfback._get_available_gpus

tf.config.list_logical_devices()

''' Generate multibeat predictions for videos of arbitrary length '''

path = "../EchoNet-Dynamic"

# Initialise GPU session
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=True))

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()

# Open a strategy scope.
with strategy.scope():
    # Your code here
    # Load filenames from csv or directory, or type one filename
    filenames = pd.read_csv("../EchoNet-Dynamic/FileList.csv", usecols=["FileName"])["FileName"].tolist()

    # Load saved model
    SAVED_MODEL = load_model("echoWeights.hdf5")
    SEQUENCE_LENGTH = 30
    STRIDE = 1

    final_predictions = []

    # only use first video for testing
    # for file in tqdm(filenames):
    for file in tqdm(filenames):

        file_path = path + f"/Videos/{file}.avi" # Complete path to video files

        predict = Predict(file_path, SEQUENCE_LENGTH, STRIDE) # Data management class object

        frames = predict.get_frames()

        image_sequence = predict.get_image_sequence(frames)

        # Input chunked to sequence length by window and stride
        chunked_sequence = predict.get_chunked_sequence(image_sequence)

        # create empty np array for predictions
        pred=np.arange(int(len(image_sequence)),dtype=float)
        pred=np.full_like(pred,np.nan,dtype=float)

        # run sliding window predictions with stride
        start=0
        end = SEQUENCE_LENGTH

        # Generate prediction for each chunked sequence
        for i in range(len(chunked_sequence)):
          tempArr=np.arange(int(len(image_sequence)),dtype=float)
          tempArr=np.full_like(tempArr,np.nan,dtype=float)
          prediction = SAVED_MODEL(np.expand_dims(chunked_sequence[i], axis=0), training=False)
          tempArr[start:end]=prediction
          pred=np.vstack([pred,tempArr])
          start+=STRIDE
          end+=STRIDE

        # Calculate the mean of all predictions
        mean = np.nanmean(pred,axis=0)

        # remove padded frames from predictions
        predictions = np.resize(mean, mean.size-predict.num_padded_frames)

        # Get predictions for ED and ES phases
        ED_predictions, ES_predictions = predict.get_predictions(predictions)

        final_predictions.append([file, "ED", ED_predictions])
        final_predictions.append([file, "ES", ES_predictions])

# Save predictions to csv file
prediction_df = pd.DataFrame(final_predictions, columns=["Filename", "Phase", "Predictions"])
prediction_df.to_csv("predictions.csv", index=False)
    
# Quit GPU session
session.close()
