from processing import Predict
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.models import load_model
import tensorflow.keras.backend as tfback
import tensorflow as tf
import argparse

# Set up argument parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--n", type=int, default=0, help="number of processes to run in parallel")
parser.add_argument("--x", type=int, default=0, help="number of videos to skip")

# Parse arguments
args = parser.parse_args()

# Extract n and x from args
n = args.n
x = args.x

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

# Set the path to the directory you want to access
#Home PC
path = r"/mnt/c/Users/Kian Kordtomeikel/Documents/Coding/Dissertation/Datasets/EchoNet-Dynamic"
#Uni HPC
# path = "../../"
# pathPSAX = "../../Datasets/EchoNet-Pediatric/PSAX"
# pathPSAX = "../../Datasets/EchoNet-Pediatric/PSAX"
# path = "../EchoNet-Dynamic"
path = "../Datasets/EchoNet-Pediatric/PSAX"

filenames = pd.read_csv(path + "/FileList.csv", usecols=["FileName"])["FileName"].tolist()

# Load filenames from csv or directory, or type one filename
# filenames = pd.read_csv("../EchoNet-Dynamic/FileList.csv", usecols=["FileName"])["FileName"].tolist()

# Load saved model
SAVED_MODEL = load_model("echoWeights.hdf5")
SEQUENCE_LENGTH = 30
STRIDE = 1

final_predictions = []
output_file = "multibeat_phase_detection_pediatrtic_psax.csv"

#check if the csv exists
try:
    df = pd.read_csv(output_file)
except FileNotFoundError:
    df = pd.DataFrame()

# if df isn't empty
if not df.empty:
    values = df[df.columns[0]].values
    values = [str(x) for x in values]
else:
    values = []

# Extract n and x from args
n = args.n
x = args.x

for i, file in zip(range(x, len(filenames), n), tqdm(filenames[x::n])):
    count = i + 1
    
    if str(count) in values:
        print("Video " + str(count) + " already processed")
        continue

    file_path = path + f"/Videos/{file}" # Complete path to video files
    
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
      prediction = SAVED_MODEL.predict(np.expand_dims(chunked_sequence[i], axis=0), verbose=0) #, verbose=0
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
    
    final_predictions.append([str(count), file, "ED", ED_predictions])
    final_predictions.append([str(count), file, "ES", ES_predictions])
    
    df = pd.DataFrame(final_predictions)
    # append to csv
    df.to_csv(output_file, mode='a', header=False, index=False)
    final_predictions = []
    
# Quit GPU session
session.close()

