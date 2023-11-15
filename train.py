## Train a model to classify the images
#
## Import libraries

import os
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt


from tensorflow import keras
np.random.seed(0)

import src.config as config
from src.dataloader import read_train, read_val
from src.preprocessing import CustomDataGenerator
from src.results import plot_history, generate_report,\
                            save_params, write_summary
from src.utils import setup_logger, header
from src.results import generate_report
import argparse
from keras.regularizers import l2,l1,l1_l2


## Set up command line parameters
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='experiment')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--ep', type=int, default=10)
parser.add_argument('--outpath', type=str, default=config.OUTPUT_FOLDER)
parser.add_argument('--partition', type=str, default="80_10_10")
parser.add_argument('--gpu', type=int, default=1)

if parser.parse_args().gpu == 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


## Set up constants
LEARNING_RATE = parser.parse_args().lr
BATCH_SIZE = parser.parse_args().bs
EPOCHS = parser.parse_args().ep
EXPNAME = parser.parse_args().name
PARTITION = parser.parse_args().partition
EVENTNAME = f"pt_{PARTITION}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}_ep_{EPOCHS}"
OUTPUT_FOLDER = os.path.join(parser.parse_args().outpath, EXPNAME, EVENTNAME)
   

PARTITION_FOLDER = os.path.join(config.INPUT_PATH, f'train_{PARTITION}')

## Create output folder                 
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

## Set up logger
logger = setup_logger(os.path.join(OUTPUT_FOLDER, 'train.log'))

## Write parameters
save_params(OUTPUT_FOLDER, lr=LEARNING_RATE, bs=BATCH_SIZE, ep=EPOCHS, partition=PARTITION)

## Load data
logger.info(header('LOAD DATA'))

X_train, y_train = read_train(path=PARTITION_FOLDER)
output_shape = np.unique(y_train).shape[0]
logger.info(f"TRAINING SIZE: {X_train.shape}")
logger.info(f"NUMBER OF CLASSES: {output_shape}")

X_val, y_val = read_val(path=PARTITION_FOLDER)
logger.info(f"VALIDATION SIZE: {X_val.shape}")
logger.info(f"NUMBER OF CLASSES: {np.unique(y_val).shape[0]}")

X_test, y_test = read_val(path=PARTITION_FOLDER)
logger.info(f"TEST SIZE: {X_test.shape}")
logger.info(f"NUMBER OF CLASSES: {np.unique(y_test).shape[0]}")

## Define architecture
logger.info(header('DEFINE MODEL'))

layers = [
            keras.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, config.N_CHANNELS)),
            keras.layers.Conv2D(32,(3,3), activation = 'relu'),
            keras.layers.MaxPooling2D(pool_size = (2, 2)),

            keras.layers.Conv2D(64,(3,3), activation = 'relu'),
            keras.layers.MaxPooling2D(pool_size = (2, 2)),

            keras.layers.Conv2D(128,(3,3), activation = 'relu'),
            keras.layers.Conv2D(256,(3,3), activation = 'relu'),

            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(output_shape, activation='softmax')
]

model = keras.Sequential(layers)

model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

write_summary(model, OUTPUT_FOLDER)

## Train model
logger.info(header('TRAINING DATA'))
train_generator = CustomDataGenerator(X_train, y_train, batch_size=BATCH_SIZE)
print(type(X_train))

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10)

model.fit(train_generator, epochs=EPOCHS, 
          validation_data=(X_val, y_val), 
          callbacks=[early_stopping])

model_file = os.path.join(OUTPUT_FOLDER, f'model.h5')
model.save(model_file)

## Evaluate model
logger.info(header('WRITE'))
logger.info(f"Write model in: {model_file}")

fig = plot_history(model)
plt.suptitle(f"Learning rate: {LEARNING_RATE} - Batch Size: {BATCH_SIZE}"
             f" - Partition: {PARTITION.replace(r'_', r'-')}", fontsize=16)
fig.savefig(os.path.join(OUTPUT_FOLDER, f'history.png'))


folder_suffix = ["train", "val", "test"]

for suff in folder_suffix:
    path = os.path.join(OUTPUT_FOLDER, suff)
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)

    X = eval(f"X_{suff}")
    y = eval(f"y_{suff}")
    logger.info(header(f"{suff.upper()} dataset"))
    generate_report(model, X, y, path)

logger.info(header('END'))