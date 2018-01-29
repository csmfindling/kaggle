## Kaggle competition :
State Farm Distracted Driver Detection : 
https://www.kaggle.com/c/state-farm-distracted-driver-detection

## Goal :

Predict the state of the driver

## Model

The model is based on vgg like networks

K. Simonyan and A. Zisserman <a href="https://arxiv.org/pdf/1409.1556.pdf" class="underline"> Very Deep Convolutional Networks for Large-Scale Image Recognition </a> \

## Script informations : 

- training contains the main script
- data_server_train and data_server_valid launches hdf5 servers
- models contains all the nn models
- exploration is a sandbox folder
- build_dataset must be launched once in the beginning to create the hdf5 files
- functions contains various useful functions


## Information on the dataset:

- there are 26 drivers in the training dataset
- each drivers has about 800 photos
- the validation set was created with 3 drivers
- the training set with 23 drivers

## Usage

- launch bokeh, server_train, server_valid
- update the model imported in train.py
- run `python train.py`

# Feature extraction

Install sklearn-theano: `pip install git+https://github.com/sklearn-theano/sklearn-theano`
