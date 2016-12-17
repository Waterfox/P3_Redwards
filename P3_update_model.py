import tensorflow as tf
import os
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Convolution2D
from keras.layers import Input, Dense, Activation, Flatten, Lambda, ELU, MaxPooling2D, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import csv
import matplotlib.image as mpimg
import argparse
import json
import cv2


#IMPORT THE DATA
parser = argparse.ArgumentParser(description='Remote Driving')
parser.add_argument('train_path', type=str,help='Path to driving_log.csv')
args = parser.parse_args()



read_path_a = args.train_path+'/'
data_file_a = 'driving_log.csv'

out_path = read_path_a

X_train=list()
y_train=list()
i=0

print('^^^Reading data^^^')
with open(read_path_a+'/'+data_file_a, 'r') as csvfile_a:
    reader_a = csv.reader(csvfile_a, delimiter=',')
    for row in reader_a:
        if len(row[0])>10: #clears header

            ##Open Normally 
#             X_train.append(mpimg.imread(row[0]))

            ##Window and chagnge color space ------0000000000
#             X_train.append(cv2.cvtColor(mpimg.imread(row[0])[50:140,:,:], cv2.COLOR_RGB2YUV))

            ##Use Udacity training data--------------------
            # r_path = '/home/robbie/Documents/simulator-linux-50hz/trainY/'
            # nm = r_path + row[0]
            nm = row[0]
            X_train.append(cv2.resize(mpimg.imread(nm)[:,:,:], (0,0), fx=0.4, fy=0.4))

            ##Use resized and grayscale data----------------
#           X_train.append(cv2.cvtColor(cv2.resize(mpimg.imread(nm)[30:150,:,:], (0,0), fx=0.6, fy=0.6), cv2.COLOR_RGB2GRAY))

            #Use windowed data----------------
#           X_train.append(mpimg.imread(nm)[30:150,:,:])
            y_train.append(float(row[3]))
            i+=1
            


# print (type(X_train))
X_train=np.array(X_train)
y_train=np.array(y_train)
# print (type(X_train))

# #flip the horizontal axis and use negative steering angles
X_train = np.append(X_train,X_train[:,:,::-1,:], axis=0)
y_train = np.append(y_train, np.negative(y_train))

print ('total number of images: ', len(X_train))
print ('total driving time: ', len(X_train)/50.0, 's')
print ('shape of X_train array:', np.shape(X_train))

X_train, X_val, y_train, y_val = train_test_split( 
#     X_train_gray, #Use grayscale data
    X_train,
    y_train,
    test_size=0.25,
    random_state=832289)

print('^^^Open Model^^^')
#OPEN MODEL ----------------------------
model_file = 'steering_angle'
#Open the model at the top level dir
with open(model_file+'.json', 'r') as jfile:
    model = model_from_json(json.load(jfile))

weights_file = model_file+ '.h5'
if os.path.exists(weights_file):
    model.load_weights(weights_file)
    print ("model weights opened")

model.summary()

batch_size = 128
nb_epoch = 5

print('^^^Train Model^^^')
##--------------TRAIN MODEL--------------------
model.compile(loss='mse',
              optimizer=Adam())


model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_val, y_val))

print("Saving model weights and configuration file.")

# if not os.path.exists("./"+out_path):
    # os.makedirs("./"+out_path)

#Save the latest model to the data dir for reference
model.save_weights("./"+out_path+"steering_angle.h5", True)
with open('./'+out_path+'steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)


#Save the latest model to the top level dir
model.save_weights("./steering_angle.h5", True)
with open('./steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)