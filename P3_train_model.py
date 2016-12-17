import tensorflow as tf
import os
import numpy as np
from keras.models import Sequential
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
            r_path = '/home/robbie/Documents/simulator-linux-50hz/trainY/'
            nm = r_path + row[0]
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

#flip the horizontal axis and use negative steering angles
X_train = np.append(X_train,X_train[:,:,::-1,:], axis=0)
y_train = np.append(y_train, np.negative(y_train))

print ('total number of images: ', len(X_train))
print ('total driving time: ', len(X_train)/50.0, 's')

X_train, X_val, y_train, y_val = train_test_split( 
#     X_train_gray, #Use grayscale data
    X_train,
    y_train,
    test_size=0.15,
    random_state=832289)


#SETUP MODEL ----------------------------
print('^^^Define the Model^^^')
batch_size = 128
nb_epoch = 6
# size of pooling area for max pooling
pool_size = (2, 2)

# input_shape = (img_rows, img_cols, 3)

# row, col, ch = 160, 320, 3  # camera format
row, col, ch = 64, 128, 3
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))

#C1---------------------------
nb_filters1 = 24
kernel1 = (6,6)

model.add(Convolution2D(nb_filters1, kernel1[0], kernel1[1],subsample=(2, 2), border_mode='same'))
model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.1))
#C2----------------------------
nb_filters2 = 42
kernel2 = (5,5)

model.add(Convolution2D(nb_filters2, kernel2[0], kernel2[1],subsample=(2, 2), border_mode='same'))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.1))
#C3----------------------------
nb_filters3 = 54
kernel3 = (4,4)

model.add(Convolution2D(nb_filters3, kernel3[0], kernel3[1],border_mode='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# #C4 # #----------------------------
nb_filters4 = 64
kernel4 = (3,3)

model.add(Convolution2D(nb_filters4, kernel4[0], kernel4[1],border_mode='valid',))
# model.add(MaxPooling2D(pool_size=pool_size))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# #C4 # #----------------------------
# nb_filters5 = 64
# kernel5 = (9,9)

# model.add(Convolution2D(nb_filters5, kernel5[0], kernel5[1],border_mode='same'))
# # model.add(ELU())
# model.add(Activation('relu'))


model.add(Flatten())
#L1
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))
#L2
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.1))
#L3
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))
#L4
model.add(Dense(12))
model.add(Activation('relu'))


model.add(Dense(1,name='output'))
# model.add(Activation('softmax'))

model.summary()




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