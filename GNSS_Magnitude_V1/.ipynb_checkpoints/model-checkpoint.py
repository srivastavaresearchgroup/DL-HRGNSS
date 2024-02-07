# -*- coding: utf-8 -*-
# ClauQC2023
# Last update April 2023
# ***************************************************************************************************
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from keras.constraints import maxnorm

# MODEL **************** Sequential-Conv2D
# Input shape: (number of stations, time windows, number of channels)
def build_model6(nst, nt, nc):
    model = Sequential()
    
    # Conv and Max-Pooling1 ____________________________________________________
    model.add(Conv2D(12,(1,3),activation='relu',input_shape=(nst, nt, nc)))
    model.add(MaxPooling2D((1,2)))
    
    # Conv and Max-Pooling2 ____________________________________________________
    model.add(Conv2D(24,(1,3),activation='relu',padding='same'))
    model.add(Conv2D(32,(1,3),activation='relu',padding='same'))
    model.add(MaxPooling2D((1,2)))
    
    # Conv and Max-Pooling3 ____________________________________________________
    model.add(Conv2D(64,(1,3),activation='relu',padding='same'))
    model.add(Conv2D(128,(1,3),activation='relu',padding='same'))
    model.add(MaxPooling2D((1,2)))
    
    # Conv ____________________________________________________
    model.add(Conv2D(256,(1,3),activation='relu'))
    
    # Flatten ____________________________________________________
    model.add(Flatten())
    
     # 128 Neurons ____________________________________________________
    model.add(Dense(128, activation='relu',kernel_initializer="normal",kernel_constraint=maxnorm(3)))    
     # 32 Neurons ____________________________________________________
    model.add(Dense(32, activation='relu',kernel_initializer="normal",kernel_constraint=maxnorm(3)))
     # 1 Neurons ____________________________________________________
    model.add(Dense(1))
    
    return model