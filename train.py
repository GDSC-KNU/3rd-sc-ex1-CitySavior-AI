import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Conv2D, Activation,BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense, Input, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.applications import ResNet50V2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError
from keras import optimizers, regularizers

# ImageDataGenerator 생성
datagen_train = ImageDataGenerator(rescale=1./255)
datagen_val = ImageDataGenerator(rescale=1./255)

# flow_from_directory 함수를 이용하여 이미지 데이터를 불러옴
train_generator = datagen_train.flow_from_directory(
    directory='Dataset/Classified_Training',  # 훈련 데이터가 있는 폴더의 경로
    target_size=(224, 224),  # 이미지 크기
    batch_size=16,  # 배치 크기
    class_mode='categorical', 
    color_mode='rgb',  # 컬러 이미지이므로 'rgb'
    shuffle=True  # 데이터를 섞음
)

validation_generator = datagen_val.flow_from_directory(
    directory='Dataset/Classified_Validation',  # 검증 데이터가 있는 폴더의 경로
    target_size=(224, 224),  # 이미지 크기
    batch_size=16,  # 배치 크기
    class_mode='categorical',  
    color_mode='rgb',  # 컬러 이미지이므로 'rgb'
    shuffle=True  # 데이터를 섞음
)    

epochs             = 70
weight_decay       = 1.25e-4

def residual_block(inputs, filters, k=2, strides=1, stack_n=1):
    shortcut = inputs
    
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(inputs)
    x = Activation('relu')(x)
    
    x = Conv2D(k*filters, kernel_size=3, strides=strides, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters*k, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    
    if strides != 1 or inputs.shape[-1] != filters*k:
        shortcut = Conv2D(filters*k, kernel_size=1, strides=strides, padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(weight_decay))(shortcut)
        
    x = Add()([x, shortcut])
    
    for i in range(1, stack_n):
        shortcut = x
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Conv2D(k*filters, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters*k, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = Add()([x, shortcut])
    
    return x


def wide_resnet(input_shape, filters):
    depth = 28
    width = 10
    stack  = (depth - 4) // 6
    inputs = Input(shape=input_shape)
    
    x = Conv2D(filters[0], kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    
    x = residual_block(x, filters[0], k=10, strides=1, stack_n=stack)
    x = residual_block(x, filters[1], k=10, strides=2, stack_n=stack)
    x = residual_block(x, filters[2], k=10, strides=2, stack_n=stack)
    
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    outputs = Dense(3, activation='softmax', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(weight_decay))(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


filters = [8, 8, 16]


def scheduler(epoch):
    lr_min=1e-5
    lr_max = 3e-1
    cos_inner = (math.pi * epoch) / 70
    lr = lr_max / 2 * (math.cos(cos_inner) + 1)
    return max(lr, lr_min)


# 모델 컴파일
Callback = [LearningRateScheduler(scheduler)]
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)

model = wide_resnet(input_shape=(224, 224, 3), filters=filters)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])
model.summary()

# 모델 학습
history = model.fit(
      train_generator,
      epochs=70,
      validation_data=validation_generator,
      verbose=1)

#모델 세이브
model.save('model/ResNet50V2_fine_tuned.h5')
tf.saved_model.save(model, 'fine_tuned_saved_model')
