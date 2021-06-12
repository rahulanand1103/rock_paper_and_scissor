import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPool2D,BatchNormalization,GlobalAveragePooling2D,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


img_width,img_height=200,200
channel=3

datagen=ImageDataGenerator(rescale=1./255,validation_split=0.30)

train_data_genetator=datagen.flow_from_directory('image_data/train',target_size=(200,200),color_mode='rgb',
                                                 class_mode='categorical',batch_size=16,subset = "training")



val_data_genetator=datagen.flow_from_directory('image_data/train',target_size=(200,200),color_mode='rgb',
                                                 class_mode='categorical',batch_size=16,subset = "validation")


from tensorflow.keras.applications import VGG16
model_train = VGG16(weights='imagenet',include_top=False,input_shape=(200,200,channel))

x = Flatten()(model_train.output)
prediction = Dense(3, activation='softmax')(x)

from tensorflow.keras.models import Model
# create a model object
model_train = Model(inputs=model_train.input, outputs=prediction)

#Add checkpoints 
from keras.callbacks import ModelCheckpoint
filepath="saved_models/weights-improvement-{epoch:02d}.hdf5" #File name includes epoch and validation accuracy.
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


model_train.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history =model_train.fit_generator(train_data_genetator,
                         steps_per_epoch = len(train_data_genetator),
                         epochs = 15,
                         validation_data = val_data_genetator,
                         validation_steps = len(val_data_genetator),
                         verbose=1
                         )

