from keras.models import load_model
import cv2
import numpy as np
import os
model = load_model('saved_models/13-0.96.hdf5')
print(model.summary())


classes_={
    0: "paper",
    1: "rock",
    2: "scissors",
}
def mapper(val):
    return classes_[val]


def prediction(file):
    # prepare the image
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200,200))
    img=img/255
    # predict the move made
    pred = model.predict(np.array([img]))
    move_code = np.argmax(pred[0])
    move_name = mapper(move_code)

    print("Predicted: {}".format(move_name))


classes_files=os.listdir("image_data/test")

for class_file in classes_files:
    file_path="image_data/test/"+class_file
    for image_name in os.listdir(file_path):
        prediction("image_data/test/"+class_file+"/"+image_name)




