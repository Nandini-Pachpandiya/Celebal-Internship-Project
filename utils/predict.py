import numpy as np
import cv2

IMG_SIZE = 128  # Same as training

def preprocess_and_predict(image, model, labels):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, IMG_SIZE, IMG_SIZE, 3)
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction, axis=1)[0]
    pred_label = labels[pred_index]
    return pred_label, prediction  
    # return labels[pred_index]
