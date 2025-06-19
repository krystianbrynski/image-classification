import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

def predict_image(img_path):
    model = load_model('model/image_model.h5')
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    
    # Mapowanie indeksów klas
    class_labels = os.listdir('dataset/train')
    return {"class": class_labels[class_index], "confidence": float(np.max(prediction))}
