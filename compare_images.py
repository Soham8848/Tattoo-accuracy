from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np

# Load the trained model
model = load_model(r"D:\Tattoo acuracy\tattoo_accuracy_classifier.h5")

# Define function to preprocess images
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

# Define function to predict class
def predict_class(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return prediction[0][0]  # Assuming it's a binary classification, if not, modify accordingly

# Compare two images
def compare_images(image_path1, image_path2):
    accuracy_image1 = predict_class(image_path1)
    accuracy_image2 = predict_class(image_path2)

    return accuracy_image1, accuracy_image2
