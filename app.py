from flask import Flask, render_template, request
from compare_images import compare_images
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model(r"D:\Tattoo acuracy\tattoo_accuracy_classifier.h5")

# Define function to preprocess images
def preprocess_image(image_path):
    img = Image.open(image_path)  # Use Image from PIL
    img = img.resize((150, 150))  # Resize the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    artist_image = request.files['artist_image']
    sketch_image = request.files['sketch_image']

    artist_image.save('static/artist_image.jpg')
    sketch_image.save('static/sketch_image.jpg')

    accuracy_artist_tattoo, accuracy_sketch_tattoo = compare_images('static/artist_image.jpg', 'static/sketch_image.jpg')

    return render_template('result.html', accuracy_artist_tattoo=accuracy_artist_tattoo, accuracy_sketch_tattoo=accuracy_sketch_tattoo)

if __name__ == '__main__':
    app.run(debug=True)
