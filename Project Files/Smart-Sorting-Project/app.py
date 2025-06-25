from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename  # Needed to safely store filenames

# Create upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = tf.keras.models.load_model("healthy_vs_rotten.h5")
class_names = ['Healthy', 'Rotten']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    if img_file:
        # Clean filename
        filename = secure_filename(img_file.filename)

        # Save the image in static/uploads
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img_file.save(img_path)

        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        # Return result with image
        return render_template('result.html', prediction=predicted_class, image_file=filename)

    return "No image uploaded"

if __name__ == '__main__':
    app.run(debug=True)
