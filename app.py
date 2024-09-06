from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('D:/new/models/skin_lesion_classifier.h5')

# Label mappings
label_mapping = {'male': 0, 'female': 1, 'unknown': 2}
localization_mapping = {
    'scalp': 0, 'ear': 1, 'face': 2, 'back': 3, 'trunk': 4, 'chest': 5,
    'upper extremity': 6, 'abdomen': 7, 'unknown': 8, 'lower extremity': 9,
    'genital': 10, 'neck': 11, 'hand': 12, 'foot': 13, 'acral': 14
}

# Updated dx_mapping with full disease names
dx_mapping = {
    0: 'Benign keratosis-like lesions (BKL)',
    1: 'Melanocytic nevi (NV)',
    2: 'Dermatofibroma (DF)',
    3: 'Melanoma (MEL)',
    4: 'Vascular lesions (VASC)',
    5: 'Basal cell carcinoma (BCC)',
    6: 'Actinic keratoses and intraepithelial carcinoma (AKIEC)'
}

# Function to process the image
def process_image(image_path):
    img = Image.open(image_path)
    img = img.resize((28, 28))
    img_array = np.array(img)
    r = img_array[:, :, 0].flatten()
    g = img_array[:, :, 1].flatten()
    b = img_array[:, :, 2].flatten()
    flat_image = np.concatenate([r, g, b])
    return flat_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        sex = request.form['sex']
        localization = request.form['localization']
        
        # Get the uploaded image
        image = request.files['image']
        image_path = "static/" + image.filename
        image.save(image_path)
        
        # Process the image
        flat_image = process_image(image_path)
        flat_image = np.array(flat_image).reshape(-1, 28, 28, 3)
        flat_image = (flat_image - np.mean(flat_image)) / np.std(flat_image)
        
        # Process the non-image data
        sex_encoded = label_mapping[sex]
        localization_encoded = localization_mapping[localization]
        non_image_data = np.array([sex_encoded, localization_encoded, age]).reshape(1, -1)
        non_image_data = np.nan_to_num(non_image_data, nan=-1)
        
        # Make the prediction
        prediction = model.predict([flat_image, non_image_data])
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = dx_mapping[predicted_class]
        
        return render_template('index.html', prediction=predicted_label, image_path=image_path)
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
