from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

app = Flask(__name__, template_folder="template", static_folder='static')
model = pickle.load(open('flower_classifier.pkl', 'rb'))  # import model

def model_output(path):
    raw_img = image.load_img(path, target_size=(64, 64))
    raw_img = image.img_to_array(raw_img)
    raw_img = np.expand_dims(raw_img, axis=0)
    raw_img = raw_img / 255.0  # Normalizing the image data
    probabilities = model.predict(raw_img)[0]

    flower = ['Rose', 'Sunflower', 'Tulip']

    max_prob_index = np.argmax(probabilities)
    max_prob = probabilities[max_prob_index]

    if max_prob > 0.5:
        result = f"It's {flower[max_prob_index]} image"
    else:
        result = "flower not confidently detected"

    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    # Save the uploaded file temporarily
    temp_path = 'temp_image.jpg'
    file.save(temp_path)

    # Process the image and get the result
    result = model_output(temp_path)

    # Save the processed image with the result
    output_image_path = 'static/output_image.png'
    img = cv2.imread(temp_path)
    plt.imshow(img)
    plt.savefig(output_image_path)  # Save the image with the result
    plt.close()

    return render_template('index.html', output_image=output_image_path, output_text=result)

if __name__ == '__main__':
    app.run(debug=True)