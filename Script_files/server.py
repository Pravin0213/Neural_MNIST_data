from flask import Flask, request, jsonify, send_file , render_template
import numpy as np
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

# --- Your Neural Network Code ---
# Replace with your actual neural network (e.g., from Nielsen's book)
class Network:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a.flatten()
    

def center_image(pixels):
    """Center the digit within the 28x28 image."""
    img = pixels.reshape(280, 280)
    rows = np.any(img > 0, axis=1)
    cols = np.any(img > 0, axis=0)

    # Find bounding box
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    # Crop the bounding box
    cropped = img[top:bottom+1, left:right+1]

    # Resize cropped image to smaller square
    from scipy.ndimage import zoom
    h, w = cropped.shape
    scale = 20 / max(h, w)
    resized = zoom(cropped, scale)

    # Pad back to 28x28, centered
    new_img = np.zeros((28, 28))
    h2, w2 = resized.shape
    top_pad = (28 - h2) // 2
    left_pad = (28 - w2) // 2
    new_img[top_pad:top_pad+h2, left_pad:left_pad+w2] = resized

    new_img = np.clip(new_img, 0, 1)
    new_img = np.round(new_img, 2)
    return new_img.reshape(-1, 1)


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for numerical stability
    return e_x / e_x.sum()

# Load your weights and biases

import pickle
with open('weights.pkl', 'rb') as f:
    weights = pickle.load(f)
with open('biases.pkl', 'rb') as f:
    biases = pickle.load(f)

net = Network(weights, biases)    # Placeholder for testing


@app.route('/')
def home():
    return render_template('base.html')

# Digit recognizer page
@app.route('/recognizer')
def recognizer():
    return render_template('recognizer.html')

# About page
@app.route('/about')
def about():
    return render_template('about.html')



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    pixels = np.array(data['pixels'], dtype=np.float32)
    pixels = center_image(pixels)
    print(pixels)
    output = net.feedforward(pixels)
    if output.shape != (10,):  # Ensure output is a 1D array with 10 elements
        return jsonify({'error': 'Invalid output shape from network'}), 500
    
    probabilities = softmax(output)
    prediction = np.argmax(probabilities)
    confidences = [
        {'digit': i, 'prob': round(float(probabilities[i] * 100),2)} for i in range(10)
    ]

    img = Image.fromarray((pixels.reshape(28, 28) * 255).astype(np.uint8), mode='L')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')   



    return jsonify({
        'prediction': int(prediction),
        'confidences': confidences,
        'image': img_base64
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)