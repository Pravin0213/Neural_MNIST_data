from flask import Flask, request, jsonify, send_file
import numpy as np

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
        return a

# Load your weights and biases
try:
    import pickle
    with open('weights.pkl', 'rb') as f:
        weights = pickle.load(f)
    with open('biases.pkl', 'rb') as f:
        biases = pickle.load(f)
except FileNotFoundError:
    # Placeholder for testing
    weights = [np.random.randn(16, 784), np.random.randn(10, 16)]
    biases = [np.random.randn(16), np.random.randn(10)]

net = Network(weights, biases)

@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pixels = np.array(data['pixels'], dtype=np.float32)
    output = net.feedforward(pixels)
    prediction = np.argmax(output)
    confidences = [
        {'digit': i, 'prob': float(output[i] * 100)} for i in range(10)
    ]
    confidences.sort(key=lambda x: x['prob'], reverse=True)
    return jsonify({
        'prediction': int(prediction),
        'confidences': confidences
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)