import pickle
with open('weights.pkl', 'rb') as f:
    weights = pickle.load(f)
with open('biases.pkl', 'rb') as f:
    biases = pickle.load(f)

print(weights, biases)