from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "Fake Review Detection API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data['review']
    prediction = model.predict([review])[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
