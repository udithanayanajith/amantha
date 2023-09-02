from flask import Flask, request, jsonify
import numpy as np
import pickle  

app = Flask(__name__)

with open('disorderModel.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = np.array(list(request.json.values())).reshape(1, -1)
        prediction = model.predict(user_input)[0]
        probabilities = model.predict_proba(user_input)[0]
        predicted_probability = probabilities[model.classes_.tolist().index(prediction)] * 100
        response = {
            'condition': prediction,
            'probability': f'{predicted_probability:.0f}%'
        }
    

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
