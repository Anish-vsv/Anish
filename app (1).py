from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pclass = int(request.form['Pclass'])
        sex = 1 if request.form['Sex'] == 'female' else 0
        age = float(request.form['Age'])
        sibsp = int(request.form['SibSp'])
        parch = int(request.form['Parch'])
        fare = float(request.form['Fare'])

        features = np.array([[pclass, sex, age, sibsp, parch, fare]])
        prediction = model.predict(features)

        result = "Survived" if prediction[0] == 1 else "Did Not Survive"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
