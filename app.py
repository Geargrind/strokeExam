from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['post'])
def predict():
    try:
        model = load_model('strokeModel.keras')
        gender = request.form.get('gender')
        age = request.form.get('age')
        hypertension = request.form.get('hypertension')
        heart_disease = request.form.get('heart_disease')
        ever_married = request.form.get('ever_married')
        work_type = request.form.get('work_type')
        Residence_type = request.form.get('Residence_type')
        avg_glucose_level = request.form.get('avg_glucose_level')
        bmi = request.form.get('bmi')
        smoking_status = request.form.get('smoking_status')

        if None in [gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]:
            return render_template('index.html', result='Missing input(s)')
        else:
            # Transform inputs to match model's expected input format
            input_data = [
                float(age),
                int(hypertension),
                int(heart_disease),
                float(avg_glucose_level),
                float(bmi),
                1 if gender == 'Female' else 0,
                1 if gender == 'Male' else 0,
                0,  # gender_Other not used
                1 if ever_married == 'No' else 0,
                1 if ever_married == 'Yes' else 0,
                1 if work_type == 'Govt_job' else 0,
                1 if work_type == 'Never_worked' else 0,
                1 if work_type == 'Private' else 0,
                1 if work_type == 'Self-employed' else 0,
                1 if work_type == 'children' else 0,
                1 if Residence_type == 'Rural' else 0,
                1 if Residence_type == 'Urban' else 0,
                1 if smoking_status == 'Unknown' else 0,
                1 if smoking_status == 'formerly smoked' else 0,
                1 if smoking_status == 'never smoked' else 0,
                1 if smoking_status == 'smokes' else 0
            ]
            arr = np.array([input_data])
            predictions = model.predict(arr)
            return render_template('index.html', result=str(predictions[0][0]))
    except Exception as e:
        return render_template('index.html', result='error ' + str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
