from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from flask import Flask, render_template, request
import numpy as np

model = load_model('DiabetesModel', compile=True)
model.load_weights('weights.hdf5')

def get_prediction(paramList):
    ''' Return the predicted output from the model'''

    param = [ 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age' ]

    x = {param[i] : np.array([float(paramList[i])]) for i in range(len(param))}
    prediction = model.predict(x)
    K.clear_session()

    return float(np.squeeze(prediction))

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def hello():
    ''' Homepage '''

    return render_template('main.html')

@app.route('/result', methods=["GET", "POST"])
def result():
    ''' Result page '''

    if request.method == "POST":

        prg = request.form["Pregnancies"]
        glc = request.form["Glucose"]
        blp = request.form["BloodPressure"]
        skt = request.form["SkinThickness"]
        ins = request.form["Insulin"]
        bmi = request.form["BMI"]
        dpf = request.form["DiabetesPedigreeFunction"]
        age = request.form["Age"]

        pred = get_prediction([prg, glc, blp, skt, ins, bmi, dpf, age])
        print('Prediction by model', pred)
        
        return render_template('result.html', prediction=round(pred*100, 2))

if __name__ == '__main__':
    app.run()
