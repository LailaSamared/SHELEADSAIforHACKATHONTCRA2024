import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('HDL.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    final_features = np.array(final_features)
    prediction = model.predict(final_features)

    output = prediction[0]
    if output == 3:
        output_label = "Very High"
    elif output == 2:
        output_label = "High"
    elif output == 1:
        output_label = "Medium"
    else:
        output_label = "Low"

    return render_template('index.html', prediction_text=f'Human development level: {output_label}')


if __name__ == "__main__":
    app.run(debug=True)