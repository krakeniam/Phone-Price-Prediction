
from flask import Flask, render_template,request
import pickle

import numpy as np
model=pickle.load(open('mobilePrediction.pkl','rb'))
app= Flask(__name__)
@app.route("/")
def home():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    values=[int(x)for x in request.form.values()]
    final=np.array(values)
    prediction=model.predict(final.reshape(1,-1))
    output=prediction[0]
    return render_template('index.html',send_value=f"Expexted price of the phone is ({output})")

if __name__=='__main__':
    app.run(debug=True)