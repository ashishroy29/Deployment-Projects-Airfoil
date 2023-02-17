### 29th May 2022 class ineuron

import pickle
from flask import Flask , request , app, jsonify , url_for , render_template
import numpy as np
import pandas as pd

#To start a flask app
app = Flask(__name__)

#Loading a Pickle file
model = pickle.load(open('model.pkl','rb'))

###HOME OF HTML
@app.route('/home')
def home():
     return render_template('home.html')


#Creating an API (single)
@app.route('/predict_api',methods = ['POST'])
def predict_api():

     data = request.json['data']
     print(data)
     new_data = [list(data.values())]
     output = model.predict(new_data)[0]
     print(output)
     return jsonify(output)

##HTML
@app.route('/predict',methods = ['POST']) #forHTML
def predict():

     data = [float(x) for x in request.form.values()]
     final_features = [np.array(data)]
     print(data)
     output = model.predict(final_features)[0]
     print(output)
     return render_template('home.html', prediction_text = "Airfoil pressure is {}".format(output))


#This is the command where the point of execution will start
if __name__ == "__main__":
    app.run(debug = True)


#Postman Input
# {
# "data" : {
#     "Frequency":1250,
#     "Angle of Attack":12.3,
#     "Chord Length":0.1016,
#     "Free-stream velocity":31.7,
#     "Suction side" : 0.041876
# }
# }