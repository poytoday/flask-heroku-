from flask import Flask
from flask import request
from flask_cors import CORS,cross_origin
from markupsafe import escape
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/<name>")
def hello(name):
    return f"Hello, {escape(name)}!"
    
@app.route("/iris",methods=['POST'])
@cross_origin()
def predict_species():
    model = joblib.load('iris.model')
    req = request.values['param']
    inputs = np.array(req.split(','),dtype=np.float32).reshape(1,-1)
    predict_target = model.predict(inputs)
    if predict_target == 0:
        return 'Setosa'
    elif predict_target == 1:
        return 'Versicolour'    
    else:
        return 'Virginica'


@app.route("/area",methods=['GET'])
@cross_origin()
def area():
    w = float(request.values['w'])
    h = float(request.values['h'])
    return str(w*h)


@app.route("/bmi",methods=['GET'])
@cross_origin()
def bmi():
    w = float(request.values['weight'])
    h = float(request.values['hight'])
    return str(w/(h*h))

# if __name__ == '__main__':
#     app.run(host='0.0.0.0',port=5000,debug=False)