import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model1.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index1.html")

@flask_app.route("/predict1", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if prediction==1:
        return render_template("index1.html", prediction_text = "you have diabetes")
    else:
        return render_template("index1.html", prediction_text = "you have no diabetes")

if __name__ == "__main__":
    flask_app.run(debug=True)