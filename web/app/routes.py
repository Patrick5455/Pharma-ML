from app import app
from flask import Flask, render_template, request, jsonify
import pickle


def load_model(path="web/models/", model_name=''):
    pickle_in = open(path + model_name, "rb")
    model = pickle.load(pickle_in)

    return model


user = {"username": "Miguel"}


@app.route('/')
@app.route('/home')
def home():
    return render_template("about.html")

# @app.route("/about_us", methods=['GET'])
# def about():
#     model = "Serve Model"
#     return render_template("about.html", model=model)
#
#
# @app.route("/predict", methods=['POST'])
# def prediction():
#     # to:do add logic to determine which model to load based the type pf prediction
#
#     model = load_model()
#     return render_template("prediction.html", model=model)
#
#
# @app.route("/analysis", methods=['GET'])
# def analysis():
#     return render_template("analysis.html")

#
# if __name__ == '__main__':
#     app.run(host='localhost', port=3030, debug=True)
