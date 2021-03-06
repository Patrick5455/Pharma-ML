from app import app
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, flash, redirect, request
from app.forms import LoginForm
from app.prediction_notes import sales_detail, \
    sales_title, cust_detail, cust_title, time_detail, \
    time_title, analysis_detail, analysis_title, business_questions
import pickle
from app.models.model_classes import Preprocessing, Regressor

columns = ["Customers", "Open", "Promo", "Promo2",

           "StateHoliday", "SchoolHoliday", "CompetitionDistance",
           ""
           "Date", "StoreType", "Assortment", ]


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


model = CustomUnpickler(open("app/models/model1.pkl", 'rb')).load()

# with open('app/models/model1.pkl','rb') as fh:
#     model = pickle.load(fh)

#model = pickle.load(open("app/models/model1.pkl", "rb"))

# user = {"username": "Miguel"}


# other web views

@app.route('/')
@app.route('/home')
def home():
    return render_template("index.html", user=user)


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash('Login requested for user {}, remember_me={}'.format(
            form.username.data, form.remember_me.data))
        return redirect("/")
    return render_template("login.html", title="Sign In", form=form)


@app.route("/about", methods=['GET'])
def about():
    model = "Serve Model"
    return render_template("about.html", model=model)


# predictions and analysis views

@app.route("/predict", methods=['POST'])
def predict():
    # to:do add logic to determine which model to load based the type pf prediction

    model_result = request.form.values()
    # print(model_result)
    features = []
    for val in model_result:
        if "-" in val:
            dt = datetime.strptime(val, '%Y-%m-%d')
            date = dt.date()
            features.append(date)
        else:
            features.append(int(val))
    [print(f, type(f)) for f in features]
    data = np.array(features)
    data=data.reshape(1, len(columns))
    data = pd.DataFrame(data, columns=columns)

    print(data.columns.values)

    data = model.predict(data)

    prediction = model.predict(data)

    return render_template("prediction.html", prediction="prediction")


@app.route("/analysis", methods=['GET'])
def analysis():
    return render_template("analysis.html", prediction_name=analysis_title,
                           prediction_detail=analysis_detail,
                           business_questions=business_questions)


@app.route("/time", methods=['GET'])
def time_series():
    return render_template("time_series.html", prediction_name=time_title,
                           prediction_detail=time_detail)


@app.route("/cust", methods=['GET'])
def cust_churn():
    return render_template("cust_churn.html", prediction_name=cust_title,
                           prediction_detail=cust_detail)


@app.route("/sales", methods=['GET'])
def sales_forecast():
    return render_template("sales_forecast.html", prediction_name=sales_title,
                           prediction_detail=sales_detail)


@app.route("/pred_charts", methods=['GET'])
def show_charts(ana_type):
    return render_template("pred_charts.html")

# if __name__ == '__main__':
#     app.run(host='localhost', port=3030, debug=True)
