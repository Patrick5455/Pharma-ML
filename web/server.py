from flask import Flask, render_template

app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():
    model = "Serve Model"
    return render_template("home.html", model=model)


@app.route("/predict", methods=['POST'])
def prediction():
    return render_template("predictions.html")


@app.route("/visualize", methods=['POST'])
def visualize():
    return render_template("visualize.html")


if __name__ == '__main__':
    app.run(host='localhost', port=3030, debug=True)
