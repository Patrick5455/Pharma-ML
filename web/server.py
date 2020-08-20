from flask import Flask

app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():
    return


if __name__ == '__main__':
    app.run(host='localhost', port=3333, debug=True)
