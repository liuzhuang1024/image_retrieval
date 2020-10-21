from flask import request, Flask, abort
import flask
from main import search
import config
import logging
from predict import FrozenPredict

app = Flask(__name__)

logging.basicConfig()
logging.info("Loading Model....")
predict = FrozenPredict().predict


@app.route("/photo", methods=['POST'])
def get_frame():
    # 接收图片
    upload_file = request.files['file']
    print(upload_file)
    if upload_file:
        _, fc_feat, _ = predict(upload_file)
        res = search(config.image, fc_feat)
        return flask.jsonify({'res': res})
    else:
        return "No File"


if __name__ == "__main__":
    app.run("0.0.0.0", port=5000, debug=True)
