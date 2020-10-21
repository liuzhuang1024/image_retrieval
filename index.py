from flask import request, Flask, abort
import flask
import h5py
from main import search
import config
import logging
from predict import FrozenPredict
import time

app = Flask(__name__)

logging.basicConfig(level=1)
logging.info("Loading Model....")
predict = FrozenPredict().predict

logging.info("Loading Image form H5.....")
with h5py.File("weights/image.h5", "r") as f:
    images = f["image"][:]
    logging.info(f"Images' shape is {images.shape}")


@app.route("/photo", methods=['POST'])
def get_frame():
    # 接收图片
    upload_file = request.files['file']
    print(upload_file)
    if upload_file:
        start = time.time()
        _, fc_feat, _ = predict(upload_file)
        logging.info(f"Predict cost time: {time.time()-start:0.2f}")
        start = time.time()
        res = search(images, fc_feat)
        logging.info(f"Search cost time: {time.time()-start:0.2f}")

        return flask.jsonify({'res': res})
    else:
        return "No File"


if __name__ == "__main__":
    app.run("0.0.0.0", port=5000, debug=True)
