from flask import request, Flask, abort
import base64
import cv2
import flask
import numpy as np
from main import get_model, extract_feature, image_preproces_v2, search, image_preproces
import config
app = Flask(__name__)

model = get_model()
# model.predict()

@app.route("/photo", methods=['POST'])
def get_frame():
    # 接收图片
    upload_file = request.files['file']
    print(upload_file)
    if upload_file:
        image_data = image_preproces(upload_file)
        _, fc_feat, _ = model([image_data])
        res = search(config.image, fc_feat)
        return flask.jsonify({'res': res})
    else:
        abort(400)


if __name__ == "__main__":
    app.run("0.0.0.0", port=5000, debug=True)
