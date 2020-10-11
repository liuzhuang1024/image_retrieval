import requests
import base64
import json
import os

file_name = 'img/gen_00600.jpg'
with open(file_name, 'rb') as f:
    file = f.read()
files = {'file': (os.path.basename(file_name), file, 'image/jpg')}
# 访问服务
res = requests.post("http://0.0.0.0:5000/photo", files=files)
if res:
    print(res.json())
else:
    print(res)