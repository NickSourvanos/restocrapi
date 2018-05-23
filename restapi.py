from flask import Flask, jsonify, render_template, request
import os
import cv2
import pytesseract
from PIL import Image
import urllib.response
import numpy as np
import uuid

app = Flask(__name__, static_folder='images')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():

    target = os.path.join(APP_ROOT, 'images')
    print("Target path: %s" %target)

    if not os.path.isdir(target):
        os.mkdir(target)

    if request.method == 'POST' or request.method == 'GET':
        file = request.files["file"]
        filename = file.filename
        destination = "\\".join([target, filename])
        print("Destination: %s" %destination)
        print("Filename: %s" %filename)
        file.save(destination)
        image = cv2.imread(destination)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        dst = cv2.fastNlMeansDenoising(th2, 10, 10, 7)
        random_image_name = str(uuid.uuid4()) + '.jpg'
        cv2.imwrite('./images/' + random_image_name, dst)
        cao = Image.open('./images/' + random_image_name)
        extracted_text = pytesseract.image_to_string(cao, lang='eng')
        print("The result is {}".format(extracted_text))

    return render_template("complete.html", image_name=random_image_name, text = extracted_text)

def url_to_image(url):
    resp = urllib.response.openurl(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


if __name__ == '__main__':
    app.run(debug=True)

