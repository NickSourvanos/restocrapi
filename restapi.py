from flask import Flask, jsonify, render_template, request
import os
from matplotlib import pyplot as plt
import cv2
import pytesseract
from PIL import Image
import urllib.response
import numpy as np
import json
import uuid
import tempfile
import base64

app = Flask(__name__, static_folder='images')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGE_SIZE = 1800
BINARY_THREHOLD = 180


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])
def upload():

    target = os.path.join(APP_ROOT, 'images')

    if not os.path.isdir(target):
        os.mkdir(target)

    if request.method == 'POST' or request.method == 'GET':
        destination = "./images/"
        random_image_name = str(uuid.uuid4()) + '.jpg'
        full = destination + random_image_name

        data = request.stream.read()  # .decode('utf-8')
        jsonified = json.loads(data)

        dictvalue = jsonified.get('image')

        imgdata = base64.b64decode(dictvalue)
        with open(full, 'wb') as file:
            file.write(imgdata)


        image = cv2.imread(full)
        image = cv2.resize(image, (0, 0), fx=1.8, fy=1.8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gaus = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 115, 1)

        dst = cv2.fastNlMeansDenoising(gaus, 10, 10, 7)
        enhanced_contrast = enhance_contrast(dst)

        smoothed = image_smoothing(enhanced_contrast)

        random_image_name = str(uuid.uuid4()) + '.jpg'
        cv2.imwrite('./images/' + random_image_name, smoothed)
        cao = Image.open('./images/' + random_image_name)
        extracted_text = pytesseract.image_to_string(cao, lang='eng')
        print("The result is: {}".format(extracted_text))



    return extracted_text


def remove_noise(image):
    filtered = cv2.adaptiveThreshold(image.astype(np.uint8), 255,
                    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((10, 10), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothing(filtered)
    or_image = cv2.bitwise_or(img, closing)
    return or_image


def image_smoothing(image):
    ret1, th1 = cv2.threshold(image, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def enhance_contrast(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(image.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[image]
    return img2



if __name__ == '__main__':
    app.run(host='192.168.2.14', debug=True)



    """
        random_image_name = str(uuid.uuid4()) + '.jpg'
        destination = "\\".join([target, random_image_name])
        print("Destination: %s" % destination)
        print("Filename: %s" % random_image_name)
        random_image_name.save(destination)
        cv2.imwrite(random_image_name, imageData)

        #print("Image Data: %s" %imageData)


        file = request.files["image"]
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
        """

    #ΜΕΤΑΤΡΟΠΗ JSON ΣΕ BASE64

    """
        destination = "./images/"
        random_image_name = str(uuid.uuid4()) + '.jpg'
        full = destination + random_image_name


        #data = {'image' : request.json['image']}
        data = request.get_json()
        image = data['image']
        print("Image: %s" %image)


        enc = image.encode()
        encoded_data = base64.b64encode(enc)
        print("Encoded data: %s" %encoded_data)
        print("Encoded data type: %s" %type(encoded_data))
        img = Image.open(BytesIO(encoded_data))
        img.show()

    """
    #LATEST IMAGE PROCESSING
    """
        file = request.files["file"]
        filename = file.filename
        destination = "\\".join([target, filename])
        print("Destination: %s" % destination)
        print("Filename: %s" % filename)
        file.save(destination)
        image = cv2.imread(destination)


        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        dst = cv2.fastNlMeansDenoising(gray, 10, 10, 7)
        #img = cv2.GaussianBlur(dst, (9,9), 10.0)

        hist, bins = np.histogram(dst.flatten(), 256, [0,256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        plt.plot(cdf_normalized, color='b')
        plt.hist(dst.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        img2 = cdf[dst]
        edges = cv2.Canny(img2, 100, 200)


        random_image_name = str(uuid.uuid4()) + '.jpg'
        cv2.imwrite('./images/' + random_image_name, edges)
        cao = Image.open('./images/' + random_image_name)
        extracted_text = pytesseract.image_to_string(cao, lang='eng')
        print("The result is {}".format(extracted_text))
    """
    #Convert JSON to Base64
    """
        destination = "./images/"
        random_image_name = str(uuid.uuid4()) + '.jpg'
        full = destination + random_image_name

        data = {'image' : request.json['image']}
        image = data['image']

        print("Image: %s" % data)
        print("Image data: %s" % image)

        enc = image.encode()
        encoded_data = base64.b64encode(enc)
        with open(full, 'wb') as file:
            file.write(base64.decodebytes(encoded_data))
            print(type(file))
        print("Encoded data: %s" % encoded_data)
    """
    # Latest conversion
    """
    
        destination = "./images/"
        random_image_name = str(uuid.uuid4()) + '.jpg'
        full = destination + random_image_name

        data = request.stream.read()#.decode('utf-8')
        #img = {"image" : data.replace("image=", "")}

        bytesEnc = base64.encodebytes(data)
        bytesDec = base64.decodebytes(bytesEnc)

        print("Bytes Encoded: ", bytesEnc)
        print(type(bytesEnc))
        print("Bytes Decoded: ", bytesDec)
        print(type(bytesDec))

        #img_data = json.dumps(img)
        bString = base64.b64encode(data)
        dString = base64.b64decode(bString)


        print("Encode Base64: ", bString)
        print(type(bString))
        print("Decode Base64: ", dString)
        print(type(dString))
        print("Original Data: ", data)
        print(type(data))
        #print(type(img_data))

        #print(img_data)
        #img_to_bytes = img.get('image').encode()
        #print("Bytes: %s" %type(img_to_bytes))

        #with open(full, 'wb') as file:
        #    file.write(img_to_bytes)

        #obj = json.loads(data)

        #json_data = json.loads(data)
        #stringified = json.dumps(json_data)

        #bytes_data = stringified.encode()
        #with open(full, 'wb') as file:
        #    file.write(bytes_data)

        #print("Image: %s" % data)
        #print("Type of data: %s" %type(data))
        #print("Image: %s" % img)
        #print("Json Data: %s" % json_data)
        #print("Json Data Type: %s" %type(json_data))
        #print("Stringified: %s" % stringified)
        #print("Stringified Type: %s" %type(stringified))
        #print("Json Image type : %s" %type(json_image))
        #print("Image String: %s" % image_string)
        #print("Jsonified: %s" %json_image)

    """

    # Features Extractions
    """
    mser = cv2.MSER_create()
    image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    vis = image.copy()

    regions = mser.detectRegions(gray)

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))

    """
    #Contrast
    """
    hist, bins = np.histogram(dst.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(dst.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[dst]
    edges = cv2.Canny(img2, 100, 200)
    """
    #Contours
    """
        ret, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        image_final = cv2.bitwise_and(gray, gray, mask=mask)
        ret, new_img = cv2.threshold(image_final, 180, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        # to manipulate the orientation of dilution , large x means
        # horizonatally dilating  more, large y means vertically dilating more
        dilated = cv2.dilate(new_img, kernel, iterations=9)
        _, contours, hierarchy = cv2.findContours(dilated,
                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # get contours

        for contour in contours:
            # get rectangle bounding contour
            [x, y, w, h] = cv2.boundingRect(contour)

            # Don't plot small false positives that aren't text
        if w < 35 and h < 35:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
    """

    #Silly Image Rotation
    """
            angle = 0
            rotation = 0
            while rotation < 19:

                (h, w) = edges.shape[:2]
                center = (w / 2, h / 2)
                scale = 1.0
                M = cv2.getRotationMatrix2D(center, angle, scale)
                rotated = cv2.warpAffine(image, M, (h, w))
                cv2.imwrite("./images/" + str(rotation) + ".jpg", rotated)
                cao = Image.open('./images/' + str(rotation) + ".jpg")
                extracted_text = pytesseract.image_to_string(cao, lang='eng')
                rotation += 1
                angle += 20
                if extracted_text:
                    break

            """

    #Ideal Image processing for ocr
    """
        file = request.files["file"]
        filename = file.filename
        destination = "\\".join([target, filename])
        print("Destination: %s" % destination)
        print("Filename: %s" % filename)
        file.save(destination)
        image = cv2.imread(destination)
        image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
        #retval, threshold = cv2.threshold(image, 12, 255, cv2.THRESH_BINARY)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #retval2, threshold2 = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
        #gaus = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
         #                            cv2.THRESH_BINARY, 115, 1)

        dst = cv2.fastNlMeansDenoising(gray, 10, 10, 7)
        enhanced_contrast = enhance_contrast(dst)

        smoothed = image_smoothening(enhanced_contrast)

        random_image_name = str(uuid.uuid4()) + '.jpg'
        cv2.imwrite('./images/' + random_image_name, smoothed)
        cao = Image.open('./images/' + random_image_name)
        extracted_text = pytesseract.image_to_string(cao, lang='eng')
        print("The result is: {}".format(extracted_text))
        filters_applied = 'Image Smoothening, enhanced contrast,' \
                          'No Thresholding'
                          
    def remove_noise(file_name):
        filtered = cv2.adaptiveThreshold(file_name.astype(np.uint8), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
        kernel = np.ones((10, 10), np.uint8)
        opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        img = image_smoothening(filtered)
        or_image = cv2.bitwise_or(img, closing)
        return or_image

    def image_smoothening(img):
        ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
        ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th2, (1, 1), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th3
    
    def enhance_contrast(image):
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        plt.plot(cdf_normalized, color='b')
        plt.hist(image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        img2 = cdf[image]
        return img2


    """