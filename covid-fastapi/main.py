# Import Libraries and setup
#from flask import Flask, jsonify, g, redirect, url_for, flash, render_template, make_response
#from flask_cors import CORS, cross_origin
#import random
#import pandas as pd
import requests
import os
import datetime
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import base64
from imageio import imread
import json
import time
import logging
import uuid

# Web server 
#from gevent.pywsgi import WSGIServer
# Server
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# For Gradcam heatmap
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable

from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

#####Set logger for FastAPI
# Initialize logging
my_logger = logging.getLogger()
my_logger.setLevel(logging.ERROR)


def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path


def handle_upload_file(
    upload_file: UploadFile, handler: Callable[[Path], None]
  ) -> None:
    tmp_path = save_upload_file_tmp(upload_file)
    try:
        handler(tmp_path)  # Do something with the saved temp file
    finally:
        tmp_path.unlink()  # Delete the temp file

# Functions
def test_rx_image_for_Covid19(model, imagePath, filename):
    img = cv2.imread(imagePath)
    img_out = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)

    img = np.array(img) / 255.0

    pred = model.predict(img)
    pred_neg = int(round(pred[0][1]*100))
    pred_pos = int(round(pred[0][0]*100))

    if np.argmax(pred, axis=1)[0] == 0:
        prediction = 'Covid-19 POSITIVE'
        prob = pred_pos
    elif np.argmax(pred, axis=1)[0] == 2:
        prediction = 'Covid-19 Negative; Bacterial Penumonia Positive'
        prob = pred_pos
    else:
        prediction = 'Covid-19 Negative; Bacterial Penumonia Negative'
        prob = pred_pos

    img_pred_name = prediction+str(prob)+filename+'.png' #prediction+'_Prob_'+str(prob)+'_Name_'+filename+'.png'
    cv2.imwrite('static/result/'+img_pred_name, img_out)
    cv2.imwrite('static/Image_Prediction.png', img_out)
    print
    return prediction, prob



##########################################################
#### Define GradCam to generate heatmap for Covid-19 #####
##########################################################
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # resize the heatmap to oringnal X-Ray image size
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap 
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        # return the resulting heatmap to the calling function
        return heatmap

################################################################
# Functions to generate heatmap for a specific image and save it
################################################################
def generate_gradcam_heatmap(model, imagePath, filename):
    orignal = cv2.imread(imagePath)
    orig = cv2.cvtColor(orignal, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(orig, (256, 256))
    dataXG = np.array(resized) / 255.0
    dataXG = np.expand_dims(dataXG, axis=0)

    preds = model.predict(dataXG)
    i = np.argmax(preds[0])

    cam = GradCAM(model=model, classIdx=i, layerName='mixed10') #mixed9_1, conv2d_93, mixed10 conv2d_93 average_pooling2d_9 - find the last 4d shape
    heatmap = cam.compute_heatmap(dataXG)

    # Old fashoined way to overlay a transparent heatmap onto original image, the same as above
    heatmapY = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmapY = cv2.applyColorMap(heatmapY, cv2.COLORMAP_OCEAN)  # COLORMAP_JET, COLORMAP_VIRIDIS, COLORMAP_HOT, COLORMAP_BONE, COLORMAP_OCEAN 
    imageY = cv2.addWeighted(heatmapY, 0.5, orignal, 1.0, 0)


    #pred = model.predict(img)
    pred_neg = int(round(preds[0][1]*100))
    pred_pos = int(round(preds[0][0]*100))

    if (np.argmax(preds, axis=1)[0] == 0) and (pred_pos > 65):
        prediction = 'Covid-19 POSITIVE'
        file_pred = 'Covid19'
        prob = pred_pos
    elif np.argmax(preds, axis=1)[0] == 2:
        prediction = 'Covid-19 Negative; Bacterial Penumonia Positive'
        file_pred = 'BacPenumonia'
        prob = pred_pos
    else:
        prediction = 'Covid-19 Negative; Bacterial Penumonia Negative'
        file_pred = 'Normal'
        prob = pred_pos

    
    img_pred_name =  file_pred + '_' + str(prob) + '_' + filename.replace('.', '_') +'.png'    
    if (np.argmax(preds, axis=1)[0] == 0) and (pred_pos > 65):
        cv2.imwrite('static/result/'+img_pred_name, imageY )
    else:
        cv2.imwrite('static/result/'+img_pred_name, orignal )

    cv2.imwrite('static/Image_Prediction.png', orignal )
    print
    
    return prediction, prob, img_pred_name




UPLOAD_FOLDER = os.path.join('static', 'source')
OUTPUT_FOLDER = os.path.join('static', 'result')
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg', 'PDF', 'PNG', 'JPG', 'JPEG'])

covid_pneumo_model = load_model('./models/inceptionv3_saved.h5') #inceptionv3_saved.h5, covid_pneumo_model.h5

##################################
#### Define the WSGI server here
##################################
#app = Flask(__name__)

app = FastAPI()

origins = [ '*' ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )
 

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")

#CORS(app)
#SEND_FILE_MAX_AGE_DEFAULT = 0

prediction=' '
confidence=0
filename='Image_Prediction.png'
image_name = filename


#UPLOAD_FOLDER = UPLOAD_FOLDER
#OUTPUT_FOLDER = OUTPUT_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/hello")
async def hello():
    return {"message": "Hello World"}

@app.get("/items/{id}")
async def read_item(request: Request, id: str = "123"):
    return templates.TemplateResponse("item.html", {"request": request, "id": id})

@app.get("/")
async def hello_world(request: Request):
	#return render_template('index.html', prediction='INCONCLUSIVE', confidence=0, filename='no image')
    return templates.TemplateResponse("index.html", {"request": request, "prediction": "INCONCLUSIVE", "confidence": 0, "filename": "no image"})

# Service healthchecks
@app.get("/covid19/api/v1/healthcheck")
async def liveness():
    return 'Covid19 detector API is live!\n\n'


#@app.post("/files/upload")
#def create_file(file: UploadFile):
#    global upload_folder
#    file_object = file.file
#    #create empty file to copy the file_object to
#    upload_folder = open(os.path.join(upload_folder, file.filename), 'wb+')
#    shutil.copyfileobj(file_object, upload_folder)
#    upload_folder.close()
#    return {"filename": file.filename}

########################################
####  AI UI Rendering for uploaded file
#########################################
@app.post("/query")
#async def query(request: Request):
async def query(request: Request, file: UploadFile = File(...)):
    if request.method == 'POST':
        # RECIBIR DATA DEL POST
        #form = await request.form()
        #filename = form["file"].filename
        #contents = await form["file"].read()
        #file = form["file"].file 
            
        #global upload_folder
        filename = file.filename
        file_object = file.file
        #create empty file to copy the file_object to
        #file_object.seek(0)  # <-- This.
        upload_folder = open(os.path.join(UPLOAD_FOLDER, file.filename), 'wb+')
        shutil.copyfileobj(file_object, upload_folder)
        upload_folder.close()
        #return {"filename": file.filename}
        
        #if 'file' not in request.files:
        if filename == '' or  file == '':
            return templates.TemplateResponse("index.html", {"request": request, "prediction": "INCONCLUSIVE", "confidence": 0, "filename": "no image"})
        
        # if user does not select file, browser also
        # submit an empty part without filename
        #if filename == '':
        #    return templates.TemplateResponse("index.html", {"request": request, "prediction": "INCONCLUSIVE", "confidence": 0, "filename": "no image"})
  
        if file and allowed_file(filename):

            #filename = str(file.filename)
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            image_name = filename

            # detection covid
            try:
                #prediction, prob = test_rx_image_for_Covid19(covid_pneumo_model, img_path, filename)
                prediction, prob, img_pred_name = generate_gradcam_heatmap(covid_pneumo_model, img_path, filename)
                output_path = os.path.join(OUTPUT_FOLDER, img_pred_name)
                #return render_template('index.html', prediction=prediction, confidence=prob, filename=image_name, xray_image=img_path, xray_image_with_heatmap=output_path)
                return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction, "confidence": prob, "filename": image_name, "xray_image": img_path, "xray_image_with_heatmap": output_path })

            except:
                # return render_template('index.html', prediction='INCONCLUSIVE', confidence=0, filename=image_name, xray_image=img_path)
                return templates.TemplateResponse("index.html", {"request": request, "prediction": "INCONCLUSIVE -- here?", "confidence": 0, "filename": image_name, "xray_image": img_path })
                
        else:
            #return render_template('index.html', name='FILE NOT ALOWED', confidence=0, filename=image_name, xray_image=img_path)
            return templates.TemplateResponse("index.html", {"request": request, "prediction": "FILE NOT ALOWED", "confidence": 0, "filename": image_name, "xray_image": img_path })

#
#
#
########################################
#### AI API - Model 2 inference endpoint
########################################
@app.post('/covid19/api/v1/predict/')
async def covid_classifier_model2(request: Request):
    form = await request.form()
    img = imread(BytesIO(base64.b64decode(form['b64'])))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = image.img_to_array(img) / 255.
    img = np.expand_dims(img, axis=0)

    # this line is added because of a bug in tf_serving(1.10.0-dev)
    img = img.astype('float16')

    data = json.dumps({"signature_name": "serving_default", 
                       "instances": img.tolist()})
    
    #MODEL2_API_URL is tensorflow serving URL in another docker
    HEADERS = {'content-type': 'application/json'}
    MODEL2_API_URL = 'http://172.17.0.1:8521/v1/models/covid19/versions/2:predict'
    CLASS_NAMES = ['Covid19', 'Normal_Lung', 'Pneumonia_Bacterial_Lung']

    json_response = requests.post(MODEL2_API_URL, data=data, headers=HEADERS)
    prediction = json.loads(json_response.text)['predictions']
    prediction = np.argmax(np.array(prediction), axis=1)[0]
    prediction = CLASS_NAMES[prediction]
    #my_logger.error("Something went wrong here?")
    return JSONResponse({"model_name": "Customised IncpetionV3",
                    "X-Ray_Classification_Result": prediction, 
                    'X-Ray_Classfication_Raw_Result': json.loads(json_response.text)['predictions'], #json_response.text,
                    #'Input_Image': 'InputFilename',
                    #'Output_Heatmap': 'OutputFilenameWithHeatmap' 
                    })

# Model 2 inference endpoint with heatmap
@app.post('/covid19/api/v1/predict/heatmap')
async def covid_classifier_model2_heatmap(request: Request):
    form = await request.form()
    img = imread(BytesIO(base64.b64decode(form['b64'])))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # same the input with a temp name
    filename = time.strftime( str(uuid.uuid4()) + "%Y%m%d-%H%M%S.png")
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(img_path, img)
    

    # normalise it into 4d input per request by backend tf serving
    img = cv2.resize(img, (256, 256))
    img = image.img_to_array(img) / 255.
    img = np.expand_dims(img, axis=0)

    # this line is added because of a bug in tf_serving(1.10.0-dev)
    img = img.astype('float16')

    data = json.dumps({"signature_name": "serving_default", 
                       "instances": img.tolist()})
    
    #MODEL2_API_URL is tensorflow serving URL in another docker
    HEADERS = {'content-type': 'application/json'}
    MODEL2_API_URL = 'http://172.17.0.1:8521/v1/models/covid19/versions/2:predict'
    CLASS_NAMES = ['Covid19', 'Normal_Lung', 'Pneumonia_Bacterial_Lung']

    json_response = requests.post(MODEL2_API_URL, data=data, headers=HEADERS)
    prediction = json.loads(json_response.text)['predictions']
    prediction = np.argmax(np.array(prediction), axis=1)[0]
    prediction = CLASS_NAMES[prediction]

    # calculate and save the result heatmap
    pred, prob, img_pred_name = generate_gradcam_heatmap(covid_pneumo_model, img_path, filename)

    RESOURCE_URL_SOURCE = 'http://localhost:8056/static/source/'
    RESOURCE_URL_RESULT = 'http://localhost:8056/static/result/'

    return JSONResponse({"model_name": "Customised Incpetion V3",
                    "X-Ray_Classification_Result": pred,
                    "X-Ray_Classification_Covid19_Probability": prob / 100, 
                    'X-Ray_Classfication_Raw_Result': json.loads(json_response.text)['predictions'], 
                    'Input_Image': RESOURCE_URL_SOURCE + filename,
                    'Output_Heatmap': RESOURCE_URL_RESULT + img_pred_name 
                    })




# No caching at all for API endpoints.
#@app.after_request
#def add_header(response):
#    response.headers['Cache-Control'] = 'public, max-age=0'
#    return response

#if __name__ == '__main__':
#    http_server = WSGIServer(('0.0.0.0', 8000), app)
#    http_server.serve_forever()


if __name__ == '__main__':
   app.run()