
1. Build the docker
docker build -t ai-service:0.0.1 .
docker build -t ai-service:0.0.2 .

2. Run the docker of Flask (with or without GPU)
docker run -it --rm -v "/zhong/flask-xray/covid19:/app" -p 8051:5000 --name ai-svr1 ai-service:0.0.1
docker run -it --rm --runtime=nvidia -v "/zhong/flask-xray/covid19:/app" -p 8051:5000 --name ai-svr1 ai-service:0.0.1
docker run -it --rm -v "/zhong/flask-xray/covid19:/app" -p 8051:5000 --name ai-svr1 ai-service:0.0.2

3. Start tensorflow serving dockers (with and without GPU) - Tensorflow serving:
docker run -itd --rm -p 8511:8501 --mount type=bind,source=/zhong/flask-xray/covid19/covid19_models,target=/models/covid19 -e MODEL_NAME=covid19 --name tf-svg1 -t tensorflow/serving
docker run -itd --rm  --runtime=nvidia -p 8521:8501 --mount type=bind,source=/zhong/flask-xray/covid19/covid19_models,target=/models/covid19 -e MODEL_NAME=covid19 --name tf-svg2 -t tensorflow/serving:latest-gpu




-----------------------
## Run with Docker

# 1. Build Docker image
$ docker build -t medicalui .

# 2. Run!
$ docker run -it --rm -p 5000:5000 medicalui

Open http://localhost:5000 and wait till the webpage is loaded.

------------------------------------------------------------------------------
## Local Installation

# 1. Install Python packages
$ pip install -r requirements.txt

# 2. Run!
$ python app.py
```

Open http://localhost:5000 and have fun. :smiley:

------------------------------------------------------------------------------


### Use your own model

Place your trained `.h5` file saved by `model.save()` under models directory.

------------------------------------------------------------------------------

## Deployment

To deploy it for public use, you need to have a public **linux server**.

### Run the app

Run the script and hide it in background with `tmux` or `screen`.
```
$ python app.py
```

You can also use gunicorn instead of gevent
```
$ gunicorn -b 127.0.0.1:5000 app:app
```
------------------------------------------------------------------------------
