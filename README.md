
Default: docker-compose up -d

Scale-up: docker-compose up --scale fastapi=2 --scale flask=2 -d

Full documentation will be pulished here:
https://community.intersystems.com/post/deploy-mldl-models-api-service-stacks

Note: Tensorflow exported models are ommited in the models directory, since they are ~250M (larger than 100M by Github). I will upload these large files seperately.  
The models are exported from Jupyter pipelines at here: 
https://community.intersystems.com/post/run-some-covid-19-lung-x-ray-classification-and-ct-detection-demos) and 
https://community.intersystems.com/post/explainability-and-visibility-covid-19-x-ray-classifiers-deep-learning
