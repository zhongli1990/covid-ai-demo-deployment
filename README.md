## Short description
*"Covid-19 AI demo in all-Docker"* deployment including dockerised Flask, FastAPI, Tensorflow Serving and HA Proxy etc etc.

## Full documentation 
Full documentation is pulished [Deploy ML/DL models into a consolidated AI demo service stack](https://community.intersystems.com/post/deploy-mldl-models-api-service-stacks)

## Scope
#### In scope:
As a jump start, we can simply use docker-compose to deploy the following dockerised components into an AWS Ubuntu server

* *HAProxy*  - load balancer
* *Gunicorn* vs. Univorn  - web gateway server
* *Flask* vs. *FastAPI* - application server
* Tensorflow-Serving vs. Tensorflow-Serving-gpu - application back-end servers for image etc classifications etc
* IRIS *IntegratedML* - consolidated App+DB AutoML with SQL interface
* Python3 in *Jupyter Notebook* to emulate a client for benchmarking
* Docker and *docker-compose*
* AWS *Ubuntu* 16.04 with a Tesla T4 GPU 
_Note_:   Tensorflow Serving  with GPU is for demo purpose only - you can simply switch off the gpu related image (in a dockerfile) and the config (in the docker-compose.yml).

#### Out of scope or on next wish list:

* Ningx or Apache etc web servers are omitted in demo for now
* RabbitMQ and Redis  - queue broker for reliable messaging that can be replace by IRIS or Ensemble.   
* IAM (Intersystems API Manger) or Kong is on wish list
* SAM (Intersystems System Alert & Monitoring) 
* ICM (Intersystems Cloud Manager) with Kubernetes Operator - always one of my favorites since its birth
* FHIR (Intesystems IRIS based FHIR R4 server and FHIR Sandbox for SMART on FHIR apps)
* CI/CD devop tools or Github Actions

## Deployment Pattern
![Logical Deployment Pattern](https://community.intersystems.com/sites/default/files/inline/images/images/image(870).png)

## Environment Topology
![Physical Deployment Topology](https://community.intersystems.com/sites/default/files/inline/images/images/image(891).png)

## Volumes Mapping & Directory Structure
Please refer to [full documentation](https://community.intersystems.com/post/deploy-mldl-models-api-service-stacks) on section "Dockerised Components" 

## Service start-up
Default start-up: `docker-compose up -d`
Scale-up start-up: `docker-compose up --scale fastapi=2 --scale flask=2 -d`

## Test App
[Sample application on AWS](http://ec2-18-134-16-118.eu-west-2.compute.amazonaws.com:8056).  Note: this service is on a temp AWS address and not up 24/07.

## Functional Testing and API docs:
Please see section "2. Test demo APIs" within [full documentation](https://community.intersystems.com/post/deploy-mldl-models-api-service-stacks) on section "Dockerised Components" 

## Benchmark testing
Please see section "3. Benchmark-test demo APIs" within [full documentation](https://community.intersystems.com/post/deploy-mldl-models-api-service-stacks) on section "Dockerised Components" 

Full documentation is pulished [here](https://community.intersystems.com/post/deploy-mldl-models-api-service-stacks)

### Note
Tensorflow exported models are ommited in the models directory, since they are **~250M** (larger than 100M by Github). I will upload these large files seperately.

The models are exported from Jupyter pipelines at [here](https://community.intersystems.com/post/run-some-covid-19-lung-x-ray-classification-and-ct-detection-demos) and [here](https://community.intersystems.com/post/explainability-and-visibility-covid-19-x-ray-classifiers-deep-learning)
