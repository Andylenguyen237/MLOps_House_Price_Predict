# Kubernetes-Based Parallel Machine Learning Training with Auto-Scaling

### Objective
This project demonstrates how to set up and deploy parallel machine learning training jobs using Kubernetes on [Google Cloud Platform (GCP)](https://cloud.google.com/run/?utm_source=PMAX&utm_medium=PMAX&utm_campaign=FY24-H2-apac-gcp-DR-campaign-AU-cloud+run&utm_content=au-en&&https://ad.doubleclick.net/ddm/trackclk/N5295.276639.GOOGLEADWORDS/B26943865.344329733;dc_trk_aid=535895606;dc_trk_cid%3D163098484;dc_lat%3D;dc_rdid%3D;tag_for_child_directed_treatment%3D;tfua%3D;ltd%3D&gad_source=1&gclid=Cj0KCQjwmt24BhDPARIsAJFYKk3s4pj0WakAOgVpjYds2Q__bRLlUqSRPzxyN3d43s0ELdY_VbNLjlEaAnOhEALw_wcB&gclsrc=aw.ds&hl=en). The purpose of this setup is to enable efficient, scalable, and parallel execution of machine learning tasks with dynamic resource allocation.

### Model Training
The model training process involves building and tuning a [Random Forest Regressor](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html) package, given the train dataset retrieved from Sklearn's [Boston dataset](https://scikit-learn.org/1.0/modules/generated/sklearn.datasets.load_boston.html) to predict house prices using the Boston Housing dataset. Since the load_boston function from scikit-learn is deprecated (see more at Sklearn's [Boston dataset](https://scikit-learn.org/1.0/modules/generated/sklearn.datasets.load_boston.html), the dataset is manually loaded from a public URL - [Boston house-price data](https://lib.stat.cmu.edu/datasets/boston), consisting of 13 features representing housing attributes, and the target variable is the median house price (MEDV). The dataset is split into training and test sets using `train_test_split`. The training process includes hyperparameter tuning using `GridSearchCV` to optimize parameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`. For each combination of parameters, the model's performance is evaluated using Mean Squared Error (MSE) across 3-fold cross-validation. After tuning, the best-performing model is selected and saved for future use, while the tuning artefacts (hyperparameters and their respective scores) are stored as JSON files for analysis and review.

### Technologies
* Google Kubernetes Engine (GKE): GKE is used to create and manage a Kubernetes cluster for orchestrating machine learning training jobs.
* Docker: A Docker container is used to package the machine learning model training code along with all dependencies, ensuring consistency across different environments.
* Google Container Registry (GCR): GCR is used to store and manage Docker images that are deployed to Kubernetes.
* Kubernetes Jobs: Kubernetes Jobs are configured to run multiple parallel training tasks (jobs), where each job trains a model and tunes hyperparameters.

[!NOTE]
We will go step by step (running and testing) from the beginning to deploy a ML model 

### Set up
#### Step 0: Initialize Git Repository
Create and clone the Git repository:
   ```
   git init
   git clone <repository-url>
   ```

#### Step 1: Set Up Docker and Project Files
Create the following essential files:
`Dockerfile` 
`.dockerignore` 
`api.py` : to build and perform model tuning (written using Python's `Flask`

#### Step 2: Set Up the Python Environment with Pipenv
Initialize Pipenv to create `Pipfile` and `Pipfile.lock`:
`pipenv install <package>`

#### Step 3: Build and Push Docker Image
Build the Docker image:
`docker build --tag your-docker-hub-username/mlops_house_price_predict .`
Test the Docker image locally:
`docker run --rm --name test-api -p 5000:5000 your-docker-hub-username/mlops_house_price_predict`
Push the Docker image to Docker Hub:
`docker push your-docker-hub-username/mlops_house_price_predict`
Model and API Endpoints
`/predict [POST]`: Accepts 13 features from the user and returns the predicted house price.
`/artefacts [GET]`: Retrieves grid search trials and evaluation scores.

#### Step 4: Kubernetes Deployment
1. Set Up GCP and Kubernetes
Install the GCP SDK:
`brew install google-cloud-sdk`
Initialize GCP configuration:
`gcloud init`
Enable the Kubernetes Engine API and create a cluster:
`gcloud container clusters create house-price --num-nodes 5 --machine-type e2-medium`
2. Authenticate Docker with GCP
Configure Docker to push images to Google Container Registry:
`gcloud auth configure-docker`
Build and tag Docker image for GCP:
`docker build -t gcr.io/<project-id>/mlops_house_price_predict:latest .`
`docker tag your-docker-hub-username/mlops_house_price_predict:latest gcr.io/<project-id>/mlops_house_price_predict:latest`
Push the Docker image to Google Container Registry:
`docker push gcr.io/<project-id>/mlops_house_price_predict:latest`
Apply Kubernetes configurations:
`kubectl apply -f deploy.yaml`

[!NOTE]Troubleshooting
- [x] CrashLoopBackOff due to insufficient memory --> Solution: Adjust resource limits in the yaml configuration file.
- [ ] 


Enable autoscaling:
`kubectl autoscale deployment house-price-predict --cpu-percent=50 --min=2 --max=10`

- Testing Kubernetes Deployment
Retrieve services:
`kubectl get services`
Test the endpoint:
`curl http://<EXTERNAL-IP>/`
Parallel Hyperparameter Tuning with Kubernetes Jobs
Configure parallel jobs in hyperparameter-tuning.yaml.
Apply the configuration:
`kubectl apply -f hyperparameter-tuning.yaml`
Monitor job status:
```
kubectl get jobs
kubectl get pods
```
Autoscaling the Kubernetes Cluster
Enable cluster autoscaling:
```
gcloud container clusters update house-price \
    --enable-autoscaling \
    --min-nodes 1 \
    --max-nodes 10 \
    --zone us-central1-c
```










