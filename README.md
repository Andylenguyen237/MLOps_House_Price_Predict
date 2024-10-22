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










