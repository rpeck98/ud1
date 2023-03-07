# Optimizing an ML Pipeline in Azure

## Overview

This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources

- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)

## Summary

This datasset contains information about bank clients such as age, job type, education, mariatal status and other indetifying features. The goal of the model is to predict whether or not an individual is likely to subscribe to a term deposit product. The original dataset is located at the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).

Several different approaches were tried to discover the best performing model (e.g. highest accuracy metric as the *primary metric*). Overall, between the HyperDrive experiments and the AutoML experiments, the best performing model was discovered by the AutoML. It is the VotingEnsamble with an accuracy score of approximatly 91.8%.

## Scikit-learn Pipeline

The scikit-learn pipeline consists of the following steps:

1. Get a reference to the ML Workspace
2. Create a new compute cluster or get reference to an existing one
3. Setup the experiment including: parameter samples, termination policy, environment configuration from the YAML file, and the training script information
4. Create a new experiment
5. Submit the experiment
6. Monitor experiment for output
7. Retrieve and save the best model
8. Clean up resources by deleting the compute cluster (if this is only experiment)

The data used in this expriment is the bank marketing data (link above in the Summary section) that contains demographic information about an individual and whether or not that individual subscribed to a term deposit product. 

Data preparation included:

- Drop NA for missing values
- Converting word values such as 'married' to a numeric 0 or 1
- One hot encoding for convrting categorical values into dummy values

The classification algorithm used in the train.py script is Logistic Regression. Logistic Regression is the "go-to method for binary classification problems (problems with two class values)" according to [Machine Learning Mastery](https://machinelearningmastery.com/logistic-regression-for-machine-learning/). Logistic Regression predicts the probability that an item belongs to one of two classes. In our specific case, the prediction is 'y' or 'n' that the individual subscribed to a term deposit product. 

Through several experimental runs, I discovered by reviewing the trial output metrics on the Accuracy chart plot and the Parallel coordiate chart that the best performing hyperparameters tended to be small values for C and a moderate/high number of iteration. Given these observations, I narrowed the ranges for C to values between 0.001 and 1. For values of max_iteration, iterations over 800 did not produce significantly better results, thus I slected four possible values of 100, 200, 400 and 800.

Early termination policy is managed by the BanditPolicy with slack_factor of 0.2 and evaluation_interval of 1. This means that the early termination policy will be applied on every iteration and terminate when (best metric/(1+slack_factor)) at the next iteration is outside the slack factor.

## AutoML

AutoML discovered the best algorithm to be a VotingEnsamble. The model consisted of 11 different algorithms each contributing a vote to the result of the classification.

## Pipeline comparison

The HyperDrive experiment produced accuracy of approximatly 90.88% in about 13-14 minutes. By contrast, the AutoML experiment produced accuracy of approximatly 91.8 in about 30 minutes (run terminated to constrain expenses). One could ask whether the additional 1% is worth the additional processing time of double over the HyperDrive experiments.

Overall the AutoML experiments were signficantly easier to setup and execute with fewer parameters needing to be considered. If one did not know what algorithm to start with (LogisticRegression), AutoML could produce some initial direction to further investigate and tune via HyperDrive.

## Future work

Ideas for future work:

- Utilize HyperDrive to further optimize the VotingEnsemble model
- Try additional HyperDrive hyperparameters on the existing LogisticRegression algorithm
- Investigate additional data cleansing or feature engineering possible with the dataset.

Attempting additional experiments with refined hyperparameters could further help to improve accuracy of the model selected.

Further enhancing the data through feature engineering could help the model improve accuracy by utilizing additional features in the probabilty calculations.

## Proof of cluster clean up

Code to cleanup the cluster was included in the Jupyter notebook.
