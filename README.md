# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
- This dataset contains information related to a direct marketing campaign of a Portuguese banking institution and if their clients did subscribe for a term deposit.<br>
We seek to predict if clients are subscibed for a term deposit based on the provided personal and financial information.

- The best performing model is a SoftVotingClassifier (VotingEnsemble), containing MaxAbcScaler and XGBoostClassifier which implements the Gradient tree boosting algorithm_  well known for its efficiency to predict accuracies. 
This model has been chosen among several models based on the highest accuracy, while tuning the hyperdrive parameters during the experiment.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

Below a general pipeline architecture is provided to explain the workflow of using Azure SDK with HyperDrive parameterization:

![alt_text](SklearnPipelineArchitecture.PNG)

Getting good or bad results is related to how well prepared your model is? <br>
### **Data Preparation:** <br>
The first part of the process will be to prepare the script to use. The main performed actions inside the _train.py_ script are: <br>
   - Retrieve data from the specified path using the TabularDatasetFactory method.<br>
   - Convert the dataset to a binary data representation by applying the _SKLearn get_dummies() method_, to use later with <br>
   - Split data into training and testing sets, using Sklearn function  *train_test_split* .<br>
   - Apply the SKLearn _LogisticRegression classifier_ by including the sample parameters, then fit the split data.<br>
   - Finally, verify the calculated accuracy and register the best model.<br>
The problem with manually handling predictions is the time lost while trying to tune the parameters seeking the best result.<br>
Hence, the benefit of Azure HyperDrive is finding the perfect fit by tuning the hyperdrive parameters among a pre-specified random set of your choice.<br>

### **Data Training & Validation:** <br>
This phase is a repeatable one as it will be running for each run of the experiment, specifying a random hyperparameter from a given list.
To prepare the HyperDrive configuration, we need to set three major parameters including:<br>

   1- Specify a parameter sampler: since we are using the SKLearn _LogisticRegression classifier_  we will be using:<br>
   
   - The inverse of regularization strength _**C**_ with a default value of _1.0_, you need to specify a discrete set of options to sample from.<br>
   - And, the maximum number of iterations taken for the solvers to converge _**max_iter**_ <br>
      
   2- Specify an early termination policy: Among three types, we decided to work with the _Bandit Policy_, classified as an aggressive saving, as it will terminate any job based on a _slack_ criteria, and a _frequency_ and _delay_ interval for evaluation. <br>
   
   - slack_factor: Specified as a ratio used to calculate the allowed distance from the best performing experiment run.<br>
   - evaluation_interval: Reflects the frequency for applying the policy.<br>
   - delay_evaluation: Reflects the number of intervals for which to delay the first policy evaluation.<br>
      
   3 - Create a SKLearn estimator to use later within the HyperDriveConfig definition.<br>
   The estimator contains the _source directory_ The path to the script directory, the _compute target_, and the _entry script_ The name of the script to use along with the experiment. <br>
   
After creating the HyperDriveConfig using the mentioned above parameters, we submit the experiment by specifying the recently created HyeperDrive configuration.<br>

 ### **Model Deployment:** <br>
 This phase is related to provisioning, visioning, access control, and scaling. However, as a first project, we focused on the registration of the  best-run model and how to explore its different metrics and features using Microsoft Azure ML studio tools.<br>
 
And since it's directly related to CI/CD concept with Azure, we can proceed with an endpoint deployment for further use of the generated model.<br>
 
**What are the benefits of the parameter sampler you chose?**

When we talk about using a parameter sampler, we need to highlight two steps:

   - **The hyperparameter type: Discrete or Continuous?** In our case, we used the discrete type because this project is about categorization. 
   - **The sampling type: Grid or Random or Bayesian sampling?**. <br> Based on the previous workshops, both grid and random yielded good results. However, and because all our hyperparameters values are discrete, we must apply the grid sampling.<br>
  
Regarding the definition of this project search space, I used  _C_ and _iter_max_ hyperparameters by creating for each a dictionary with the appropriate parameter expression.

That being said, parameter sampler benefits are:<br>

   - Automate the finding hyperparameters configuration process that results in the best performance by simultaneously running experiments and optimizing hyperparameters.
   - Decrease the process of computing expenses, errors, and trials' number.

**What are the benefits of the early stopping policy you chose?**

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
