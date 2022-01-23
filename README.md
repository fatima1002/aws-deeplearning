# Image Classification using AWS SageMaker

The work carried out is part of the AWS Machine Learning Engineer Nanodegree Program. It uses AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. It has been done with he provided dog breed classication data set. 

The project has been carried out with the following: 
1. Project Set Up and Installation: Download the data and upload to S3 to be accessed from Sagemaker. 
2. Choose pre-train CNN and train model with hyperparameter tuning.
3. Using the best parameters ,re-train the model with AWS Debugger and Profiler. 
4. Deploy the model.

The following provides more detail:

1. Training jobs:
    1. Screenshot of hyperparameter training job:
    2. Metrics during training 
    
    1. Screenshot of best estimator training job with profiler and debugger:
    2. Metrics during training:


2. Debugging and Profiling
Debugger helps us understand what is happening when the model is training, this helps us see if ther are issues specifically with vanishing gradients, overfitting and the loss not decreasing. Profiler helps us check how well out model is training and inspect CPU/GPU utilisation , CPU/GPU memory utilisation aswell as instance metrics. This has been perfomed by adding debugger and profiler hooks into the training script. 

Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?
From debugger we can view the cross entropy loss:

The profiler report was saved in S3, please view [this file] to view the report.

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
The model was deployed to an endpoint. Yoy may send dogimages to the endpoint and expect a prediction. 


**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.


