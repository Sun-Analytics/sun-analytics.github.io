---
layout: splash-narrow
permalink: /workshops/sagemaker/
hidden: true
toc: true
toc_sticky: true
header:
  overlay_color: "#5e616c"
  overlay_image: /assets/images/workshops/sagemaker/background-page.png
#   actions:
#     - label: "<i class='fas fa-download'></i> Install now"
#       url: "/docs/quick-start-guide/"
title: "Amazon SageMaker Training"
excerpt: >
  This 1-day training provides you with hands-on experience to build, train, and deploy machine learning (ML) models on Amazon SageMaker. Welcome to [contact us](mailto:info@sun-analytics.nl?subject=[Workshop]%3ASageMaker%20training)
---

## SageMaker in General
Amazon SageMaker is a fully managed service that allows developers and data scientists can use to build, train, and deploy machine learning models. There are a lot of components to SageMaker, whether you’re using the manged development environments, the ephemeral training clusters, the hyperparamter tuning, or the deployed endpoint

Amazon SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning (ML) models quickly. SageMaker removes the heavy lifting from each step of the machine learning process to make it easier to develop high quality models.

Traditional ML development is a complex, expensive, iterative process made even harder because there are no integrated tools for the entire machine learning workflow. You need to stitch together tools and workflows, which is time-consuming and error-prone. SageMaker solves this challenge by providing all of the components used for machine learning in a single toolset so models get to production faster with much less effort and at lower cost.


Amazon SageMaker is a fully managed machine learning service. With SageMaker, data scientists and developers can quickly and easily build and train machine learning models, and then directly deploy them into a production-ready hosted environment. It provides an integrated Jupyter authoring notebook instance for easy access to your data sources for exploration and analysis, so you don't have to manage servers. It also provides common machine learning algorithms that are optimized to run efficiently against extremely large data in a distributed environment. With native support for bring-your-own-algorithms and frameworks, SageMaker offers flexible distributed training options that adjust to your specific workflows. Deploy a model into a secure and scalable environment by launching it with a few clicks from SageMaker Studio or the SageMaker console. Training and hosting are billed by minutes of usage, with no minimum fees and no upfront commitments.

## SageMaker workshop
An Amazon SageMaker Immersion Day provides our customers with hands-on experience to build, train, and deploy machine learning (ML) models quickly. SageMaker removes the heavy lifting from each step of the machine learning process to make it easier to develop high quality models. It is specifically designed to help us accelerate a customer opportunity for Machine-learning workload in AWS.

Amazon SageMaker Immersion Days leverage a modular content format, allowing you to select from ready-made presentations and labs and adapt your curriculum to your customer’s needs.

After attending an Amazon SageMaker Immersion Day our customers must be able to champion the value AWS could bring to their organization, even if they are not yet experts in AWS.

Amazon SageMaker Immersion Day help customers and partners to provide end to end understanding of building ML use cases from feature engineering to understanding various in-built algorithm and Train , Tune and Deploy the ML model in production like scenario. It guides you to bring your own model and perform on-premise ML workload Lift-and-Shift to Amazon SageMaker platform. It further demonstrate advance concept like Model Debugging , Model Monitoring and AutoML and guide to evaluate your machine leaning workload through AWS ML Well-architect lens.

## 
It's ideal (but not required) for participants to have a data science or analytics background.

## Goal
Empower AWS champions within the customer organization and help us advance an opportunity.

Discover additional use cases within an account for increased AWS adoption.

Provide hands-on experience to build Machine Learning use cases in AWS

## What this workshop is not
An Amazon SageMaker Immersion Day is not a substitute for AWS Training and Certification. If your customer needs in-depth training, reach out to the T&C team at AWS Training or engage your local Training BDM. The Training BD team can help organize training courses, from introductory to advanced level.

Amazon SageMaker Immersion Days are typically not used for demand generation or prospecting activities. If there isn’t an existing Opportunity in SFDC, you should ask whether an Immersion Day is the right solution.


## Material
- Amazon SageMaker Technical Deep Dive: https://www.youtube.com/playlist?list=PLhr1KZpdzukcOr_6j_zmSrvYnLUtgqsZz


## setup account
The customer should use a AWS account that is not running production systems.

Any accounts should be created a minimum of three days ahead of time. It takes time for new accounts to be completely ready, payment methods to be confirmed, and limits to be set.

If your customer will share an account between multiple people:

Ensure they create IAM accounts for each user that will take labs.
Adjust account limits to support the number of students doing labs. Note this process can take up to a week, depending on the specific limits.
Check limits for EC2 instances, VPCs (# of VPCs, # of IGWs, # of security groups, EIPs), ELBs, etc. The limits that need to be adjusted depends on the labs that will be delivered.
SAs can increase some limits via BubbleTea, for EC2 limits the Account SA should help the customer request the limit increase through the AWS Console, and then contact Support once there is a case number assigned.
More information about limits can be found on the AWS Service Limits page.

## During your Immersion Day
Use up to date software: Ensure students are using a recent version of Edge, Chrome, Firefox, or Safari.

Students using Windows laptops should have a SSH client (like PuTTY) installed.

Generate and send a CSAT survey to the customer. A report of your survey responses will be emailed to you 3 days after your event.

Post-event clean-up: If the customer was using their own AWS accounts, remind them that resources left running in their own AWS accounts will result in a bill. Help them clean up any resources still running.

Ensure that participants not only terminate Sagemaker Notebook instance but also stop any SageMaker deployment endpoint.

## After your Immersion Day
Capture any new opportunities generated by the Immersion Day in Salesforce.

Record your Immersion Day using the SA Activity Tracker.

Post-event clean-up: If the customer was using their own AWS accounts, contact them 2-3 days after the Immersion day and remind them that resources left running in their own AWS accounts will result in a bill.

Let us know your feedback! We are always looking for ways to improve our content for this Immersion Day.


This is a 1-day workshop.

## module

### Understanding Data for Machine Learning
SageMaker Introduction , ML Basics,Feature Engineering for Machine Learning - Learn all about the built-in notebook instances with Amazon SageMaker- Feature Engineering and Data Labeling.

### Lab- Notebook Instances and Feature Engineering
Get hands-on experience with SageMaker Console and Jupyter Notebook. Play around code to do feature engineering of sample dataset.

### Understanding Built-in Algorithms
Built-in Machine Learning Algorithms with Amazon SageMaker and Model Evaluation - Amazon SageMaker comes built-in with a number of high-performance algorithms for different use cases. Learn the fundamentals and then dive deep into these algorithms.

### Lab - Train, Tune and Deploy model using SageMaker Built-in Algorithm
Get hands-on experience in one of the most famous in-built ML algorithm Xgboost to build you model. Learn how you can get the best version of your machine learning model using hyperparameter tuning .Amazon SageMaker enables you to quickly and easily deploy your ML models to the most scalable infrastructure. You will learn deployment options and autoscaling for your ML models endpoint. Real time and batch inference techniques.

### Create Model, Prediction and Inference
Train, Tune and Deploy ML Models with Amazon SageMaker - A key aspect of training machine learning models is the ability to tune them to the highest accuracy. you will learn how to train and tune your ML models and deploy them into production. You will also learn real time and batch inference techniques to get prediction from model.

### Bring Your Own Custom Models
With Amazon SageMaker, you have the flexibility to bring in your own model and leverage the capabilities of the service. You will dive deep into how you can bring your own model.

### Lab - Bring your own Model
Wrap-up your own model in docker container and bring to Sagemaker for training and deployment. Script your model with AWS managed container

### SageMaker Model - Debug, Monitor and AutoML
Learn all new feature launched recently which will help you to debug and monitor your ML model. AutoML help to create model for tabular data automatically and SageMaker studio help team to collaborate on model development.

### Lab - Model Debugging , Model Monitoring and AutoML
Learn to debug your model and perform monitoring in production to prevent inference data drift.

### SageMaker Well Architected
Learn how to well architected your ML pipeline with optimization of cost, security, operation, Reliability and performance.




#### What you will learn



#### Format


#### Agenda

| Time  | Title                 | Content |
|-------|-----------------------|---------|
| 09:00 | Introductions         |         |
| 09:05 | Name and Value        |         |
|       | Object                |         |
|       | Break                 |         |
|       | The Myth of Namespace |         |
|       | Module and package    |         |
|       | break                 |         |
|       | Garbage collection    |         |
|       | Speed up your code    |         |
|       | Misc                  |         |

#### Prerequisit

