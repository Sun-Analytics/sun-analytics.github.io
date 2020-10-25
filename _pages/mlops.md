---
layout: training-page
permalink: /training/mlops/
hidden: true
toc: true
toc_sticky: true
header:
  overlay_color: "#5e616c"
  overlay_image: /assets/images/training/mlops/background-page.jpg
  actions:
    - label: "<i class='far fa-calendar-check'></i>&nbsp; &nbsp; Book a training now"
      url: "mailto:info@sun-analytics.nl?subject=[Training]%3AMLOps%20training"
title: "Machine Learning (ML) Ops on AWS"
excerpt: >
  This 1-day training provides you with hands-on experience to deploy a ML pipeline in an automation, infrastructure-as-code and monitorable way on AWS via a CI/CD pipeline.<br />

training-setting:
  hours: 8
  sessions: 7
  labs: 9
  type: Virtual/Classroom
  seats: 6-12
---

## Content of This Training

<details>
  <summary><b>An Overview of MLOps and Why MLOps on AWS</b></summary>
  <p>
    <li><strong>Content:&ensp;</strong> <i>We first introduce the basic ideas of MLOps, and the advantage of MLOps on AWS ecosystem.</i></li>
    <li><strong>Duration:</strong> <i>45 mintues</i></li>
    <li><strong>Type:&emsp;&emsp;</strong><i>presentation</i></li>
  </p>
</details>

<details>
  <summary><b>Build an automation MLOps pipeline</b></summary>
  <p>
    <li><strong>Content:&ensp;</strong> <i>We visit each component of ML pipeline and SageMaker quickly, then introduce AWS Step Function and learn how step function orchastrate the ML pipeline easily. In the lab, we build a end-to-end ML pipeline orchestrate by step function.</i></li>
    <li><strong>Duration:</strong> <i>90 mintues</i></li>
    <li><strong>Type:&emsp;&emsp;</strong><i>presentation and lab</i></li>
  </p>  
</details>

<details>
  <summary><b>Operate Your Model</b></summary>
  <p>
    <li><strong>Content:&ensp;</strong> <i>We introduce Model Debugging and Model Monitoring on Amazon SageMaker. We also discuss the CloudWatch Dashboard and Alarm. Then we practice these topics in the lab.</i></li>
    <li><strong>Duration:</strong> <i>60 mintues</i></li>
    <li><strong>Type:&emsp;&emsp;</strong><i>presentation and lab</i></li>
  </p>  
</details>

<details>
  <summary><b>Infrastructure-as-Code: AWS CloudFormation</b></summary>
  <p>
    <li><strong>Content:&ensp;</strong> <i>We discuss the advantage of infrastructure as code, then introduce the basic concept of AWS CloudFormation. In the lab, we will build a ML pipeline via CloudFormation</i></li>
    <li><strong>Duration:</strong> 1.5 hour</li>
    <li><strong>Type:&emsp;&emsp;</strong>presentation and lab</li>
  </p>  
</details>

<details>
  <summary><b>Infrastructure-as-Code: AWS Cloud Development Kit (Optional)</b></summary>
  <p>
    <li><strong>Content:&ensp;</strong> <i>AWS CDK can define your cloud application resources using your familiar programming languages. We introduce the basic feature of CDK, and practice a simple infrastructure deployment by using AWS CDK.</i></li>
    <li><strong>Duration:</strong> 1 hour</li>
    <li><strong>Type:&emsp;&emsp;</strong>presentation and lab</li>
  </p>  
</details>

<details>
  <summary><b>CI/CD in MLOps</b></summary>
  <p>
    <li><strong>Content:&ensp;</strong> <i>This session will introduce the CI/CD pipeline, and AWS services such as CodeCommit, CodeBuild, CodeDeploy, CodePipeline. Then We deploy our ML pipeline via CI/CD in the lab.</i></li>
    <li><strong>Duration:</strong> <i>2 hours</i></li>
    <li><strong>Type:&emsp;&emsp;</strong> <i>presentation and lab</i></li>
  </p>  
</details>

## What is MLOps
MLOps (a compound of "machine learning" and "operations") is a practice for collaboration and communication between data scientists and operations professionals to help manage production ML lifecycle. Similar to the DevOps or DataOps approaches, MLOps looks to increase automation and improve the quality of production ML while also focusing on business and regulatory requirements.

## Who Needs This Training
- Someone who wants to build or promote a product-ready ML/AI solution on AWS in your organization.
- Participants needs to know basic AWS knowledge of Amazon SageMaker, S3, EC2, IAM, etc. 
- It is ideal but not required that participants have some software developer background with Python scripting experience.

## What This Training Is
- It introduces the basic concept and methods of MLOps on AWS, also guides you to embrace the culture of MLOps
- It provides you with hands-on experience to build an automated, monitorable, Infrastracture-as-Code ML product in a CI/CD pipeline
- This training will empower you as an AWS champion, and you can bring the value of MLOps on AWS back to your organization.

## What This Trianing Is Not
- This training does not bring you to a professional level immediately. The best way to achieve it is learning by doing - applying what you learn from this training on your daily work and keep on practicing.
- This training does not focus on the fundamental knowledge of Amazon SageMaker. If you are interested with learning SageMaker, please check our **[Amazon SageMaker Training](/training/sagemaker/)**.
- This training does not cover the topic of some open source stack such ash Apache Airflow, Kubeflow and MLflow.

## Process

### Step 1: Alignment
- Align the expectation of this training
- Confirm the training day, the number of participants, the conference meeting setup, etc

### Step 2: Preparation

**Setup Account**
- You should use a AWS account that is not running production systems.
- Any accounts should be created a minimum of three days ahead of time. It takes time for new accounts to be completely ready, payment methods to be confirmed, and limits to be set.
- If participants will share an account between multiple people:
  - Ensure they create IAM accounts for each user that will take labs.
  - Adjust account limits to support the number of students doing labs. Note this process can take up to a week, depending on the specific limits.
  - Check limits for EC2 instances, VPCs (# of VPCs, # of IGWs, # of security groups, EIPs), ELBs, etc. The limits that need to be adjusted depends on the labs that will be delivered.
  - More information about limits can be found on the [AWS Service Limits page](https://console.aws.amazon.com/servicequotas/home){:target="_blank"}.

**Software**
- Use up to date software: Ensure participants are using a recent version of Edge, Chrome, Firefox, or Safari.
- Participants using Windows laptops should have a SSH client (like PuTTY) installed.

### Step 3: Training Day
- Enjoy it :)

### Step 4: Follow-up
- Post-event clean-up: help participants clean up any resources still running. Ensure that participants terminate running resources such as SageMaker deployment endpoints or instances.
- Let us know your feedback! We are always looking for ways to improve our content for this training.
