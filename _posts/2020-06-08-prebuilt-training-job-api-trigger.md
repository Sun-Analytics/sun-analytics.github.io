---
title: Trigger Pre-built Framework Training Job via Amazon SageMaker API
classes: wide
categories:
  - AWS
tags:
  - SageMaker
  - Training
---

### TL;DR
The SageMaker training job with customized training script in frameworks such as TensorFlow/PyTorch/scikit-learn can also be triggered by pure SageMaker API, by configuring the request body fields:
- **HyperParameters.sagemaker_submit_directory**: the S3 location of the uploaded source.tar.gz file, which tars the training script.
- **HyperParameters.sagemaker_program**: the name of the entry point file
- **AlgorithmSpecification.TrainingImage**: the Amazon ECR registry path of the pre-built framework container images. You can find the images URL [here](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#general-framework-containers){:target="_blank"} or [here](https://github.com/aws/sagemaker-scikit-learn-container){:target="_blank"}.

### Reason for This Blog
Amazon SageMaker Python SDK is a great package for SageMaker practices. Still, in some scenarios it is required to trigger pre-built framework (TensorFlow, Pytorch, scikit-learn, etc) training job via SageMaker API directly. However, there does not seem to have any explicit document/tutorial to describe the solution. So, I write this short article, and hope it can help you.

### Running Pre-built Framework Training Job with Amazon SageMaker Python SDK
[Amazon SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/index.html){:target="_blank"} is an open source library for training and deploying machine-learned models on Amazon SageMaker. There are bunch of examples for TensorFlow, PyTorch, scikit-learn and more frameworks in this open source repository [amazon-sagemaker-examples](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/sagemaker-python-sdk){:target="_blank"}. In short, we can create an Estimator with the customized script and fit the estimator as the code piece below. For the parameter of [PyTorch estimator](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html#sagemaker.pytorch.estimator.PyTorch){:target="_blank"}, **entry_point** indicates the training script, **framework_version** and **py_verison** decide the pre-built container image.

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(entry_point='mnist.py',
                    role=role,
                    framework_version='1.4.0',
                    py_version='py3',
                    train_instance_count=2,
                    train_instance_type='ml.c4.xlarge',
                    hyperparameters={
                        'epochs': 6,
                        'backend': 'gloo'
                    })

estimator.fit({'training': inputs})
```
Example code of starting a SageMaker PyTorch Training job by SageMaker Python SDK. The code is copied from this [example](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/pytorch_mnist/pytorch_mnist.ipynb){:target="_blank"} in amaozn-sagemaker-example repository

### Scenarios of Using SageMaker API Directly
Although SageMaker Python SDK is very convienient to run managed training job for a variety of machine learning frameworks, there are still some scenarios that we need to trigger SageMaker Training job directly via [Amazon SageMaker CreateTrainingJob API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html){:target="_blank"}, such as:
- Machine Learning engineers or software developers use other languages instead of Python.
- The machine learning operational pipeline is constructed by AWS Step Functions. The [**Step Functions SageMaker connection**](https://docs.aws.amazon.com/step-functions/latest/dg/connect-sagemaker.html){:target="_blank"} uses SageMaker API interface.

### Use SageMaker API to Trigger Pre-build Framework Training Job with Training Scripts
People may think it is not supported by SageMaker API to trigger TensorFlow/PyTorch/… training job with customized training script, because the SageMaker API seems have no place to setup training script location at the first glance. The good news is we can!

First, **we need to tar the training script as source.tar.gzand upload to a S3 location**, e.g., _s3://bucket/prefix/source.tar.gz_. This step can be done as a step of the step functions, or in the CI/CD build stage, depends on how we operate the ML pipeline.

Then, **we need to set up these fields** in SageMaker CreateTrainingJob API request body or the state definition of Step Functions SageMaker connector.
- **HyperParameters.sagemaker_submit_directory**: the S3 location of the uploaded source.tar.gz file, e.g., s3://bucket/prefix/source.tar.gz
- **HyperParameters.sagemaker_program**: the name of the entry point file
- **AlgorithmSpecification.TrainingImage**: the Amazon ECR registry path of the pre-built framework container images. **You can find [the images URL](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#general-framework-containers){:target="_blank"} here**.

Below is an example to trigger the same training job by using [Python boto3 SDK](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_training_job){:target="_blank"}. For more information about HyperParameters, you can refer to this [code](https://github.com/aws/sagemaker-training-toolkit/blob/master/src/sagemaker_training/params.py){:target="_blank"} in [sagemaker-training-toolkit](https://github.com/aws/sagemaker-training-toolkit){:target="_blank"} package.

``` python
...
import boto3
src_path = 's3://bucket/prefix/source.tar.gz'
def trigger_train():
    training_job_name = '<YOUR-TRAINING-JOB-NAME>'
    sm = boto3.client('sagemaker')
    resp = sm.create_training_job(
            TrainingJobName = training_job_name, 
            AlgorithmSpecification={
                'TrainingImage': '763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.4.0-gpu-py36-cu101-ubuntu16.04',
                ...
            }, 
            RoleArn=role_arn,
            HyperParameters={
                'sagemaker_program' : "mnist.py",
                'sagemaker_submit_directory': src_path,
                'sagemaker_region': "<your-aws-region>",                
            },
            InputDataConfig=[<Input-data-setup>], 
            OutputDataConfig={<output-data-setup>},
            ResourceConfig={<resource-setup>},         
            ...
    )
...
```