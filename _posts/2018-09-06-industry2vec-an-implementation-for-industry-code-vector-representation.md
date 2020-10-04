---
classes: wide
categories:
  - Deep Learning
tags:
  - Embedding
  - LSTM
  - Siamese Network
---

### TL;DR or Executive Summary
Industry code organizes companies into industrial groupings based on similar production or behavior. It is an important factor in banking and financial institutions data.
However, since industry code is a kind of category code, it is not ideal for Machine Learning numerical computation. In this blog, we use a Siamese network architecture to generate a vector numerical representation of the industry code.
This vector representations has a good interpretability and can be used in the numerical computation scenarios such as computing company similarities. We implement it in Keras and the open source repository is here .