---
classes: wide
---

### TL;DR
Comparing very large feature vectors and picking the best matches, in practice often results in performing a sparse matrix multiplication followed by selecting the top-n multiplication results. In this blog, we implement a customized Cython function for this purpose. When comparing our Cythonic approach to doing the same with SciPy and NumPy functions, our approach improves the speed by about 40% and reduces memory consumption. The GitHub code of our approach is available here.

### Introduction
ING Wholesale Banking has huge amounts of data about many companies, but because the data comes from different source systems, inside and outside the bank, there is no single identifier that can be used to easily connect the data sets. Therefore, we have to connect the data sets based on the names of the companies in the different sets, which are often written in different ways.