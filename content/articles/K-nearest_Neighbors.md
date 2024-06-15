+++
title = '3. K-Nearest Neighbors'
date = 2024-06-10
+++


<!-- this is a code block for using mathematical symbols -->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<!-- This part includes the Javascript file -->
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

<!-- this part configures it -->
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['\\(','\\)']],
    displayMath: [['$$','$$'], ['\[','\]']],
    processEscapes: true,
    processEnvironments: true,
    skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
    TeX: { equationNumbers: { autoNumber: "AMS" },
         extensions: ["AMSmath.js", "AMSsymbols.js"] }
  }
});
</script>

![image info](/images/knn-1.png "KNN 1")

## Tab of content
- [Tab of content](#tab-of-content)
  - [1. Introduction](#1-introduction)
  - [2. Mathematical modeling](#2-mathematical-modeling)
  - [3. Code example on iris flower dataset](#Code_example)


<a id="Introduction"></a>
### 1. Introduction

The **K-Nearest Neighbors** (**KNN**) is a supervised machine learning method **employed** to tackle classification and regression problems. When training, it doesn’t learn anything from the data but mechanically memorizes it. Therefore, this algorithm is called **Lazy Learning**. With KNN in classification problems, the label of a new data point is inferred from its K nearest data points. With KNN in regression problems, the values of a new data point is inferred from its a data point nearst with (**k** = 1). In another case, the new value is predicted by the average value of the k nearest neighbors (point).


![image info](/images/KNN_solv.png "KNN 2")

Looking at the piture, we can predict it! so, what color is the data? **Red** or **Blue** ? 

Through this algorithm, you will know which one it belongs to.


<a id="2-mathematical-modeling"></a>
### 2. Mathematical modeling

K-nearest Neighbors algorithm is Lazy learning. it has any loss Function. KNN mainly calculates the distance. There are two things to pay attention to at this point. How is the distance defined? How to calculate distances effectively?

- Each data point is represented as a vector. The distance between two points is the distance between two vectors. We have many ways to calculate distance. Among them, the [L2 norm](https://s.net.vn/fTI5) is mainly used.
- When the amount of data is large, calculating the distance from a data point will be very time-consuming. If there is no effective calculation method, time will be wasted.


Distance from one point to each point in dataset:


$$
  L_2 = || \mathbf{z}- \mathbf{x_i}||_2
$$

In the quation:
- **z** is one data point that needs to be predicted
- \\( \mathbf{x_i} \\) is one data point in the data set.


During the calculation process, the main purpose is to serve for sorting, so there is no need to calculate the square root. Therefore, the equation will be written as:

$$
  L_2 = || \mathbf{z}- \mathbf{x_i}||_2^2
$$

Distributive property:

$$
  L_2 = || \mathbf{z}- \mathbf{x_i}||_2^2 = ( \mathbf{z} - \mathbf{x_i})^T ( \mathbf{z} - \mathbf{x_i}) = ||\mathbf{z}||_2^2 + ||\mathbf{x_i}||_2^2 + 2\mathbf{x_i}^T\mathbf{z}
$$

We can ignore \\( ||\mathbf{z}||_2^2  \\) because the calculation of the distance from \\( \mathbf{z} \\) to \\( \mathbf{x_i} \\) does not need the distance of \\( \mathbf{z} \\). Furthermore, \\( \mathbf{x_i} \\) can be calculated and stored in memory. Then, the main task is computational \\( 2\mathbf{x_i}^T\mathbf{z} \\) .



<a id="Code_example"></a>

### 3. Code example
I write code KNN with K = 1. [Code KNN in my github](https://github.com/thanhhoai2k4/Machine_Learning_Basic/tree/main/K-Nearest_Neighbors)



When i excuted [knn.py](https://github.com/thanhhoai2k4/Machine_Learning_Basic/blob/main/K-Nearest_Neighbors/Knn.py) with split = 0.2. and my result is returned with 100 % percent exact rate. Moreover, when i set test_size = 0.5, percent is  a 94.6666 %.
So i am really lucky <3.