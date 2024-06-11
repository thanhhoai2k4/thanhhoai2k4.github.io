+++
title = '3. K-nearest neighbors'
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

![image info](/images/knn-1.png "Linear regression")

## Tab of content
- [Tab of content](#tab-of-content)
  - [1. Introduction](#1-introduction)


<a id="Introduction"></a>
### 1. Introduction

the **K-Nearest Neighbors** (KNN) is a supervised machine learning method employed to tackle classification and regression problems. When traning, it doesn't learn anything from the data but mechanically memorizes it. Therefore, this algorithm is called as **Lazy Learning**. With KNN in classification problems, the label of a new data point is innferred from its K nearest data points.