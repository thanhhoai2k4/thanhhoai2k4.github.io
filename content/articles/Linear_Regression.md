+++
title = '2. Linear Regression'
date = 2024-05-19T06:49:04-07:00
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


![image info](/images/2linear-regression-Head.png "Linear regression")

## Tab of content
- [Tab of content](#tab-of-content)
  - [1. Introduction](#1-introduction)
  - [2. Mathematical modeling](#2-mathematical-modeling)
    - [2.1 Mathematical analysis](#21-mathematical-analysis)
    - [2.2 Loss Function](#22-loss-function)
    - [2.3multiple linear regression](#23multiple-linear-regression)
  - [3. Methods](#3-methods)
    - [3.1 Closed-Form Solution](#31-closed-form-solution)
    - [3.2 Gradient descent](#32-gradient-descent)
  - [4. Numerical studies](#4-numerical-studies)

<!-- headings -->
<a id="Introduction"></a>
### 1. Introduction


Linear regression is a Supervised Learning algorithm. It's purpose is to find relationship between
the independent variable and a dependent variable to predict the outcome of the new feature. Linear regression models are often used to predict the value of a continuous number such as salary, house price, age,...

A traight line is a representation of the linear regression model.

**Example the about figure:**

![alt text](/images/2LinearRegression.png "Title")

  

<a id="equationone" style="text-decoration:none">Linear regression equation: </a>
> $$y = \theta_1 + \theta_2X $$  


This is a simple linear regression model. it's only has one independent variable and one dependent variable. Looking at the piture, it is displayed in two-dimensional space.

>What does it stand for in the above equation?
  - \\(X\\) is the independent variable.
  - \\(y\\) is the output ,which is a depentdent.
  - \\(\theta_1\\) is the gradient of the staight line.
  - \\(\theta_2\\) is the intercept of the traight line.


Our goal is to **find** the **coefficients** such that the line fits the data points as shown in the figure above.
<a id="Mathematical"></a>
### 2. Mathematical modeling


#### 2.1 Mathematical analysis
Suppose, we want to predict the value of a house knowing that it has \\(x_1\\) bedrooms, is \\(x_2\\) km away from the city, and has \\(x_3\\) floors. If we have a collection of approximately 1000 houses like that, what form will our equation take?

[In the system of equations](#equationone). I stipulate that \\(\mathbf{x} = [1, x_1, x_2, x_3]\\) (it is called x bar) is a row vector and \\(\mathbf{w} = [w_0, w_1, w_2, w_3]\\) is a column vector. The number 1 at the beginning is added to simplify and facilitate calculations. In that case, equation can be rewritten as:


<!-- https://www.baeldung.com/cs/latex-bold-math-symbols -->

$$
  \hat{y} \approx y= \mathbf{x} \mathbf{w}
$$

#### 2.2 Loss Function
we will use **Mean Squared Erorr** (MSE) and
We want the forecast error and the actual value to be minimized. In other words, we aim to make the equation below as small as possible:

$$
   L(w) = \frac{1}{2} \sum_{n=1}^{n} (y_i - \hat{y_i})^2 = \frac{1}{2} ( \mathbf{y} - \mathbf{X}\mathbf{w} )^2
$$

In the problem, 1/2 is used for easy computation of the derivative later. [ You can refer to it here](https://datascience.stackexchange.com/questions/53171/why-does-putting-a-1-2-in-front-of-the-squared-error-make-the-math-easier). And we need the square of \\(y_i - \hat{y}\\) to make it positive because loss Function can be negative. Another purpose is to create 2 after differentiating to eliminate with  \\( \frac{1}{2} \\).


#### 2.3multiple linear regression

I stipulate that \\(   \mathbf{X} = [ \mathbf{x_1} , \mathbf{x_2} , \mathbf{x_3} , ..., \mathbf{x_n}] \\) is a matrix where each row represents a data point in the dataset. When we have the cost function in the following form:
$$
  L( \mathbf{w}) = \mathbf{X} \mathbf{w}
$$


and Loss Function can be rewritten as:

$$
  L(w) = \frac{1}{2} (\mathbf{y} - \mathbf{X}\mathbf{w})^2 
$$


<a id="Methods  "></a>
### 3. Methods

#### 3.1 Closed-Form Solution

we use closed from solution. We take the loss function, find its derivative, and set it equal to 0. Solve the equation:


$$
  \frac{\nabla L(\mathbf{w})}{\nabla \mathbf{w}} =  \frac{1}{2} (\mathbf{y} - \mathbf{X}\mathbf{w})^2_{\mathbf{w}}
$$


$$
  \frac{\nabla L(\mathbf{w})}{\nabla \mathbf{w}} = 0 
$$


$$
  \implies - \mathbf{X^T} ( \mathbf{y} - \mathbf{X} \mathbf{w} ) = 0
$$


$$t
   -\mathbf{X} ( \mathbf{y} - \mathbf{X} \mathbf{w} ) = 0
$$


$$
  \mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X^T} \mathbf{y}
$$

So, \\( \mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X^T} \mathbf{y} \\)

to refer to [Here](/images/Derivative_MSE_LINEAR.png )


#### 3.2 Gradient descent

optimizer is used for fine-tunes \\( \mathbf{w} \\) with equation:

$$
  \mathbf{w_{t+1}} = \mathbf{w_{t}} - \alpha \nabla_{t} L(\mathbf{w})
$$

in the equation, \\( \alpha \\) is a learning rate. Common, learning rate set it 0.001

Gradien descent have variant. You can refer to it here for [more information](https://deepchecks.com/glossary/learning-rate-in-machine-learning/#:~:text=The%20learning%20rate%2C%20denoted%20by,network%20concerning%20the%20loss%20gradient%3E.)


### 4. Numerical studies

Suppose that the equation has form \\( y= 3x + 5 \\). 

add Noice:


....