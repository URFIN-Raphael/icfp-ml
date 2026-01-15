# Machine Learning - [Master ICFP](https://www.phys.ens.fr/en/formations/m2-icfp)

**Prerequisites**:
- Proficiency in Python: please use the tutorial [here](https://cs231n.github.io/python-numpy-tutorial/) for those who aren't as familiar with Python
- Basic Calculus, Linear Algebra
- Basic Probability and Statistics

**Logistic**:
- course: [Marc Lelarge](https://www.di.ens.fr/~lelarge/)
- practicals: Leonardo Defilippis and [Raphaël Urfin](https://urfin-raphael.github.io/)
- Thursday 14h-17h30
- Room 14-24 305 from 15/01/2026 to 19/02/2026 except 12/02 54-55 205
- To validate the course, you need to register on [Moodle](https://moodle.psl.eu/course/view.php?id=35560)

## Tentative Schedule:
## 1. Fundamentals of predictions and supervised learning 

### Fundamentals of predictions
- Minimizing errors
- Modeling knowledge
- Prediction via optimization
- Types of errors and successes
- Properties of ROC curves

### Ref
- [Fundamentals of prediction](https://mlstory.org/prediction.html) from Patterns, Predictions, and Actions (A story about machine learning) by Moritz Hardt and Benjamin Recht

### practicals
- Exact ROC curves for Gaussian mixtures: https://github.com/mlelarge/icfp-ml/blob/main/Exact_ROC_GM.ipynb

### supervised learning
- Sample versus Population
- A first learning algorithm: the perceptron
- Connection to empirical risk minimization
- Formal guarantees for the perceptron

### Ref: 
- [Supervised learning](https://mlstory.org/supervised.html)  from Patterns, Predictions, and Actions (A story about machine learning) by Moritz Hardt and Benjamin Recht

### practicals
- Naive Bayes and logistic regression: https://github.com/mlelarge/icfp-ml/blob/main/01_NaivesBayes_Logistic_empty.ipynb

## 2. Pytorch basics and autodiff 

[Module 2a - Pytorch tensors](https://dataflowr.github.io/website/modules/2a-pytorch-tensors/)

[Module 2b - Automatic differentiation](https://dataflowr.github.io/website/modules/2b-automatic-differentiation/)

## 3. Optimization for machine learning 

- gradient descent
- SGD
- over-parameterized models:https://hackmd.io/@mlelarge/S1y5bEAhj

### Ref: 
- In Chapter 5 (Sections 5.2.1 and 5.4) of [Learning Theory from First Principles](https://www.di.ens.fr/~fbach/ltfp_book.pdf) by Francis Bach


### practicals
- [Module 5 - Stacking layers](https://dataflowr.github.io/website/modules/5-stacking-layers/)

- Heavy Ball Method: https://github.com/mlelarge/icfp-ml/blob/main/HeavyBall_empty.ipynb

## 4. Kernels 

- Local averaging methods
    - partitions estimators
    - k-nearest neighbors
    - kernel smoothing
- Positive-definite kernel methods
    - representer theorem
    - kernel trick

### Ref: 
- Chapters 6 and 7 of [Learning Theory from First Principles](https://www.di.ens.fr/~fbach/ltfp_book.pdf) by Francis Bach

### practicals
- Kernel with [random Fourier features](https://github.com/mlelarge/icfp-ml/blob/main/03_kernel_random_fourier_empty.ipynb)

## 5. Unsupervised Learning 

- K-means clustering
- Mixtures of Gaussian
- Expectation-Maximization for GMM

### Ref:
- [Expectation-Maximization for the Gaussian Mixture Model](https://perso.telecom-paristech.fr/bonald/documents/gmm.pdf) by Thomas Bonald
- [The Expectation Maximization Algorithm - A short tutorial](https://www.seanborman.com/publications/EM_algorithm.pdf) by Sean Borman

### practicals
- SVD
- [Eigenfaces](https://github.com/mlelarge/icfp-ml/blob/main/02_SVD_Eigenfaces_empty.ipynb)

## 6. Bayesian and Variational Inference 

- Gaussian
- Linear regression
- Logistic regression
- Laplace method
- Variational inference

### Ref:
- Chapters 2.3 - 3.3 - 4.4 - 4.5 - 10 [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) by Christopher Bishop
 
## 7. Deep generative models: flows and diffusions 

- [Normalizing flows](https://dataflowr.github.io/website/modules/9c-flows/)
- [Denoising Diffusion Probabilistic Models](https://dataflowr.github.io/website/modules/18a-diffusion/)
