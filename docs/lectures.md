# Lectures

Complete lecture materials with slides, review questions, theory connections, and PhD-level insights — all in PDF format.

---

## Course Administration

### Assessment Breakdown

| Component | Weight | Details |
|-----------|--------|---------|
| **Quizzes** | **40%** | 4-5 quizzes throughout the course |
| **Midterm Examination** | **20%** | Covers Part I: Mathematical Foundations |
| **Final Examination** | **30%** | Comprehensive exam covering all topics |
| **Reading & Review Papers** | **10%** | 4-5 research papers to review |
| **TOTAL** | **100%** | - |

### Course Level & Prerequisites

**Target Audience:** Undergraduate/Graduate Level

**Flexible Prerequisites:**

- Basic linear algebra and vector calculus recommended
- Course structure allows for adjustment based on student background
- Mathematical concepts presented with varying levels of rigor

---

## Lecture Structure

Every lecture follows a **consistent four-part approach** designed to build deep understanding:

| Part | What It Covers |
|------|----------------|
| **Concepts & Explanation** | Key definitions, intuitive explanations, "Think of it as..." plain-English descriptions |
| **Mathematical Examples & Tutorials** | Step-by-step worked examples with detailed solutions |
| **Review & Practice** | 10 structured review questions with hints, common mistakes, and concepts-at-a-glance tables |
| **Advanced Perspectives** | 5 Theory Connection slides (AI/ML applications) + 5 PhD View slides (research-level topics) |

---

## Part I: Mathematical Foundations

---

### Lecture 0: Introduction & Course Overview

Course organization, assessment structure, textbooks, teaching philosophy, notation conventions, and what to expect.

- [Download Lecture 0 (PDF)](pdfs/Mathematics_for_Machine_Learning.pdf)

---

### Lecture 2: Linear Algebra

<div class="admonition note" markdown="1">
<p class="admonition-title">Foundation of Everything in AI</p>

Linear algebra is the computational engine behind all of machine learning. Every dataset is a matrix, every data point is a vector, and every neural network layer computes a linear transformation.
</div>

**Topics covered (8 sections):**

1. Systems of Linear Equations
2. Matrices
3. Solving Systems of Linear Equations
4. Vector Spaces
5. Linear Independence
6. Basis and Rank
7. Linear Mappings
8. Affine Spaces

**What's inside the slides:**

- Definitions with plain-English "Think of it as..." intuitions
- Full Math Tutorial section with 15 worked examples
- 10 Review Questions with hints (systems, matrices, inverse/transpose, Gaussian elimination, vector spaces, linear independence, basis/dimension, rank, linear mappings, affine spaces)
- Common Mistakes to Avoid (5 key pitfalls)
- Concepts at a Glance comparison table
- Key Takeaways summary
- 5 Theory Connection slides (ML, Computer Vision, Deep Learning, Optimization, Data Science)
- 5 PhD View slides (Functional Analysis, Matrix Decompositions, Tensor Algebra, Numerical LA, Spectral Theory)

**Materials:**

- [Download Lecture 2: Linear Algebra (PDF)](pdfs/Lecture_2_Linear_Algebra.pdf)
- [Tutorial: Linear Algebra](tutorials/Tutorial_02_Linear_Algebra.md)
- [Notebook: NumPy Basics](notebooks/01_Python_Libraries_Basics.ipynb)

---

### Lecture 3: Analytic Geometry

<div class="admonition note" markdown="1">
<p class="admonition-title">The Geometry Behind Machine Learning</p>

Analytic geometry gives us the tools to measure similarity, distance, and angles in high-dimensional spaces — the foundation of recommendation systems, NLP embeddings, and PCA.
</div>

**Topics covered (9 sections):**

1. Norms
2. Inner Products
3. Lengths and Distances
4. Angles and Orthogonality
5. Orthonormal Basis
6. Orthogonal Complement
7. Inner Product of Functions
8. Orthogonal Projections
9. Rotations

**What's inside the slides:**

- Complete concept chain: Inner Product → Norm → Distance → Angle → Orthogonality → Projection
- Projection formulas for 1D lines and general subspaces
- Gram-Schmidt orthogonalization with worked examples
- 10 Review Questions with hints (norms, inner products, Cauchy-Schwarz, angles, orthogonal matrices, Gram-Schmidt, orthogonal complement, projections onto lines, general projections, rotations)
- Common Mistakes to Avoid (5 key pitfalls)
- Concepts at a Glance comparison table
- 5 Theory Connection slides (ML, NLP, Deep Learning, Computer Vision, Data Science)
- 5 PhD View slides (Hilbert Spaces, RKHS, Compressed Sensing, Riemannian Geometry, Random Projections)

**Materials:**

- [Download Lecture 3: Analytic Geometry (PDF)](pdfs/Lecture_3_Analytic_Geometry.pdf)
- [Tutorial: Analytic Geometry](tutorials/Tutorial_03_Analytic_Geometry.md)
- [Notebook: Analytic Geometry in Python](notebooks/02_Analytic_Geometry.ipynb)

---

### Lecture 4: Matrix Decomposition

<div class="admonition note" markdown="1">
<p class="admonition-title">Revealing Hidden Structure in Data</p>

Matrix decompositions like eigendecomposition and SVD are the workhorses of data science — they power PCA, recommender systems, image compression, and Google's PageRank.
</div>

**Topics covered (6 sections):**

1. Determinants and Trace
2. Eigenvalues and Eigenvectors
3. Cholesky Decomposition
4. Eigendecomposition and Diagonalization
5. Singular Value Decomposition (SVD)
6. Matrix Approximation

**What's inside the slides:**

- Eigenvalue computation with characteristic polynomials
- SVD: geometric interpretation and step-by-step examples
- Matrix approximation via truncated SVD
- 10 Review Questions with hints
- Common Mistakes to Avoid
- 5 Theory Connection slides (ML applications of decompositions)
- 5 PhD View slides (advanced decomposition theory)

**Materials:**

- [Download Lecture 4: Matrix Decomposition (PDF)](pdfs/Lecture_4_Matrix_Decomposition.pdf)
- [Tutorial: Matrix Decomposition](tutorials/Tutorial_04_Matrix_Decomposition.md)
- [Notebook: Matrix Decomposition in Python](notebooks/03_Matrix_Decomposition.ipynb)

---

### Lecture 5: Vector Calculus

<div class="admonition note" markdown="1">
<p class="admonition-title">The Mathematics of Learning</p>

Vector calculus provides the gradient and chain rule — without these, there is no backpropagation, no gradient descent, and no training of neural networks.
</div>

**Topics covered (6 sections):**

1. Differentiation of Univariate Functions
2. Partial Differentiation and Gradients
3. Gradients of Vector-Valued Functions
4. Gradients of Matrices
5. Useful Identities for Computing Gradients
6. Backpropagation and Automatic Differentiation

**What's inside the slides:**

- From single-variable derivatives to Jacobians and gradients
- Chain rule for vector and matrix functions
- Backpropagation derivation with computation graphs
- 10 Review Questions with hints
- Common Mistakes to Avoid
- 5 Theory Connection slides (optimization, neural networks, physics)
- 5 PhD View slides (differential geometry, automatic differentiation theory)

**Materials:**

- [Download Lecture 5: Vector Calculus (PDF)](pdfs/Lecture_5_Vector_Calculus.pdf)
- [Tutorial: Vector Calculus](tutorials/Tutorial_05_Vector_Calculus.md)
- [Notebook: Vector Calculus in Python](notebooks/04_Vector_Calculus.ipynb)

---

### Lecture 6: Probability & Distributions

<div class="admonition note" markdown="1">
<p class="admonition-title">Reasoning Under Uncertainty</p>

Probability theory is the language of uncertainty — it underpins Bayesian inference, generative models, statistical testing, and every probabilistic ML algorithm.
</div>

**Topics covered (8 sections):**

1. Construction of a Probability Space
2. Discrete and Continuous Probabilities
3. Sum Rule, Product Rule, and Bayes' Theorem
4. Summary Statistics and Independence
5. Gaussian Distribution
6. Conjugacy and Exponential Family
7. Change of Variables / Inverse Transform
8. Probability in Machine Learning

**What's inside the slides:**

- From sample spaces to random variables to distributions
- Discrete (Bernoulli, Binomial, Geometric) and Continuous (Uniform, Exponential, Gaussian) distributions
- Bayes' Theorem with real-world examples
- Marginal and conditional distributions
- Covariance, correlation, and independence
- 10 Review Questions with hints
- Common Mistakes to Avoid
- 5 Theory Connection slides (Bayesian ML, generative models, information theory)
- 5 PhD View slides (measure theory, stochastic processes, information geometry)

**Materials:**

- [Download Lecture 6: Probability & Distributions (PDF)](pdfs/Lecture_6_Probability_Distributions.pdf)
- [Tutorial: Probability and Distributions](tutorials/Tutorial_06_Probability_Distributions.md)
- [Notebook: Probability and Distributions in Python](notebooks/05_Probability_Distributions.ipynb)

---

## Part II: Machine Learning Applications

---

### Application 1: When Models Meet Data

**Description:** Introduction to the practical aspects of applying mathematical models to real-world datasets and the challenges that arise.

**What you'll learn:**

- Data preprocessing and feature engineering
- Model selection and evaluation
- Overfitting and regularization
- Bias-variance tradeoff

**Materials:**

- [Lecture Slides (Complete PDF)](pdfs/Mathematics_for_Machine_Learning.pdf)

---

### Application 2: Dimensionality Reduction (PCA)

**Description:** Using Principal Component Analysis to reduce data complexity while preserving essential information for efficient learning.

**What you'll learn:**

- PCA theory: eigenvalues meet data variance
- Step-by-step PCA computation
- Variance explained and choosing dimensions
- Eigenfaces and image compression

**Materials:**

- [Lecture Slides (Complete PDF)](pdfs/Mathematics_for_Machine_Learning.pdf)

---

### Application 3: Density Estimation (GMM)

**Description:** Probabilistic approaches to understanding data distributions and clustering using Gaussian Mixture Models.

**What you'll learn:**

- Probability density estimation
- Gaussian Mixture Models
- Expectation-Maximization (EM) algorithm
- Clustering applications

**Materials:**

- [Lecture Slides (Complete PDF)](pdfs/Mathematics_for_Machine_Learning.pdf)

---

### Application 4: Classification (SVM)

**Description:** Geometric and optimization-based methods for supervised learning and decision boundary determination using Support Vector Machines.

**What you'll learn:**

- Support Vector Machines (SVM)
- Kernel methods and the kernel trick
- Margin maximization
- Multi-class classification

**Materials:**

- [Lecture Slides (Complete PDF)](pdfs/Mathematics_for_Machine_Learning.pdf)

---

## Quick Reference: All Lecture Downloads

| Lecture | Topic | Download |
|---------|-------|----------|
| Lecture 0 | Introduction & Course Overview | [PDF](pdfs/Mathematics_for_Machine_Learning.pdf) |
| Lecture 2 | Linear Algebra | [PDF](pdfs/Lecture_2_Linear_Algebra.pdf) |
| Lecture 3 | Analytic Geometry | [PDF](pdfs/Lecture_3_Analytic_Geometry.pdf) |
| Lecture 4 | Matrix Decomposition | [PDF](pdfs/Lecture_4_Matrix_Decomposition.pdf) |
| Lecture 5 | Vector Calculus | [PDF](pdfs/Lecture_5_Vector_Calculus.pdf) |
| Lecture 6 | Probability & Distributions | [PDF](pdfs/Lecture_6_Probability_Distributions.pdf) |
| Applications | ML Applications (all 4) | [PDF](pdfs/Mathematics_for_Machine_Learning.pdf) |

---

## Primary Textbooks

### Mathematics for Machine Learning
**Authors:** Deisenroth, Faisal, and Ong
**Publisher:** Cambridge University Press
**Website:** [mml-book.github.io](https://mml-book.github.io/)

### Convex Optimization
**Authors:** Boyd and Vandenberghe
**Publisher:** Cambridge University Press
**Website:** [stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/)

### Introduction to Probability
**Authors:** Bertsekas and Tsitsiklis
**Publisher:** Athena Scientific (2nd Ed.)

---

## Learning Tips

**Before Each Lecture:**

- Download and review the lecture slides
- Complete prerequisite readings
- Review previous lecture concepts

**During Lectures:**

- Follow the four-part structure
- Take notes during explanations
- Work through examples actively
- Try coding exercises immediately

**After Lectures:**

- Work through the 10 Review Questions (use hints if stuck)
- Study the Common Mistakes to Avoid
- Explore Theory Connection slides for AI/ML context
- Review PhD View slides for deeper understanding
- Complete Python notebooks

---

<div class="signature" markdown="1">

*Lectures prepared by Mohammed Alnemari*
*Mathematics of AI &bull; Spring 2026*

</div>

---

<div class="last-updated" markdown="1">
**Last Updated:** February 8, 2026
</div>
