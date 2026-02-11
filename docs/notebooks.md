# Python Notebooks

Jupyter notebooks for hands-on implementation and computational exercises.

---

## Getting Started

**Notebook 0: Python Libraries Basics**
Introduction to NumPy, SciPy, and Pandas — Essential libraries for this course.

[Open in GitHub](notebooks/01_Python_Libraries_Basics.ipynb) | [Open in Google Colab](https://colab.research.google.com/github/alnemari-m/mathai-website/blob/main/docs/notebooks/01_Python_Libraries_Basics.ipynb) | [Download](notebooks/01_Python_Libraries_Basics.ipynb)

---

## Lecture Notebooks

### Notebook 2: Analytic Geometry
Norms, inner products, projections, Gram-Schmidt, and rotations — all implemented in Python with visualizations.

[Open in GitHub](notebooks/02_Analytic_Geometry.ipynb) | [Open in Google Colab](https://colab.research.google.com/github/alnemari-m/mathai-website/blob/main/docs/notebooks/02_Analytic_Geometry.ipynb) | [Download](notebooks/02_Analytic_Geometry.ipynb)

**What's inside:**

- Compute and visualize $\ell_1$, $\ell_2$, $\ell_\infty$ norms and unit balls
- Inner products, cosine similarity, and angle computation
- Projection onto lines and general subspaces with 2D visualization
- Gram-Schmidt implementation from scratch (compare with `scipy.linalg.qr`)
- 2D rotation matrices and vector transformations

---

### Notebook 3: Matrix Decomposition
Eigenvalues, SVD, and matrix approximation — the computational workhorses of data science.

[Open in GitHub](notebooks/03_Matrix_Decomposition.ipynb) | [Open in Google Colab](https://colab.research.google.com/github/alnemari-m/mathai-website/blob/main/docs/notebooks/03_Matrix_Decomposition.ipynb) | [Download](notebooks/03_Matrix_Decomposition.ipynb)

**What's inside:**

- Determinants, trace, and property verification
- Eigenvalue computation and eigenvector visualization
- Cholesky decomposition for positive definite matrices
- Eigendecomposition: $A = PDP^{-1}$ reconstruction
- SVD: $A = U\Sigma V^T$ with image compression demo
- Matrix approximation: reconstruction error vs rank

---

### Notebook 4: Vector Calculus
Gradients, Jacobians, and backpropagation — the mathematics behind training neural networks.

[Open in GitHub](notebooks/04_Vector_Calculus.ipynb) | [Open in Google Colab](https://colab.research.google.com/github/alnemari-m/mathai-website/blob/main/docs/notebooks/04_Vector_Calculus.ipynb) | [Download](notebooks/04_Vector_Calculus.ipynb)

**What's inside:**

- Numerical differentiation (forward and central differences)
- Gradient computation and gradient field visualization
- Gradient descent from scratch (simple and Rosenbrock functions)
- Descent path visualization on contour plots
- Chain rule and backpropagation for a 2-layer neural network
- Hessian computation and Newton's method comparison

---

### Notebook 5: Probability and Distributions
From coin flips to Gaussians — simulate, visualize, and understand probability distributions.

[Open in GitHub](notebooks/05_Probability_Distributions.ipynb) | [Open in Google Colab](https://colab.research.google.com/github/alnemari-m/mathai-website/blob/main/docs/notebooks/05_Probability_Distributions.ipynb) | [Download](notebooks/05_Probability_Distributions.ipynb)

**What's inside:**

- Discrete distributions (Bernoulli, Binomial, Geometric) with PMF plots
- Continuous distributions (Uniform, Exponential, Gaussian) with PDF/CDF plots
- Bayes' Theorem: medical test example with prior vs posterior
- Joint distributions and marginalization with heatmaps
- Covariance and correlation with scatter plots
- Multivariate Gaussian: contour plots and sampling
- Central Limit Theorem demonstration

---

## Quick Reference: All Notebooks

| # | Topic | Colab | Download |
|---|-------|-------|----------|
| 0 | Python Libraries Basics | [Open](https://colab.research.google.com/github/alnemari-m/mathai-website/blob/main/docs/notebooks/01_Python_Libraries_Basics.ipynb) | [ipynb](notebooks/01_Python_Libraries_Basics.ipynb) |
| 2 | Analytic Geometry | [Open](https://colab.research.google.com/github/alnemari-m/mathai-website/blob/main/docs/notebooks/02_Analytic_Geometry.ipynb) | [ipynb](notebooks/02_Analytic_Geometry.ipynb) |
| 3 | Matrix Decomposition | [Open](https://colab.research.google.com/github/alnemari-m/mathai-website/blob/main/docs/notebooks/03_Matrix_Decomposition.ipynb) | [ipynb](notebooks/03_Matrix_Decomposition.ipynb) |
| 4 | Vector Calculus | [Open](https://colab.research.google.com/github/alnemari-m/mathai-website/blob/main/docs/notebooks/04_Vector_Calculus.ipynb) | [ipynb](notebooks/04_Vector_Calculus.ipynb) |
| 5 | Probability & Distributions | [Open](https://colab.research.google.com/github/alnemari-m/mathai-website/blob/main/docs/notebooks/05_Probability_Distributions.ipynb) | [ipynb](notebooks/05_Probability_Distributions.ipynb) |

---

## Setup & Usage

### Option 1: Google Colab (Recommended)
Click any "Open in Google Colab" link above — no setup required!

### Option 2: Local Jupyter
```bash
# Install dependencies
pip install jupyter numpy scipy matplotlib pandas

# Start Jupyter
jupyter notebook
```

### Option 3: VS Code
Install Python and Jupyter extensions, then open `.ipynb` files directly.

---

## Notebook Structure

Each notebook includes:

1. **Learning Objectives** - What you'll learn
2. **Theory Review** - Key concepts from lectures
3. **Implementation** - Step-by-step code with explanations
4. **Visualizations** - Plots and graphs for intuition
5. **Practice Exercises** - 4 exercises per notebook

---

## Best Practices

**Before Starting:**

- Review corresponding lecture slides
- Read the matching math tutorial
- Understand the theory before coding

**While Working:**

- Run cells in order
- Experiment with parameters
- Add your own test cases
- Write notes in markdown cells

**After Completing:**

- Compare your implementations with library functions (SciPy, scikit-learn)
- Test edge cases
- Try the practice exercises

---

<div class="signature" markdown="1">

*Notebooks developed by Mohammed Alnemari*
*Mathematics of AI &bull; Spring 2026*

</div>

---

<div class="last-updated" markdown="1">
**Last Updated:** February 8, 2026
</div>
