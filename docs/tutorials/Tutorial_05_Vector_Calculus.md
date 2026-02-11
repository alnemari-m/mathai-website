# Tutorial 4: Vector Calculus

**Course:** Mathematics for Machine Learning
**Instructor:** Mohammed Alnemari

---

## 📚 Learning Objectives

By the end of this tutorial, you will understand:

1. Differentiation of univariate functions and basic derivative rules
2. Taylor series and polynomial approximation
3. Partial derivatives and gradients
4. Jacobians for vector-valued functions
5. Matrix calculus rules and gradient identities
6. The chain rule in single-variable and multivariate settings
7. Backpropagation and computation graphs
8. Higher-order derivatives and the Hessian matrix
9. Useful gradient identities for machine learning

---

## Part 1: Differentiation of Univariate Functions

### 1.1 Definition of the Derivative

The **derivative** of a function $f(x)$ measures the instantaneous rate of change of $f$ with respect to $x$.

$$\frac{df}{dx} = f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

**Geometric interpretation:** The derivative at a point gives the slope of the tangent line to the curve at that point.

### 1.2 Basic Derivative Rules

| Rule | Function $f(x)$ | Derivative $f'(x)$ | Example |
|------|-----------------|---------------------|---------|
| Constant | $c$ | $0$ | $\frac{d}{dx}(5) = 0$ |
| Power Rule | $x^n$ | $nx^{n-1}$ | $\frac{d}{dx}(x^3) = 3x^2$ |
| Exponential | $e^x$ | $e^x$ | $\frac{d}{dx}(e^x) = e^x$ |
| Logarithm | $\ln(x)$ | $\frac{1}{x}$ | $\frac{d}{dx}(\ln x) = \frac{1}{x}$ |
| Sine | $\sin(x)$ | $\cos(x)$ | $\frac{d}{dx}(\sin x) = \cos x$ |
| Cosine | $\cos(x)$ | $-\sin(x)$ | $\frac{d}{dx}(\cos x) = -\sin x$ |

### 1.3 Combination Rules

**Sum Rule:**
$$\frac{d}{dx}\left[f(x) + g(x)\right] = f'(x) + g'(x)$$

**Product Rule:**
$$\frac{d}{dx}\left[f(x) \cdot g(x)\right] = f'(x) \cdot g(x) + f(x) \cdot g'(x)$$

**Quotient Rule:**
$$\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x) \cdot g(x) - f(x) \cdot g'(x)}{\left[g(x)\right]^2}$$

**Chain Rule (single variable):**
$$\frac{d}{dx}\left[f(g(x))\right] = f'(g(x)) \cdot g'(x)$$

### 1.4 Worked Examples

**Example 1 (Product Rule):** Find $\frac{d}{dx}\left[x^2 \cdot e^x\right]$.

$$\frac{d}{dx}\left[x^2 \cdot e^x\right] = 2x \cdot e^x + x^2 \cdot e^x = e^x(2x + x^2)$$

**Example 2 (Chain Rule):** Find $\frac{d}{dx}\left[e^{-x^2}\right]$.

Let $u = -x^2$, so $f(u) = e^u$.

$$\frac{d}{dx}\left[e^{-x^2}\right] = e^{-x^2} \cdot (-2x) = -2x \, e^{-x^2}$$

**Example 3 (Quotient Rule):** Find the derivative of the sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$.

$$\sigma'(x) = \frac{0 \cdot (1 + e^{-x}) - 1 \cdot (-e^{-x})}{(1 + e^{-x})^2} = \frac{e^{-x}}{(1 + e^{-x})^2}$$

This simplifies to the elegant identity:

$$\sigma'(x) = \sigma(x)\left(1 - \sigma(x)\right)$$

---

## Part 2: Taylor Series

### 2.1 Taylor Series Definition

A **Taylor series** expands a smooth function $f(x)$ around a point $x_0$ as an infinite polynomial:

$$f(x) = \sum_{k=0}^{\infty} \frac{f^{(k)}(x_0)}{k!}(x - x_0)^k$$

$$= f(x_0) + f'(x_0)(x - x_0) + \frac{f''(x_0)}{2!}(x - x_0)^2 + \frac{f'''(x_0)}{3!}(x - x_0)^3 + \cdots$$

### 2.2 Taylor Polynomial Approximations

In machine learning, we often use truncated Taylor polynomials for local approximation.

**First-order (linear) approximation:**
$$f(x) \approx f(x_0) + f'(x_0)(x - x_0)$$

**Second-order (quadratic) approximation:**
$$f(x) \approx f(x_0) + f'(x_0)(x - x_0) + \frac{f''(x_0)}{2}(x - x_0)^2$$

### 2.3 Why Taylor Series Matter in ML

| Approximation Order | Use in Machine Learning |
|---------------------|------------------------|
| First-order | Gradient descent (linear approximation of loss function) |
| Second-order | Newton's method (quadratic approximation of loss function) |

### 2.4 Worked Example

**Example:** Approximate $e^x$ around $x_0 = 0$ to second order.

We need $f(0)$, $f'(0)$, and $f''(0)$. Since $f(x) = e^x$, all derivatives are $e^x$, so $f(0) = f'(0) = f''(0) = 1$.

$$e^x \approx 1 + x + \frac{x^2}{2}$$

Checking: at $x = 0.1$, the true value is $e^{0.1} = 1.10517...$

$$1 + 0.1 + \frac{0.01}{2} = 1.105$$

The approximation is excellent near $x_0$!

---

## Part 3: Partial Derivatives

### 3.1 Definition

For a function of multiple variables $f(x_1, x_2, \ldots, x_n)$, the **partial derivative** with respect to $x_i$ measures how $f$ changes when only $x_i$ varies, with all other variables held constant.

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

### 3.2 Notation

Partial derivatives have several equivalent notations:

| Notation | Meaning |
|----------|---------|
| $\frac{\partial f}{\partial x}$ | Partial derivative of $f$ with respect to $x$ |
| $f_x$ | Shorthand for $\frac{\partial f}{\partial x}$ |
| $\partial_x f$ | Another shorthand |
| $D_x f$ | Differential operator notation |

### 3.3 How to Compute Partial Derivatives

**Rule:** To find $\frac{\partial f}{\partial x_i}$, treat every variable except $x_i$ as a constant, then differentiate with respect to $x_i$ using the standard rules.

**Example 1:** Let $f(x, y) = x^2 y + 3xy^2 - 2y$.

$$\frac{\partial f}{\partial x} = 2xy + 3y^2$$

$$\frac{\partial f}{\partial y} = x^2 + 6xy - 2$$

**Example 2:** Let $f(x, y) = e^{xy} + \sin(x)$.

$$\frac{\partial f}{\partial x} = y \, e^{xy} + \cos(x)$$

$$\frac{\partial f}{\partial y} = x \, e^{xy}$$

---

## Part 4: Gradients

### 4.1 Definition

The **gradient** of a scalar-valued function $f: \mathbb{R}^n \to \mathbb{R}$ is a vector of all its partial derivatives:

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} \in \mathbb{R}^n$$

The gradient "lives" in the same space as the input $\mathbf{x}$.

### 4.2 Gradient as Direction of Steepest Ascent

The gradient has a fundamental geometric meaning:

- $\nabla f(\mathbf{x})$ **points in the direction of steepest ascent** of $f$ at $\mathbf{x}$
- $-\nabla f(\mathbf{x})$ **points in the direction of steepest descent**
- $\|\nabla f(\mathbf{x})\|$ gives the **rate of steepest ascent**

This is why **gradient descent** updates parameters as:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \, \nabla f(\mathbf{x}_t)$$

where $\eta > 0$ is the learning rate.

### 4.3 Worked Example

**Example:** Find the gradient of $f(x_1, x_2, x_3) = x_1^2 + 2x_1 x_2 + x_3^3$.

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \frac{\partial f}{\partial x_3} \end{bmatrix} = \begin{bmatrix} 2x_1 + 2x_2 \\ 2x_1 \\ 3x_3^2 \end{bmatrix}$$

At the point $\mathbf{x} = \begin{bmatrix} 1 \\ 2 \\ -1 \end{bmatrix}$:

$$\nabla f\big|_{\mathbf{x}} = \begin{bmatrix} 2(1) + 2(2) \\ 2(1) \\ 3(-1)^2 \end{bmatrix} = \begin{bmatrix} 6 \\ 2 \\ 3 \end{bmatrix}$$

The direction of steepest descent at this point is $-\nabla f = \begin{bmatrix} -6 \\ -2 \\ -3 \end{bmatrix}$.

---

## Part 5: Jacobians

### 5.1 Definition

For a **vector-valued function** $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ with components $f_1, f_2, \ldots, f_m$, the **Jacobian** is the $m \times n$ matrix of all first-order partial derivatives:

$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix} \in \mathbb{R}^{m \times n}$$

**Key observation:** Each row of the Jacobian is the gradient (transposed) of one output component $f_i$.

### 5.2 Relationship to Gradients

| Object | Input | Output | Derivative |
|--------|-------|--------|-----------|
| Gradient $\nabla f$ | $\mathbb{R}^n$ | $\mathbb{R}$ (scalar) | Vector in $\mathbb{R}^n$ |
| Jacobian $\mathbf{J}$ | $\mathbb{R}^n$ | $\mathbb{R}^m$ (vector) | Matrix in $\mathbb{R}^{m \times n}$ |

### 5.3 Worked Example

**Example:** Let $\mathbf{f}: \mathbb{R}^2 \to \mathbb{R}^3$ be defined by:

$$\mathbf{f}(x_1, x_2) = \begin{bmatrix} x_1^2 x_2 \\ 5x_1 + \sin(x_2) \\ x_2^2 \end{bmatrix}$$

The Jacobian is:

$$\mathbf{J} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} \\
\frac{\partial f_3}{\partial x_1} & \frac{\partial f_3}{\partial x_2}
\end{bmatrix} = \begin{bmatrix}
2x_1 x_2 & x_1^2 \\
5 & \cos(x_2) \\
0 & 2x_2
\end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

---

## Part 6: Gradients of Matrices

### 6.1 Matrix Calculus Rules

When working with vectors and matrices, we need special differentiation rules.

**Gradient of a linear function:** For $f(\mathbf{x}) = \mathbf{a}^T \mathbf{x}$ where $\mathbf{a}, \mathbf{x} \in \mathbb{R}^n$:

$$\nabla_{\mathbf{x}}(\mathbf{a}^T \mathbf{x}) = \mathbf{a}$$

**Gradient of a quadratic form:** For $f(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x}$ where $\mathbf{A} \in \mathbb{R}^{n \times n}$:

$$\nabla_{\mathbf{x}}(\mathbf{x}^T \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^T)\mathbf{x}$$

If $\mathbf{A}$ is symmetric ($\mathbf{A} = \mathbf{A}^T$), this simplifies to:

$$\nabla_{\mathbf{x}}(\mathbf{x}^T \mathbf{A} \mathbf{x}) = 2\mathbf{A}\mathbf{x}$$

### 6.2 Useful Matrix Calculus Identities

| Function $f(\mathbf{x})$ | Gradient $\nabla_{\mathbf{x}} f$ |
|--------------------------|----------------------------------|
| $\mathbf{a}^T \mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^T \mathbf{a}$ | $\mathbf{a}$ |
| $\mathbf{x}^T \mathbf{x}$ | $2\mathbf{x}$ |
| $\mathbf{x}^T \mathbf{A} \mathbf{x}$ | $(\mathbf{A} + \mathbf{A}^T)\mathbf{x}$ |
| $\|\mathbf{x} - \mathbf{b}\|^2$ | $2(\mathbf{x} - \mathbf{b})$ |
| $\mathbf{b}^T \mathbf{A} \mathbf{x}$ | $\mathbf{A}^T \mathbf{b}$ |

### 6.3 Worked Example

**Example:** Find the gradient of the least-squares loss.

The loss function is:

$$L(\mathbf{w}) = \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 = (\mathbf{X}\mathbf{w} - \mathbf{y})^T(\mathbf{X}\mathbf{w} - \mathbf{y})$$

Expanding:

$$L(\mathbf{w}) = \mathbf{w}^T \mathbf{X}^T \mathbf{X} \mathbf{w} - 2\mathbf{y}^T \mathbf{X} \mathbf{w} + \mathbf{y}^T \mathbf{y}$$

Taking the gradient with respect to $\mathbf{w}$:

$$\nabla_{\mathbf{w}} L = 2\mathbf{X}^T \mathbf{X} \mathbf{w} - 2\mathbf{X}^T \mathbf{y}$$

Setting $\nabla_{\mathbf{w}} L = \mathbf{0}$ gives the **normal equation**:

$$\mathbf{X}^T \mathbf{X} \mathbf{w}^* = \mathbf{X}^T \mathbf{y} \quad \Longrightarrow \quad \mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

---

## Part 7: The Chain Rule

### 7.1 Single Variable Chain Rule

If $y = f(g(x))$, then:

$$\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

**Example:** $y = (3x + 1)^4$

Let $g = 3x + 1$, so $y = g^4$.

$$\frac{dy}{dx} = 4g^3 \cdot 3 = 12(3x + 1)^3$$

### 7.2 Multivariate Chain Rule

If $f$ depends on $\mathbf{x}$ through intermediate variables $\mathbf{u}$:

$$\mathbf{x} \in \mathbb{R}^n \xrightarrow{\mathbf{g}} \mathbf{u} \in \mathbb{R}^m \xrightarrow{f} y \in \mathbb{R}$$

Then:

$$\frac{\partial f}{\partial x_i} = \sum_{j=1}^{m} \frac{\partial f}{\partial u_j} \cdot \frac{\partial u_j}{\partial x_i}$$

In matrix form (using Jacobians):

$$\frac{\partial f}{\partial \mathbf{x}} = \frac{\partial f}{\partial \mathbf{u}} \cdot \frac{\partial \mathbf{u}}{\partial \mathbf{x}}$$

### 7.3 Chain Rule for Neural Networks

Consider a simple two-layer neural network:

$$\mathbf{x} \xrightarrow{\mathbf{W}_1} \mathbf{z}_1 \xrightarrow{\sigma} \mathbf{a}_1 \xrightarrow{\mathbf{W}_2} \mathbf{z}_2 \xrightarrow{\text{loss}} L$$

To find $\frac{\partial L}{\partial \mathbf{W}_1}$, we apply the chain rule through the entire computation:

$$\frac{\partial L}{\partial \mathbf{W}_1} = \frac{\partial L}{\partial \mathbf{z}_2} \cdot \frac{\partial \mathbf{z}_2}{\partial \mathbf{a}_1} \cdot \frac{\partial \mathbf{a}_1}{\partial \mathbf{z}_1} \cdot \frac{\partial \mathbf{z}_1}{\partial \mathbf{W}_1}$$

Each term in this product corresponds to a specific operation in the network.

---

## Part 8: Backpropagation

### 8.1 Computation Graphs

A **computation graph** represents a function as a directed acyclic graph (DAG) where:
- **Nodes** represent operations or variables
- **Edges** represent data flow

**Example:** For $f(x, y) = (x + y) \cdot (y + 1)$:

```
x ---\
      (+) = a ---\
y ---/            (*) = f
y ---\           /
      (+) = b --/
1 ---/
```

Here $a = x + y$, $b = y + 1$, and $f = a \cdot b$.

### 8.2 Forward Pass

In the **forward pass**, we compute the output by evaluating the graph from inputs to output.

**Example:** With $x = 2$, $y = 3$:

| Step | Computation | Value |
|------|-------------|-------|
| 1 | $a = x + y$ | $a = 2 + 3 = 5$ |
| 2 | $b = y + 1$ | $b = 3 + 1 = 4$ |
| 3 | $f = a \cdot b$ | $f = 5 \cdot 4 = 20$ |

### 8.3 Backward Pass (Backpropagation)

In the **backward pass**, we compute gradients by traversing the graph from output to inputs, applying the chain rule at each node.

Starting from $\frac{\partial f}{\partial f} = 1$:

| Step | Gradient | Computation | Value |
|------|----------|-------------|-------|
| 1 | $\frac{\partial f}{\partial a}$ | $b$ | $4$ |
| 2 | $\frac{\partial f}{\partial b}$ | $a$ | $5$ |
| 3 | $\frac{\partial f}{\partial x}$ | $\frac{\partial f}{\partial a} \cdot \frac{\partial a}{\partial x} = b \cdot 1$ | $4$ |
| 4 | $\frac{\partial f}{\partial y}$ | $\frac{\partial f}{\partial a} \cdot \frac{\partial a}{\partial y} + \frac{\partial f}{\partial b} \cdot \frac{\partial b}{\partial y} = b \cdot 1 + a \cdot 1$ | $4 + 5 = 9$ |

**Note:** Since $y$ appears in two paths ($a$ and $b$), we **sum** the contributions from both paths.

### 8.4 General Backpropagation Algorithm

For a computation graph with output $L$:

1. **Forward pass:** Compute all intermediate values from inputs to output
2. **Initialize:** Set $\frac{\partial L}{\partial L} = 1$
3. **Backward pass:** For each node $v$ in reverse topological order:

$$\frac{\partial L}{\partial v} = \sum_{u \in \text{children}(v)} \frac{\partial L}{\partial u} \cdot \frac{\partial u}{\partial v}$$

This is the foundation of training neural networks.

---

## Part 9: Higher-Order Derivatives

### 9.1 Second-Order Partial Derivatives

For a function $f(x_1, x_2, \ldots, x_n)$, we can differentiate partial derivatives again:

$$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial}{\partial x_i}\left(\frac{\partial f}{\partial x_j}\right)$$

**Symmetry of mixed partials (Schwarz's theorem):** If $f$ has continuous second partial derivatives:

$$\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}$$

### 9.2 The Hessian Matrix

The **Hessian** collects all second-order partial derivatives into a matrix:

$$\mathbf{H} = \nabla^2 f = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix} \in \mathbb{R}^{n \times n}$$

**Properties:**
- The Hessian is **symmetric** (by Schwarz's theorem): $\mathbf{H} = \mathbf{H}^T$
- If $\mathbf{H}$ is **positive definite** at a critical point, the point is a **local minimum**
- If $\mathbf{H}$ is **negative definite**, the point is a **local maximum**
- If $\mathbf{H}$ has both positive and negative eigenvalues, the point is a **saddle point**

### 9.3 Second-Order Taylor Expansion (Multivariate)

The multivariate second-order Taylor expansion around $\mathbf{x}_0$ is:

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T (\mathbf{x} - \mathbf{x}_0) + \frac{1}{2}(\mathbf{x} - \mathbf{x}_0)^T \mathbf{H}(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0)$$

This is the basis for **Newton's method** in optimization.

### 9.4 Worked Example

**Example:** Find the Hessian of $f(x_1, x_2) = x_1^3 + 2x_1 x_2^2 - x_2$.

First, compute the gradient:

$$\nabla f = \begin{bmatrix} 3x_1^2 + 2x_2^2 \\ 4x_1 x_2 - 1 \end{bmatrix}$$

Then, compute the Hessian:

$$\mathbf{H} = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2}
\end{bmatrix} = \begin{bmatrix}
6x_1 & 4x_2 \\
4x_2 & 4x_1
\end{bmatrix}$$

Notice that $\mathbf{H} = \mathbf{H}^T$, confirming symmetry.

---

## Part 10: Useful Gradient Identities

### 10.1 Reference Table

These identities appear frequently in machine learning derivations. Here $\mathbf{x}, \mathbf{a}, \mathbf{b} \in \mathbb{R}^n$ and $\mathbf{A} \in \mathbb{R}^{n \times n}$.

| # | Function | Gradient $\nabla_{\mathbf{x}}$ |
|---|----------|-------------------------------|
| 1 | $\mathbf{a}^T \mathbf{x}$ | $\mathbf{a}$ |
| 2 | $\mathbf{x}^T \mathbf{x}$ | $2\mathbf{x}$ |
| 3 | $\mathbf{x}^T \mathbf{A} \mathbf{x}$ | $(\mathbf{A} + \mathbf{A}^T)\mathbf{x}$ |
| 4 | $(\mathbf{A}\mathbf{x} - \mathbf{b})^T(\mathbf{A}\mathbf{x} - \mathbf{b})$ | $2\mathbf{A}^T(\mathbf{A}\mathbf{x} - \mathbf{b})$ |
| 5 | $\|\mathbf{x}\|^2 = \mathbf{x}^T\mathbf{x}$ | $2\mathbf{x}$ |
| 6 | $\mathbf{b}^T \mathbf{A} \mathbf{x}$ | $\mathbf{A}^T \mathbf{b}$ |

### 10.2 Deriving Identity 3

Let us prove $\nabla_{\mathbf{x}}(\mathbf{x}^T \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^T)\mathbf{x}$.

Write $f(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x} = \sum_{i}\sum_{j} x_i \, A_{ij} \, x_j$.

Taking the partial derivative with respect to $x_k$:

$$\frac{\partial f}{\partial x_k} = \sum_{j} A_{kj} \, x_j + \sum_{i} x_i \, A_{ik}$$

$$= (\mathbf{A}\mathbf{x})_k + (\mathbf{A}^T\mathbf{x})_k$$

Collecting into a vector:

$$\nabla_{\mathbf{x}} f = \mathbf{A}\mathbf{x} + \mathbf{A}^T\mathbf{x} = (\mathbf{A} + \mathbf{A}^T)\mathbf{x}$$

### 10.3 When $\mathbf{A}$ is Symmetric

If $\mathbf{A} = \mathbf{A}^T$, then $\mathbf{A} + \mathbf{A}^T = 2\mathbf{A}$, so:

$$\nabla_{\mathbf{x}}(\mathbf{x}^T \mathbf{A} \mathbf{x}) = 2\mathbf{A}\mathbf{x}$$

This is a very common case in machine learning, since covariance matrices and Hessians are symmetric.

---

## Summary: Key Takeaways

### Differentiation Fundamentals
- Derivatives measure rates of change; partial derivatives fix all variables except one
- The gradient $\nabla f$ collects all partial derivatives into a vector
- The Jacobian generalizes the gradient for vector-valued functions

### The Chain Rule and Backpropagation
- The multivariate chain rule composes Jacobians through multiplication
- Backpropagation applies the chain rule on a computation graph, working backward from the loss
- Gradients with respect to variables appearing in multiple paths are **summed**

### Higher-Order Information
- The Hessian matrix $\mathbf{H}$ captures second-order (curvature) information
- Positive definite Hessian at a critical point indicates a local minimum

### Matrix Calculus
- $\nabla_{\mathbf{x}}(\mathbf{a}^T \mathbf{x}) = \mathbf{a}$
- $\nabla_{\mathbf{x}}(\mathbf{x}^T \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^T)\mathbf{x}$
- The normal equation for least squares: $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

---

## Practice Problems

### Problem 1
Find the derivative of $f(x) = x^3 e^{2x}$ using the product and chain rules.

### Problem 2
Let $f(x, y) = x^2 y - 3xy^3 + 2x$. Find $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$, then compute the gradient at the point $(1, -1)$.

### Problem 3
Compute the Jacobian of the function $\mathbf{f}: \mathbb{R}^2 \to \mathbb{R}^2$ defined by:

$$\mathbf{f}(x, y) = \begin{bmatrix} x^2 + y \\ xy - y^2 \end{bmatrix}$$

### Problem 4
Find the Hessian of $f(x_1, x_2) = x_1^2 + 4x_1 x_2 + x_2^2$. Is this Hessian positive definite?

### Problem 5
Consider the computation graph for $f(x) = (x + 2)^2$. Perform the forward pass with $x = 3$, then use backpropagation to compute $\frac{df}{dx}$.

### Problem 6
Let $\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}$ and $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$. Compute $\nabla_{\mathbf{x}}(\mathbf{x}^T \mathbf{A} \mathbf{x})$ using the identity from Part 10, and verify by expanding $\mathbf{x}^T \mathbf{A} \mathbf{x}$ and differentiating directly.

---

## Solutions

**Solution 1:**

Using the product rule with $u = x^3$ and $v = e^{2x}$:

$$f'(x) = \frac{d}{dx}(x^3) \cdot e^{2x} + x^3 \cdot \frac{d}{dx}(e^{2x})$$

$$= 3x^2 \cdot e^{2x} + x^3 \cdot 2e^{2x}$$

$$= e^{2x}(3x^2 + 2x^3) = x^2 e^{2x}(3 + 2x)$$

---

**Solution 2:**

$$\frac{\partial f}{\partial x} = 2xy - 3y^3 + 2$$

$$\frac{\partial f}{\partial y} = x^2 - 9xy^2$$

At $(1, -1)$:

$$\frac{\partial f}{\partial x}\bigg|_{(1,-1)} = 2(1)(-1) - 3(-1)^3 + 2 = -2 + 3 + 2 = 3$$

$$\frac{\partial f}{\partial y}\bigg|_{(1,-1)} = (1)^2 - 9(1)(-1)^2 = 1 - 9 = -8$$

$$\nabla f\big|_{(1,-1)} = \begin{bmatrix} 3 \\ -8 \end{bmatrix}$$

---

**Solution 3:**

$$\mathbf{J} = \begin{bmatrix}
\frac{\partial f_1}{\partial x} & \frac{\partial f_1}{\partial y} \\
\frac{\partial f_2}{\partial x} & \frac{\partial f_2}{\partial y}
\end{bmatrix}$$

For $f_1 = x^2 + y$: $\frac{\partial f_1}{\partial x} = 2x$, $\frac{\partial f_1}{\partial y} = 1$

For $f_2 = xy - y^2$: $\frac{\partial f_2}{\partial x} = y$, $\frac{\partial f_2}{\partial y} = x - 2y$

$$\mathbf{J} = \begin{bmatrix} 2x & 1 \\ y & x - 2y \end{bmatrix}$$

---

**Solution 4:**

First, compute the gradient:

$$\nabla f = \begin{bmatrix} 2x_1 + 4x_2 \\ 4x_1 + 2x_2 \end{bmatrix}$$

The Hessian (matrix of second derivatives):

$$\mathbf{H} = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2}
\end{bmatrix} = \begin{bmatrix} 2 & 4 \\ 4 & 2 \end{bmatrix}$$

To check positive definiteness, compute the eigenvalues. For a $2 \times 2$ matrix:

$$\det(\mathbf{H} - \lambda \mathbf{I}) = (2 - \lambda)^2 - 16 = 0$$

$$\lambda^2 - 4\lambda + 4 - 16 = 0 \implies \lambda^2 - 4\lambda - 12 = 0$$

$$\lambda = \frac{4 \pm \sqrt{16 + 48}}{2} = \frac{4 \pm 8}{2}$$

So $\lambda_1 = 6$ and $\lambda_2 = -2$.

Since one eigenvalue is negative, the Hessian is **not positive definite**. It is **indefinite**, meaning any critical point of $f$ would be a saddle point.

---

**Solution 5:**

Decompose $f(x) = (x + 2)^2$ into elementary steps:

- $a = x + 2$
- $f = a^2$

**Forward pass** with $x = 3$:

| Step | Computation | Value |
|------|-------------|-------|
| 1 | $a = x + 2$ | $a = 3 + 2 = 5$ |
| 2 | $f = a^2$ | $f = 5^2 = 25$ |

**Backward pass:**

| Step | Gradient | Computation | Value |
|------|----------|-------------|-------|
| 1 | $\frac{\partial f}{\partial f}$ | (seed) | $1$ |
| 2 | $\frac{\partial f}{\partial a}$ | $2a$ | $2(5) = 10$ |
| 3 | $\frac{\partial f}{\partial x}$ | $\frac{\partial f}{\partial a} \cdot \frac{\partial a}{\partial x} = 10 \cdot 1$ | $10$ |

**Verification:** $f'(x) = 2(x + 2)$, so $f'(3) = 2(5) = 10$. Correct!

---

**Solution 6:**

**Using the identity:**

Since $\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}$ is symmetric ($\mathbf{A} = \mathbf{A}^T$):

$$\nabla_{\mathbf{x}}(\mathbf{x}^T \mathbf{A} \mathbf{x}) = 2\mathbf{A}\mathbf{x} = 2\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}\begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 4x_1 + 2x_2 \\ 2x_1 + 6x_2 \end{bmatrix}$$

**Direct verification:**

Expand $\mathbf{x}^T \mathbf{A} \mathbf{x}$:

$$\mathbf{x}^T \mathbf{A} \mathbf{x} = \begin{bmatrix} x_1 & x_2 \end{bmatrix}\begin{bmatrix} 2x_1 + x_2 \\ x_1 + 3x_2 \end{bmatrix} = 2x_1^2 + x_1 x_2 + x_1 x_2 + 3x_2^2 = 2x_1^2 + 2x_1 x_2 + 3x_2^2$$

Taking partial derivatives:

$$\frac{\partial}{\partial x_1}(2x_1^2 + 2x_1 x_2 + 3x_2^2) = 4x_1 + 2x_2$$

$$\frac{\partial}{\partial x_2}(2x_1^2 + 2x_1 x_2 + 3x_2^2) = 2x_1 + 6x_2$$

$$\nabla_{\mathbf{x}}(\mathbf{x}^T \mathbf{A} \mathbf{x}) = \begin{bmatrix} 4x_1 + 2x_2 \\ 2x_1 + 6x_2 \end{bmatrix}$$

Both methods agree, confirming the identity.

---

**Course:** Mathematics for Machine Learning
**Instructor:** Mohammed Alnemari

**Next:** Tutorial 5 - Probability and Distributions
