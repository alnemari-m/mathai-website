# Tutorial 3: Matrix Decomposition

**Course:** Mathematics for Machine Learning
**Instructor:** Mohammed Alnemari

---

## 📚 Learning Objectives

By the end of this tutorial, you will understand:

1. How to compute determinants for 2x2 and 3x3 matrices
2. The trace of a matrix and its properties
3. Eigenvalues and eigenvectors and how to compute them
4. Cholesky decomposition for symmetric positive definite matrices
5. Eigendecomposition and diagonalization
6. Singular Value Decomposition (SVD) and its geometric meaning
7. Matrix approximation using truncated SVD

---

## Part 1: Determinants

### 1.1 What is a Determinant?

The **determinant** is a scalar value computed from a square matrix that captures important information about the matrix. Think of it as a single number that tells you:

- Whether the matrix is invertible (determinant is nonzero)
- How the matrix scales areas or volumes when used as a linear transformation
- The "signed volume" of the parallelepiped formed by the column vectors

**Notation:** For a matrix $A$, the determinant is written as $\det(A)$ or $|A|$.

### 1.2 Determinant of a 2x2 Matrix

For a $2 \times 2$ matrix:

$$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$$

The determinant is:

$$\det(A) = ad - bc$$

In plain English: multiply the diagonals and subtract. The main diagonal product minus the off-diagonal product.

**Worked Example:**

$$A = \begin{bmatrix} 3 & 7 \\ 1 & 5 \end{bmatrix}$$

$$\det(A) = 3(5) - 7(1) = 15 - 7 = 8$$

Since $\det(A) = 8 \neq 0$, the matrix $A$ is invertible.

### 1.3 Determinant of a 3x3 Matrix

For a $3 \times 3$ matrix:

$$A = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}$$

#### Method 1: Sarrus' Rule

Write the matrix and repeat the first two columns to the right:

$$\begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix} \begin{matrix} a_{11} & a_{12} \\ a_{21} & a_{22} \\ a_{31} & a_{32} \end{matrix}$$

Then sum the products along the three downward diagonals and subtract the products along the three upward diagonals:

$$\det(A) = a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32} - a_{13}a_{22}a_{31} - a_{12}a_{21}a_{33} - a_{11}a_{23}a_{32}$$

#### Method 2: Cofactor Expansion (along the first row)

$$\det(A) = a_{11} \begin{vmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{vmatrix} - a_{12} \begin{vmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{vmatrix} + a_{13} \begin{vmatrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{vmatrix}$$

Each smaller determinant is called a **minor**, and the signed minor is a **cofactor**.

**Worked Example:**

$$B = \begin{bmatrix} 2 & 1 & 3 \\ 0 & 4 & 5 \\ 1 & 0 & 2 \end{bmatrix}$$

Using cofactor expansion along the first row:

$$\det(B) = 2 \begin{vmatrix} 4 & 5 \\ 0 & 2 \end{vmatrix} - 1 \begin{vmatrix} 0 & 5 \\ 1 & 2 \end{vmatrix} + 3 \begin{vmatrix} 0 & 4 \\ 1 & 0 \end{vmatrix}$$

$$= 2(4 \cdot 2 - 5 \cdot 0) - 1(0 \cdot 2 - 5 \cdot 1) + 3(0 \cdot 0 - 4 \cdot 1)$$

$$= 2(8) - 1(-5) + 3(-4)$$

$$= 16 + 5 - 12 = 9$$

### 1.4 Properties of Determinants

| Property | Statement | Example / Note |
|----------|-----------|----------------|
| Identity | $\det(I) = 1$ | The identity matrix always has determinant 1 |
| Transpose | $\det(A^T) = \det(A)$ | Transposing does not change the determinant |
| Product | $\det(AB) = \det(A) \cdot \det(B)$ | Determinant of a product is the product of determinants |
| Inverse | $\det(A^{-1}) = \frac{1}{\det(A)}$ | Only defined when $\det(A) \neq 0$ |
| Scalar multiple | $\det(cA) = c^n \det(A)$ | For an $n \times n$ matrix |
| Singular matrix | $\det(A) = 0$ | Matrix is not invertible |
| Row swap | Swapping two rows flips the sign | $\det(\text{swapped}) = -\det(A)$ |
| Triangular | $\det(A) = \prod_{i=1}^{n} a_{ii}$ | Product of diagonal entries for triangular matrices |

---

## Part 2: Trace

### 2.1 Definition

The **trace** of a square matrix $A$ is the sum of its diagonal entries:

$$\text{tr}(A) = \sum_{i=1}^{n} a_{ii} = a_{11} + a_{22} + \cdots + a_{nn}$$

In plain English: just add up all the numbers on the main diagonal.

**Example:**

$$A = \begin{bmatrix} 5 & 2 & 1 \\ 0 & 3 & 4 \\ 7 & 6 & 8 \end{bmatrix}$$

$$\text{tr}(A) = 5 + 3 + 8 = 16$$

### 2.2 Properties of the Trace

| Property | Statement |
|----------|-----------|
| Linearity | $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$ |
| Scalar multiplication | $\text{tr}(cA) = c \cdot \text{tr}(A)$ |
| Transpose | $\text{tr}(A^T) = \text{tr}(A)$ |
| Cyclic property | $\text{tr}(AB) = \text{tr}(BA)$ |
| Cyclic property (3 matrices) | $\text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB)$ |
| Sum of eigenvalues | $\text{tr}(A) = \sum_{i=1}^{n} \lambda_i$ |
| Frobenius norm | $\text{tr}(A^T A) = \sum_{i,j} a_{ij}^2 = \|A\|_F^2$ |

The **cyclic property** $\text{tr}(AB) = \text{tr}(BA)$ is especially important in machine learning. It lets you rearrange matrix products inside a trace, which simplifies many derivations in optimization and statistics.

**Worked Example:** Verify $\text{tr}(AB) = \text{tr}(BA)$.

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$$

$$AB = \begin{bmatrix} 1(5)+2(7) & 1(6)+2(8) \\ 3(5)+4(7) & 3(6)+4(8) \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

$$\text{tr}(AB) = 19 + 50 = 69$$

$$BA = \begin{bmatrix} 5(1)+6(3) & 5(2)+6(4) \\ 7(1)+8(3) & 7(2)+8(4) \end{bmatrix} = \begin{bmatrix} 23 & 34 \\ 31 & 46 \end{bmatrix}$$

$$\text{tr}(BA) = 23 + 46 = 69$$

Both sides give 69, confirming the property.

---

## Part 3: Eigenvalues and Eigenvectors

### 3.1 Definition

Given a square matrix $A$, a nonzero vector $\mathbf{v}$ is an **eigenvector** of $A$ if multiplying $A$ by $\mathbf{v}$ simply scales $\mathbf{v}$ by some scalar $\lambda$:

$$A\mathbf{v} = \lambda \mathbf{v}$$

- $\lambda$ is called the **eigenvalue** corresponding to $\mathbf{v}$
- $\mathbf{v}$ is the **eigenvector** corresponding to $\lambda$

In plain English: an eigenvector is a special direction that the matrix only stretches (or flips), but does not rotate. The eigenvalue tells you by how much it stretches.

### 3.2 The Characteristic Polynomial

To find eigenvalues, we rearrange $A\mathbf{v} = \lambda \mathbf{v}$:

$$A\mathbf{v} - \lambda \mathbf{v} = \mathbf{0}$$

$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

For a nonzero solution $\mathbf{v}$ to exist, the matrix $(A - \lambda I)$ must be singular, meaning:

$$\det(A - \lambda I) = 0$$

This equation is called the **characteristic equation**, and the polynomial on the left side is the **characteristic polynomial**. Solving it gives us the eigenvalues.

### 3.3 Worked Example: Finding Eigenvalues and Eigenvectors

**Find the eigenvalues and eigenvectors of:**

$$A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$

**Step 1: Characteristic polynomial.**

$$A - \lambda I = \begin{bmatrix} 4 - \lambda & 1 \\ 2 & 3 - \lambda \end{bmatrix}$$

$$\det(A - \lambda I) = (4 - \lambda)(3 - \lambda) - (1)(2)$$

$$= 12 - 4\lambda - 3\lambda + \lambda^2 - 2$$

$$= \lambda^2 - 7\lambda + 10$$

**Step 2: Solve the characteristic equation.**

$$\lambda^2 - 7\lambda + 10 = 0$$

$$(\lambda - 5)(\lambda - 2) = 0$$

$$\lambda_1 = 5, \quad \lambda_2 = 2$$

**Step 3: Find eigenvectors for each eigenvalue.**

**For $\lambda_1 = 5$:**

$$(A - 5I)\mathbf{v} = \mathbf{0}$$

$$\begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

From the first row: $-v_1 + v_2 = 0$, so $v_2 = v_1$.

Choosing $v_1 = 1$:

$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

**For $\lambda_2 = 2$:**

$$(A - 2I)\mathbf{v} = \mathbf{0}$$

$$\begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

From the first row: $2v_1 + v_2 = 0$, so $v_2 = -2v_1$.

Choosing $v_1 = 1$:

$$\mathbf{v}_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$$

**Quick check:** $\text{tr}(A) = 4 + 3 = 7 = 5 + 2 = \lambda_1 + \lambda_2$ and $\det(A) = 4(3) - 1(2) = 10 = 5 \times 2 = \lambda_1 \cdot \lambda_2$. Both checks pass.

### 3.4 Key Facts About Eigenvalues

| Fact | Statement |
|------|-----------|
| Sum of eigenvalues | $\sum \lambda_i = \text{tr}(A)$ |
| Product of eigenvalues | $\prod \lambda_i = \det(A)$ |
| Symmetric matrices | All eigenvalues are real; eigenvectors are orthogonal |
| Positive definite | All eigenvalues are strictly positive |
| Singular matrix | At least one eigenvalue is zero |

---

## Part 4: Cholesky Decomposition

### 4.1 What is Cholesky Decomposition?

For a **symmetric positive definite (SPD)** matrix $A$, the Cholesky decomposition factors $A$ into:

$$A = LL^T$$

where $L$ is a **lower triangular** matrix with positive diagonal entries.

Think of it as the "square root" of a matrix. Just as every positive number has a square root, every SPD matrix has a Cholesky factor.

### 4.2 What Does "Symmetric Positive Definite" Mean?

A matrix $A$ is **symmetric positive definite** if:

1. **Symmetric:** $A = A^T$ (the matrix equals its transpose)
2. **Positive definite:** $\mathbf{x}^T A \mathbf{x} > 0$ for all nonzero vectors $\mathbf{x}$

Equivalently, $A$ is SPD if it is symmetric and all its eigenvalues are strictly positive.

### 4.3 The Cholesky Algorithm (for 2x2)

Given:

$$A = \begin{bmatrix} a_{11} & a_{12} \\ a_{12} & a_{22} \end{bmatrix} = \begin{bmatrix} l_{11} & 0 \\ l_{21} & l_{22} \end{bmatrix} \begin{bmatrix} l_{11} & l_{21} \\ 0 & l_{22} \end{bmatrix}$$

Expanding the right side and matching entries:

$$l_{11} = \sqrt{a_{11}}$$

$$l_{21} = \frac{a_{12}}{l_{11}}$$

$$l_{22} = \sqrt{a_{22} - l_{21}^2}$$

### 4.4 The Cholesky Algorithm (general)

For an $n \times n$ SPD matrix, the entries of $L$ are computed as:

$$l_{jj} = \sqrt{a_{jj} - \sum_{k=1}^{j-1} l_{jk}^2}$$

$$l_{ij} = \frac{1}{l_{jj}} \left( a_{ij} - \sum_{k=1}^{j-1} l_{ik} l_{jk} \right), \quad \text{for } i > j$$

### 4.5 Worked Example

**Find the Cholesky decomposition of:**

$$A = \begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix}$$

**Step 1:** Check that $A$ is SPD. It is symmetric ($A = A^T$). Its eigenvalues are $\lambda = \frac{9 \pm \sqrt{1}}{2}$, giving $\lambda_1 \approx 5.56$ and $\lambda_2 \approx 3.44$, both positive. (Alternatively, $\det(A) = 16 > 0$ and $a_{11} = 4 > 0$.)

**Step 2:** Compute $L$.

$$l_{11} = \sqrt{4} = 2$$

$$l_{21} = \frac{2}{2} = 1$$

$$l_{22} = \sqrt{5 - 1^2} = \sqrt{4} = 2$$

**Result:**

$$L = \begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix}$$

**Verification:**

$$LL^T = \begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix} = A \quad \checkmark$$

### 4.6 Why Cholesky Decomposition Matters

| Application | How It Helps |
|-------------|-------------|
| Solving linear systems | $Ax = b$ becomes two easy triangular solves: $Ly = b$, then $L^Tx = y$ |
| Sampling from multivariate Gaussians | If $\Sigma = LL^T$, generate $\mathbf{x} = L\mathbf{z} + \boldsymbol{\mu}$ where $\mathbf{z} \sim \mathcal{N}(0, I)$ |
| Numerical stability | More stable and about twice as fast as general LU decomposition for SPD matrices |

---

## Part 5: Eigendecomposition

### 5.1 Definition

If a square matrix $A$ has $n$ linearly independent eigenvectors, then $A$ can be factored as:

$$A = PDP^{-1}$$

where:
- $P$ is the matrix whose columns are the eigenvectors of $A$: $P = [\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_n]$
- $D$ is a diagonal matrix with the corresponding eigenvalues on the diagonal:

$$D = \begin{bmatrix} \lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n \end{bmatrix}$$

This factorization is called the **eigendecomposition** or **spectral decomposition** (for symmetric matrices).

### 5.2 When Does Eigendecomposition Exist?

| Condition | Eigendecomposition Exists? |
|-----------|---------------------------|
| $A$ has $n$ distinct eigenvalues | Always yes |
| $A$ is symmetric ($A = A^T$) | Always yes, and $P$ is orthogonal ($P^{-1} = P^T$) |
| $A$ has repeated eigenvalues | Sometimes (depends on geometric multiplicity) |
| $A$ is defective (not enough independent eigenvectors) | No |

For **symmetric matrices**, the decomposition simplifies to:

$$A = PDP^T$$

because the eigenvectors are orthogonal.

### 5.3 Worked Example

**Diagonalize the matrix from Part 3:**

$$A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$

We already found $\lambda_1 = 5$, $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ and $\lambda_2 = 2$, $\mathbf{v}_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$.

$$P = \begin{bmatrix} 1 & 1 \\ 1 & -2 \end{bmatrix}, \quad D = \begin{bmatrix} 5 & 0 \\ 0 & 2 \end{bmatrix}$$

Compute $P^{-1}$. For a $2 \times 2$ matrix $\begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the inverse is $\frac{1}{ad - bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$:

$$P^{-1} = \frac{1}{(1)(-2) - (1)(1)} \begin{bmatrix} -2 & -1 \\ -1 & 1 \end{bmatrix} = \frac{1}{-3} \begin{bmatrix} -2 & -1 \\ -1 & 1 \end{bmatrix} = \begin{bmatrix} \frac{2}{3} & \frac{1}{3} \\ \frac{1}{3} & -\frac{1}{3} \end{bmatrix}$$

**Verification:**

$$PDP^{-1} = \begin{bmatrix} 1 & 1 \\ 1 & -2 \end{bmatrix} \begin{bmatrix} 5 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} \frac{2}{3} & \frac{1}{3} \\ \frac{1}{3} & -\frac{1}{3} \end{bmatrix}$$

$$= \begin{bmatrix} 5 & 2 \\ 5 & -4 \end{bmatrix} \begin{bmatrix} \frac{2}{3} & \frac{1}{3} \\ \frac{1}{3} & -\frac{1}{3} \end{bmatrix} = \begin{bmatrix} \frac{10}{3} + \frac{2}{3} & \frac{5}{3} - \frac{2}{3} \\ \frac{10}{3} - \frac{4}{3} & \frac{5}{3} + \frac{4}{3} \end{bmatrix} = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} = A \quad \checkmark$$

### 5.4 Why Eigendecomposition is Useful

**Computing matrix powers:** If $A = PDP^{-1}$, then:

$$A^k = PD^kP^{-1}$$

Since $D$ is diagonal, $D^k$ is just each diagonal entry raised to the $k$-th power:

$$D^k = \begin{bmatrix} \lambda_1^k & 0 \\ 0 & \lambda_2^k \end{bmatrix}$$

This makes computing $A^{100}$ just as easy as computing $A^2$.

---

## Part 6: Singular Value Decomposition (SVD)

### 6.1 Definition

**Every** matrix $A$ (of any shape) can be decomposed as:

$$A = U \Sigma V^T$$

where:
- $A$ is $m \times n$
- $U$ is $m \times m$ orthogonal matrix (columns are **left singular vectors**)
- $\Sigma$ is $m \times n$ diagonal matrix (diagonal entries are **singular values** $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$)
- $V$ is $n \times n$ orthogonal matrix (columns are **right singular vectors**)

This is the **most important matrix decomposition** in applied mathematics and machine learning.

### 6.2 Geometric Interpretation

Any linear transformation $A$ can be broken down into three simple steps:

1. **$V^T$: Rotate** (in the input space)
2. **$\Sigma$: Scale** along each axis (possibly changing dimensions)
3. **$U$: Rotate** (in the output space)

In plain English: every matrix transformation is just a rotation, followed by a stretch, followed by another rotation.

### 6.3 Relationship to Eigendecomposition

The singular values and vectors are connected to eigenvalues:

| SVD Component | Obtained From |
|---------------|---------------|
| $V$ (right singular vectors) | Eigenvectors of $A^T A$ |
| $U$ (left singular vectors) | Eigenvectors of $A A^T$ |
| $\sigma_i$ (singular values) | $\sigma_i = \sqrt{\lambda_i}$ where $\lambda_i$ are eigenvalues of $A^T A$ |

### 6.4 Worked Example: Computing the SVD

**Find the SVD of:**

$$A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix}$$

This is already a diagonal matrix, so the SVD is straightforward.

**Step 1: Compute $A^T A$.**

$$A^T A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 9 & 0 \\ 0 & 4 \end{bmatrix}$$

**Step 2: Find eigenvalues of $A^T A$.**

Eigenvalues: $\lambda_1 = 9$, $\lambda_2 = 4$.

Singular values: $\sigma_1 = \sqrt{9} = 3$, $\sigma_2 = \sqrt{4} = 2$.

**Step 3: Find $V$ (eigenvectors of $A^T A$).**

$$V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I$$

**Step 4: Find $U$ (eigenvectors of $A A^T$).**

$$A A^T = \begin{bmatrix} 9 & 0 \\ 0 & 4 \end{bmatrix}$$

$$U = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I$$

**Result:**

$$A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 3 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I \Sigma I = \Sigma$$

### 6.5 Worked Example: Non-Diagonal Matrix

**Find the SVD of:**

$$A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$$

**Step 1: Compute $A^T A$.**

$$A^T A = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & 2 \end{bmatrix}$$

**Step 2: Find eigenvalues of $A^T A$.**

$$\det(A^T A - \lambda I) = (1 - \lambda)(2 - \lambda) - 1 = \lambda^2 - 3\lambda + 1 = 0$$

$$\lambda = \frac{3 \pm \sqrt{5}}{2}$$

$$\lambda_1 = \frac{3 + \sqrt{5}}{2} \approx 2.618, \quad \lambda_2 = \frac{3 - \sqrt{5}}{2} \approx 0.382$$

Singular values:

$$\sigma_1 = \sqrt{\frac{3 + \sqrt{5}}{2}} \approx 1.618, \quad \sigma_2 = \sqrt{\frac{3 - \sqrt{5}}{2}} \approx 0.618$$

(Notice: $\sigma_1 \approx \phi$, the golden ratio, and $\sigma_2 \approx 1/\phi$.)

**Step 3: Find $V$.**

For $\lambda_1 = \frac{3+\sqrt{5}}{2}$, solve $(A^TA - \lambda_1 I)\mathbf{v} = 0$:

$$\begin{bmatrix} 1 - \lambda_1 & 1 \\ 1 & 2 - \lambda_1 \end{bmatrix}\mathbf{v} = 0$$

This gives $v_2 = (\lambda_1 - 1) v_1$. Normalizing:

$$\mathbf{v}_1 = \frac{1}{\sqrt{1 + (\lambda_1 - 1)^2}}\begin{bmatrix} 1 \\ \lambda_1 - 1 \end{bmatrix}$$

Similarly for $\lambda_2$. Then $V = [\mathbf{v}_1 \mid \mathbf{v}_2]$.

**Step 4: Find $U$.**

Compute $\mathbf{u}_i = \frac{1}{\sigma_i} A \mathbf{v}_i$ for each singular vector.

The full numerical result gives $A = U\Sigma V^T$ as desired.

### 6.6 Key Properties of SVD

| Property | Statement |
|----------|-----------|
| Existence | Every matrix has an SVD (unlike eigendecomposition) |
| Rank | $\text{rank}(A) =$ number of nonzero singular values |
| Norm | $\|A\|_2 = \sigma_1$ (largest singular value) |
| Frobenius norm | $\|A\|_F = \sqrt{\sigma_1^2 + \sigma_2^2 + \cdots + \sigma_r^2}$ |
| Condition number | $\kappa(A) = \sigma_1 / \sigma_n$ (ratio of largest to smallest) |

---

## Part 7: Matrix Approximation

### 7.1 Truncated SVD

Given the full SVD $A = U\Sigma V^T$ with $r$ nonzero singular values, we can approximate $A$ by keeping only the $k$ largest singular values (where $k < r$):

$$A_k = U_k \Sigma_k V_k^T = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

where:
- $U_k$ is the first $k$ columns of $U$
- $\Sigma_k$ is the top-left $k \times k$ block of $\Sigma$
- $V_k$ is the first $k$ columns of $V$

In plain English: keep the $k$ most important "layers" of the matrix and throw away the rest. Each layer is a rank-1 matrix $\sigma_i \mathbf{u}_i \mathbf{v}_i^T$ weighted by its singular value.

### 7.2 The Eckart-Young Theorem

The truncated SVD gives the **best** rank-$k$ approximation to $A$:

$$A_k = \arg\min_{\text{rank}(B) \leq k} \|A - B\|_F$$

In other words, among all matrices with rank at most $k$, $A_k$ is the closest to $A$ in the Frobenius norm. The approximation error is:

$$\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_r^2}$$

This is a remarkable result: the best low-rank approximation is obtained simply by discarding the smallest singular values.

### 7.3 Worked Example: Rank-1 Approximation

**Find the best rank-1 approximation to:**

$$A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$$

**Step 1: Compute the SVD.**

$$A^T A = \begin{bmatrix} 10 & 6 \\ 6 & 10 \end{bmatrix}$$

Eigenvalues of $A^T A$: $\lambda_1 = 16$, $\lambda_2 = 4$.

Singular values: $\sigma_1 = 4$, $\sigma_2 = 2$.

Eigenvectors of $A^T A$:

For $\lambda_1 = 16$: $\mathbf{v}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix}$

For $\lambda_2 = 4$: $\mathbf{v}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix}$

Left singular vectors: $\mathbf{u}_i = \frac{1}{\sigma_i}A\mathbf{v}_i$

$$\mathbf{u}_1 = \frac{1}{4} \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{1}{4\sqrt{2}} \begin{bmatrix} 4 \\ 4 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

$$\mathbf{u}_2 = \frac{1}{2} \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ -1 \end{bmatrix} = \frac{1}{2\sqrt{2}} \begin{bmatrix} 2 \\ -2 \end{bmatrix} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}$$

**Step 2: Compute the rank-1 approximation.**

$$A_1 = \sigma_1 \mathbf{u}_1 \mathbf{v}_1^T = 4 \cdot \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \end{bmatrix} \cdot \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \end{bmatrix} = 4 \cdot \frac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix}$$

**Step 3: Check the error.**

$$\|A - A_1\|_F = \sigma_2 = 2$$

$$A - A_1 = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix} - \begin{bmatrix} 2 & 2 \\ 2 & 2 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$$

$$\|A - A_1\|_F = \sqrt{1 + 1 + 1 + 1} = \sqrt{4} = 2 \quad \checkmark$$

### 7.4 Applications of Low-Rank Approximation

| Application | How Truncated SVD Helps |
|-------------|------------------------|
| **Image compression** | An $m \times n$ image stored as a matrix can be approximated with rank $k$, requiring only $k(m + n + 1)$ numbers instead of $mn$ |
| **Dimensionality reduction (PCA)** | Principal Component Analysis keeps the top $k$ singular vectors to reduce feature dimensions |
| **Recommender systems** | User-item rating matrices are approximated at low rank to predict missing ratings |
| **Noise reduction** | Small singular values often correspond to noise; discarding them denoises the data |
| **Latent semantic analysis** | In NLP, document-term matrices are approximated to find semantic structure |

**Image compression example:** Suppose you have a $1000 \times 1000$ grayscale image (1,000,000 pixel values). A rank-50 SVD approximation stores only $50 \times (1000 + 1000 + 1) = 100{,}050$ numbers, which is about 10% of the original, yet often captures the essential visual content.

---

## Summary: Key Takeaways

### Matrix Scalars
- **Determinant** $\det(A)$: tells you invertibility and volume scaling
- **Trace** $\text{tr}(A)$: sum of diagonal entries = sum of eigenvalues

### Eigenvalues and Eigenvectors
- $A\mathbf{v} = \lambda \mathbf{v}$: eigenvectors are special directions, eigenvalues are scale factors
- Found by solving $\det(A - \lambda I) = 0$

### Matrix Decompositions
- **Cholesky:** $A = LL^T$ for symmetric positive definite matrices
- **Eigendecomposition:** $A = PDP^{-1}$ for diagonalizable square matrices
- **SVD:** $A = U\Sigma V^T$ for any matrix (the universal decomposition)

### Matrix Approximation
- Truncated SVD gives the best rank-$k$ approximation (Eckart-Young theorem)
- Central to PCA, image compression, recommender systems, and denoising

---

## Practice Problems

### Problem 1
Compute the determinant:

$$A = \begin{bmatrix} 1 & 3 & 2 \\ 4 & 1 & 3 \\ 2 & 5 & 2 \end{bmatrix}$$

### Problem 2
Let $A = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}$ and $B = \begin{bmatrix} 0 & 4 \\ 1 & 2 \end{bmatrix}$.

Verify that $\text{tr}(AB) = \text{tr}(BA)$.

### Problem 3
Find the eigenvalues and eigenvectors of:

$$C = \begin{bmatrix} 5 & 4 \\ 2 & 3 \end{bmatrix}$$

### Problem 4
Find the Cholesky decomposition of:

$$A = \begin{bmatrix} 9 & 6 \\ 6 & 8 \end{bmatrix}$$

### Problem 5
Diagonalize the matrix from Problem 3. That is, find $P$ and $D$ such that $C = PDP^{-1}$.

### Problem 6
The singular values of a $3 \times 3$ matrix $M$ are $\sigma_1 = 10$, $\sigma_2 = 5$, $\sigma_3 = 1$.

(a) What is $\|M\|_F$?

(b) What is the Frobenius-norm error of the best rank-2 approximation $M_2$?

(c) What percentage of $\|M\|_F^2$ is captured by $M_2$?

---

## Solutions

**Solution 1:**

Using cofactor expansion along the first row:

$$\det(A) = 1 \begin{vmatrix} 1 & 3 \\ 5 & 2 \end{vmatrix} - 3 \begin{vmatrix} 4 & 3 \\ 2 & 2 \end{vmatrix} + 2 \begin{vmatrix} 4 & 1 \\ 2 & 5 \end{vmatrix}$$

$$= 1(1 \cdot 2 - 3 \cdot 5) - 3(4 \cdot 2 - 3 \cdot 2) + 2(4 \cdot 5 - 1 \cdot 2)$$

$$= 1(2 - 15) - 3(8 - 6) + 2(20 - 2)$$

$$= 1(-13) - 3(2) + 2(18)$$

$$= -13 - 6 + 36 = 17$$

---

**Solution 2:**

$$AB = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}\begin{bmatrix} 0 & 4 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 2(0)+1(1) & 2(4)+1(2) \\ 1(0)+3(1) & 1(4)+3(2) \end{bmatrix} = \begin{bmatrix} 1 & 10 \\ 3 & 10 \end{bmatrix}$$

$$\text{tr}(AB) = 1 + 10 = 11$$

$$BA = \begin{bmatrix} 0 & 4 \\ 1 & 2 \end{bmatrix}\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix} = \begin{bmatrix} 0(2)+4(1) & 0(1)+4(3) \\ 1(2)+2(1) & 1(1)+2(3) \end{bmatrix} = \begin{bmatrix} 4 & 12 \\ 4 & 7 \end{bmatrix}$$

$$\text{tr}(BA) = 4 + 7 = 11$$

Both traces equal 11, confirming $\text{tr}(AB) = \text{tr}(BA)$.

---

**Solution 3:**

Characteristic polynomial:

$$\det(C - \lambda I) = (5 - \lambda)(3 - \lambda) - (4)(2) = \lambda^2 - 8\lambda + 15 - 8 = \lambda^2 - 8\lambda + 7$$

$$(\lambda - 7)(\lambda - 1) = 0$$

$$\lambda_1 = 7, \quad \lambda_2 = 1$$

**Eigenvector for $\lambda_1 = 7$:**

$$(C - 7I)\mathbf{v} = \begin{bmatrix} -2 & 4 \\ 2 & -4 \end{bmatrix}\mathbf{v} = \mathbf{0}$$

From the first row: $-2v_1 + 4v_2 = 0$, so $v_1 = 2v_2$. Choosing $v_2 = 1$:

$$\mathbf{v}_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

**Eigenvector for $\lambda_2 = 1$:**

$$(C - I)\mathbf{v} = \begin{bmatrix} 4 & 4 \\ 2 & 2 \end{bmatrix}\mathbf{v} = \mathbf{0}$$

From the first row: $4v_1 + 4v_2 = 0$, so $v_1 = -v_2$. Choosing $v_2 = 1$:

$$\mathbf{v}_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}$$

**Check:** $\text{tr}(C) = 5 + 3 = 8 = 7 + 1$ and $\det(C) = 15 - 8 = 7 = 7 \times 1$. Both pass.

---

**Solution 4:**

$$l_{11} = \sqrt{9} = 3$$

$$l_{21} = \frac{6}{3} = 2$$

$$l_{22} = \sqrt{8 - 2^2} = \sqrt{4} = 2$$

$$L = \begin{bmatrix} 3 & 0 \\ 2 & 2 \end{bmatrix}$$

**Verification:**

$$LL^T = \begin{bmatrix} 3 & 0 \\ 2 & 2 \end{bmatrix}\begin{bmatrix} 3 & 2 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 9 & 6 \\ 6 & 8 \end{bmatrix} = A \quad \checkmark$$

---

**Solution 5:**

From Solution 3: $\lambda_1 = 7$, $\mathbf{v}_1 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$, $\lambda_2 = 1$, $\mathbf{v}_2 = \begin{bmatrix} -1 \\ 1 \end{bmatrix}$.

$$P = \begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix}, \quad D = \begin{bmatrix} 7 & 0 \\ 0 & 1 \end{bmatrix}$$

$$P^{-1} = \frac{1}{2(1) - (-1)(1)} \begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix} = \frac{1}{3}\begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix}$$

**Verification:**

$$PDP^{-1} = \begin{bmatrix} 2 & -1 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} 7 & 0 \\ 0 & 1 \end{bmatrix} \cdot \frac{1}{3}\begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix}$$

$$= \begin{bmatrix} 14 & -1 \\ 7 & 1 \end{bmatrix} \cdot \frac{1}{3}\begin{bmatrix} 1 & 1 \\ -1 & 2 \end{bmatrix}$$

$$= \frac{1}{3}\begin{bmatrix} 14(1)+(-1)(-1) & 14(1)+(-1)(2) \\ 7(1)+1(-1) & 7(1)+1(2) \end{bmatrix} = \frac{1}{3}\begin{bmatrix} 15 & 12 \\ 6 & 9 \end{bmatrix} = \begin{bmatrix} 5 & 4 \\ 2 & 3 \end{bmatrix} = C \quad \checkmark$$

---

**Solution 6:**

**(a)** The Frobenius norm is:

$$\|M\|_F = \sqrt{\sigma_1^2 + \sigma_2^2 + \sigma_3^2} = \sqrt{100 + 25 + 1} = \sqrt{126} = 3\sqrt{14} \approx 11.22$$

**(b)** The best rank-2 approximation discards $\sigma_3 = 1$. The error is:

$$\|M - M_2\|_F = \sqrt{\sigma_3^2} = \sigma_3 = 1$$

**(c)** The squared Frobenius norm captured by $M_2$ is:

$$\frac{\sigma_1^2 + \sigma_2^2}{\sigma_1^2 + \sigma_2^2 + \sigma_3^2} = \frac{100 + 25}{100 + 25 + 1} = \frac{125}{126} \approx 99.2\%$$

So the rank-2 approximation captures about 99.2% of the total "energy" of $M$.

---

**Course:** Mathematics for Machine Learning
**Instructor:** Mohammed Alnemari

**Next:** Tutorial 4 - Matrix Calculus and Optimization
