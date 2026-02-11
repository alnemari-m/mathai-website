# Tutorial 2: Analytic Geometry

**Course:** Mathematics for Machine Learning
**Instructor:** Mohammed Alnemari

---

## 📚 Learning Objectives

By the end of this tutorial, you will understand:

1. Norms and their role in measuring vector magnitude
2. Inner products and their defining axioms
3. How lengths, distances, angles, and orthogonality arise from inner products
4. Orthogonal matrices, orthonormal bases, and orthogonal complements
5. Orthogonal projections and the Gram-Schmidt process
6. Rotation matrices and their geometric meaning

---

## Part 1: Norms

### 1.1 What is a Norm?

A **norm** is a function $\|\cdot\| : \mathbb{R}^n \to \mathbb{R}$ that assigns a non-negative "length" to every vector.

> **Think of it as...** a ruler for vectors. Different norms are like different ways of measuring distance — walking along city blocks versus flying in a straight line.

A norm must satisfy these properties for all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ and all $\lambda \in \mathbb{R}$:

| Property | Statement | Intuition |
|----------|-----------|-----------|
| Non-negativity | $\|\mathbf{x}\| \geq 0$ | Lengths are never negative |
| Definiteness | $\|\mathbf{x}\| = 0 \iff \mathbf{x} = \mathbf{0}$ | Only the zero vector has zero length |
| Absolute homogeneity | $\|\lambda \mathbf{x}\| = |\lambda| \|\mathbf{x}\|$ | Scaling a vector scales its length |
| Triangle inequality | $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$ | The shortcut is never longer than going around |

### 1.2 Common Norms

| Norm | Name | Formula | Also Called |
|------|------|---------|-------------|
| $\ell_1$ | Manhattan norm | $\|\mathbf{x}\|_1 = \displaystyle\sum_{i=1}^{n} \|x_i\|$ | Taxicab norm |
| $\ell_2$ | Euclidean norm | $\|\mathbf{x}\|_2 = \sqrt{\displaystyle\sum_{i=1}^{n} x_i^2}$ | Standard norm |
| $\ell_\infty$ | Max norm | $\|\mathbf{x}\|_\infty = \max_{i} |x_i|$ | Chebyshev norm |

### 1.3 Worked Example: Computing Norms

Let $\mathbf{x} = \begin{bmatrix} 3 \\ -4 \\ 2 \end{bmatrix}$.

**$\ell_1$ norm:**
$$\|\mathbf{x}\|_1 = |3| + |-4| + |2| = 3 + 4 + 2 = 9$$

**$\ell_2$ norm:**
$$\|\mathbf{x}\|_2 = \sqrt{3^2 + (-4)^2 + 2^2} = \sqrt{9 + 16 + 4} = \sqrt{29} \approx 5.39$$

**$\ell_\infty$ norm:**
$$\|\mathbf{x}\|_\infty = \max\{|3|, |-4|, |2|\} = 4$$

> **Think of it as...** The $\ell_1$ norm counts total blocks walked in a grid city. The $\ell_2$ norm is the straight-line (as the crow flies) distance. The $\ell_\infty$ norm is the longest single step you take along any one axis.

---

## Part 2: Inner Products

### 2.1 Definition

An **inner product** on a vector space $V$ is a function $\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$ that satisfies four axioms:

| Axiom | Statement | For all |
|-------|-----------|---------|
| Symmetry | $\langle \mathbf{x}, \mathbf{y} \rangle = \langle \mathbf{y}, \mathbf{x} \rangle$ | $\mathbf{x}, \mathbf{y} \in V$ |
| Linearity in 1st argument | $\langle \lambda\mathbf{x} + \mathbf{z}, \mathbf{y} \rangle = \lambda\langle \mathbf{x}, \mathbf{y} \rangle + \langle \mathbf{z}, \mathbf{y} \rangle$ | $\mathbf{x}, \mathbf{y}, \mathbf{z} \in V,\ \lambda \in \mathbb{R}$ |
| Positive semi-definiteness | $\langle \mathbf{x}, \mathbf{x} \rangle \geq 0$ | $\mathbf{x} \in V$ |
| Positive definiteness | $\langle \mathbf{x}, \mathbf{x} \rangle = 0 \iff \mathbf{x} = \mathbf{0}$ | $\mathbf{x} \in V$ |

> **Think of it as...** an inner product is a generalized way of multiplying two vectors together to get a single number that tells you "how much" the vectors agree in direction.

### 2.2 The Dot Product

The most common inner product in $\mathbb{R}^n$ is the **dot product**:

$$\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^T \mathbf{y} = \sum_{i=1}^{n} x_i y_i$$

**Example:**
$$\left\langle \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \begin{bmatrix} 4 \\ 0 \\ -1 \end{bmatrix} \right\rangle = 1(4) + 2(0) + 3(-1) = 4 + 0 - 3 = 1$$

### 2.3 General Inner Products and Positive Definite Matrices

Not every inner product is the dot product. We can define a more general inner product using a **symmetric positive definite matrix** $A$:

$$\langle \mathbf{x}, \mathbf{y} \rangle_A = \mathbf{x}^T A \mathbf{y}$$

A symmetric matrix $A$ is **positive definite** if:
$$\mathbf{x}^T A \mathbf{x} > 0 \quad \text{for all } \mathbf{x} \neq \mathbf{0}$$

**Example:** Let $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ and $\mathbf{x} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

$$\mathbf{x}^T A \mathbf{x} = \begin{bmatrix} 1 & 1 \end{bmatrix} \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 3 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 6 > 0$$

> **Think of it as...** the standard dot product uses the identity matrix $I$ as $A$. Choosing a different positive definite $A$ stretches or skews the geometry, like measuring distance on a tilted surface instead of a flat table.

---

## Part 3: Lengths and Distances

### 3.1 Induced Norm

Every inner product **induces** a norm:

$$\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$$

For the standard dot product this gives the Euclidean norm:

$$\|\mathbf{x}\| = \sqrt{\mathbf{x}^T \mathbf{x}} = \sqrt{\sum_{i=1}^{n} x_i^2}$$

### 3.2 Distance

The **distance** between two vectors $\mathbf{x}$ and $\mathbf{y}$ is:

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\| = \sqrt{\langle \mathbf{x} - \mathbf{y}, \mathbf{x} - \mathbf{y} \rangle}$$

A distance function (metric) satisfies:

| Property | Statement |
|----------|-----------|
| Non-negativity | $d(\mathbf{x}, \mathbf{y}) \geq 0$ |
| Identity | $d(\mathbf{x}, \mathbf{y}) = 0 \iff \mathbf{x} = \mathbf{y}$ |
| Symmetry | $d(\mathbf{x}, \mathbf{y}) = d(\mathbf{y}, \mathbf{x})$ |
| Triangle inequality | $d(\mathbf{x}, \mathbf{z}) \leq d(\mathbf{x}, \mathbf{y}) + d(\mathbf{y}, \mathbf{z})$ |

### 3.3 Cauchy-Schwarz Inequality

One of the most important inequalities in all of mathematics:

$$|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\| \cdot \|\mathbf{y}\|$$

Equality holds if and only if $\mathbf{x}$ and $\mathbf{y}$ are linearly dependent (i.e., one is a scalar multiple of the other).

> **Think of it as...** the dot product can never exceed the product of the lengths. This is what guarantees that the cosine of the angle between two vectors always stays between $-1$ and $1$.

**Example:** Let $\mathbf{x} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ and $\mathbf{y} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$.

- $|\langle \mathbf{x}, \mathbf{y} \rangle| = |1(3) + 2(1)| = |5| = 5$
- $\|\mathbf{x}\| \cdot \|\mathbf{y}\| = \sqrt{1+4}\,\sqrt{9+1} = \sqrt{5}\,\sqrt{10} = \sqrt{50} \approx 7.07$
- Check: $5 \leq 7.07$ ✓

---

## Part 4: Angles and Orthogonality

### 4.1 Angle Between Vectors

The **angle** $\theta$ between two non-zero vectors $\mathbf{x}$ and $\mathbf{y}$ is defined via:

$$\cos \theta = \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\| \cdot \|\mathbf{y}\|}$$

The Cauchy-Schwarz inequality guarantees that the right-hand side lies in $[-1, 1]$, so $\theta$ is well-defined.

**Example:** Find the angle between $\mathbf{x} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\mathbf{y} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

$$\cos \theta = \frac{1(1) + 0(1)}{\sqrt{1}\,\sqrt{2}} = \frac{1}{\sqrt{2}} \implies \theta = \frac{\pi}{4} = 45^\circ$$

### 4.2 Orthogonality

Two vectors are **orthogonal** (perpendicular) if their inner product is zero:

$$\mathbf{x} \perp \mathbf{y} \iff \langle \mathbf{x}, \mathbf{y} \rangle = 0$$

> **Think of it as...** orthogonal vectors carry completely independent information — knowing one tells you nothing about the other. This is exactly the idea behind "uncorrelated features" in machine learning.

**Example:**
$$\left\langle \begin{bmatrix} 1 \\ -1 \end{bmatrix}, \begin{bmatrix} 1 \\ 1 \end{bmatrix} \right\rangle = 1(1) + (-1)(1) = 0 \quad \checkmark \text{ Orthogonal!}$$

### 4.3 Orthogonal and Orthonormal Sets

A set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ is:

| Term | Condition |
|------|-----------|
| **Orthogonal** | $\langle \mathbf{v}_i, \mathbf{v}_j \rangle = 0$ for all $i \neq j$ |
| **Orthonormal** | Orthogonal **and** $\|\mathbf{v}_i\| = 1$ for all $i$ |

**Example of an orthonormal set in $\mathbb{R}^2$:**
$$\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

- $\langle \mathbf{e}_1, \mathbf{e}_2 \rangle = 0$ (orthogonal)
- $\|\mathbf{e}_1\| = 1$ and $\|\mathbf{e}_2\| = 1$ (unit length)

---

## Part 5: Orthogonal Matrices

### 5.1 Definition

A square matrix $A \in \mathbb{R}^{n \times n}$ is **orthogonal** if its columns form an orthonormal set. Equivalently:

$$A^T A = I \implies A^{-1} = A^T$$

> **Think of it as...** an orthogonal matrix performs a "rigid" transformation — it can rotate or reflect vectors but never stretches or squishes them.

### 5.2 Key Properties

| Property | Statement |
|----------|-----------|
| Inverse equals transpose | $A^{-1} = A^T$ |
| Columns are orthonormal | $\langle \mathbf{a}_i, \mathbf{a}_j \rangle = \delta_{ij}$ |
| Rows are orthonormal | $A A^T = I$ |
| Preserves lengths | $\|A\mathbf{x}\| = \|\mathbf{x}\|$ |
| Preserves angles | $\langle A\mathbf{x}, A\mathbf{y} \rangle = \langle \mathbf{x}, \mathbf{y} \rangle$ |
| Determinant | $\det(A) = \pm 1$ |
| Product is orthogonal | If $A, B$ orthogonal, then $AB$ is orthogonal |

**Proof that orthogonal matrices preserve lengths:**
$$\|A\mathbf{x}\|^2 = (A\mathbf{x})^T(A\mathbf{x}) = \mathbf{x}^T A^T A \mathbf{x} = \mathbf{x}^T I \mathbf{x} = \mathbf{x}^T \mathbf{x} = \|\mathbf{x}\|^2$$

### 5.3 Example

$$A = \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\[4pt] \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix}$$

**Verify $A^T A = I$:**
$$A^T A = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\[4pt] -\frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix} \begin{bmatrix} \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \\[4pt] \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I \quad \checkmark$$

---

## Part 6: Orthonormal Basis

### 6.1 Definition

An **orthonormal basis** (ONB) for a subspace $U \subseteq \mathbb{R}^n$ is a basis $\{\mathbf{u}_1, \ldots, \mathbf{u}_k\}$ such that:

$$\langle \mathbf{u}_i, \mathbf{u}_j \rangle = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

### 6.2 Why Orthonormal Bases are Useful

With an orthonormal basis, finding coordinates becomes trivially easy. If $\{\mathbf{u}_1, \ldots, \mathbf{u}_k\}$ is an ONB for $U$ and $\mathbf{x} \in U$, then:

$$\mathbf{x} = \sum_{i=1}^{k} \langle \mathbf{x}, \mathbf{u}_i \rangle \, \mathbf{u}_i$$

> **Think of it as...** with an orthonormal basis, you find each coordinate by simply taking a dot product — no system of equations to solve. It is the easiest possible coordinate system.

### 6.3 How to Find an Orthonormal Basis

Given any basis, use the **Gram-Schmidt process** (covered in Part 9) to convert it into an orthonormal basis.

---

## Part 7: Orthogonal Complement

### 7.1 Definition

Let $U$ be a subspace of $\mathbb{R}^n$. The **orthogonal complement** $U^\perp$ is the set of all vectors orthogonal to every vector in $U$:

$$U^\perp = \{\mathbf{v} \in \mathbb{R}^n : \langle \mathbf{v}, \mathbf{u} \rangle = 0 \text{ for all } \mathbf{u} \in U\}$$

> **Think of it as...** if $U$ is a plane through the origin in 3D, then $U^\perp$ is the line perpendicular to that plane. Together they account for all of $\mathbb{R}^3$.

### 7.2 Key Properties

| Property | Statement |
|----------|-----------|
| Subspace | $U^\perp$ is itself a subspace |
| Dimension | $\dim(U) + \dim(U^\perp) = n$ |
| Double complement | $(U^\perp)^\perp = U$ |
| Direct sum | $\mathbb{R}^n = U \oplus U^\perp$ (every vector splits uniquely) |

### 7.3 Connection to the Kernel and Row Space

For a matrix $A \in \mathbb{R}^{m \times n}$:

$$\ker(A) = \text{row}(A)^\perp$$

This means: a vector $\mathbf{x}$ is in the null space of $A$ if and only if $\mathbf{x}$ is orthogonal to every row of $A$.

**Example:** Let $A = \begin{bmatrix} 1 & 2 \\ 3 & 6 \end{bmatrix}$.

The row space is $\text{span}\left\{\begin{bmatrix} 1 \\ 2 \end{bmatrix}\right\}$ (the rows are linearly dependent).

The null space is $\ker(A) = \text{span}\left\{\begin{bmatrix} -2 \\ 1 \end{bmatrix}\right\}$.

**Check:** $\left\langle \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \begin{bmatrix} -2 \\ 1 \end{bmatrix} \right\rangle = 1(-2) + 2(1) = 0$ ✓

---

## Part 8: Orthogonal Projections

### 8.1 Projection onto a Line

Given a non-zero vector $\mathbf{b}$ (defining a line through the origin), the **projection** of $\mathbf{x}$ onto the line spanned by $\mathbf{b}$ is:

$$\pi_{\mathbf{b}}(\mathbf{x}) = \frac{\langle \mathbf{x}, \mathbf{b} \rangle}{\langle \mathbf{b}, \mathbf{b} \rangle} \mathbf{b} = \frac{\mathbf{b}\mathbf{b}^T}{\mathbf{b}^T\mathbf{b}} \mathbf{x}$$

The **projection matrix** is:

$$P_\pi = \frac{\mathbf{b}\mathbf{b}^T}{\mathbf{b}^T\mathbf{b}}$$

> **Think of it as...** shining a flashlight straight down onto a line and seeing where the shadow of your vector lands. The projection is that shadow.

**Example:** Project $\mathbf{x} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$ onto $\mathbf{b} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$.

$$\pi_{\mathbf{b}}(\mathbf{x}) = \frac{\mathbf{x}^T\mathbf{b}}{\mathbf{b}^T\mathbf{b}} \mathbf{b} = \frac{3(1) + 1(2)}{1^2 + 2^2} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \frac{5}{5} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$

### 8.2 Projection onto a General Subspace

Let $U = \text{span}\{\mathbf{b}_1, \ldots, \mathbf{b}_k\}$ and define $B = [\mathbf{b}_1 \mid \cdots \mid \mathbf{b}_k]$. The projection of $\mathbf{x}$ onto $U$ is:

$$\pi_U(\mathbf{x}) = B(B^T B)^{-1} B^T \mathbf{x}$$

The **projection matrix** is:

$$P = B(B^T B)^{-1} B^T$$

**Properties of projection matrices:**

| Property | Statement |
|----------|-----------|
| Idempotent | $P^2 = P$ (projecting twice is the same as projecting once) |
| Symmetric | $P^T = P$ |
| Residual | $\mathbf{x} - P\mathbf{x}$ is orthogonal to $U$ |

### 8.3 Connection to the Pseudo-Inverse

The **Moore-Penrose pseudo-inverse** of $B$ is:

$$B^\dagger = (B^T B)^{-1} B^T$$

So the projection simplifies to:

$$\pi_U(\mathbf{x}) = B B^\dagger \mathbf{x}$$

The pseudo-inverse is central to solving **least-squares** problems: when $A\mathbf{x} = \mathbf{b}$ has no exact solution, the best approximate solution is $\hat{\mathbf{x}} = A^\dagger \mathbf{b}$.

### 8.4 Worked Example: Projection onto a Subspace

Project $\mathbf{x} = \begin{bmatrix} 6 \\ 0 \\ 0 \end{bmatrix}$ onto $U = \text{span}\left\{\begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}\right\}$.

**Step 1:** Form the matrix $B$:
$$B = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}$$

**Step 2:** Compute $B^T B$:
$$B^T B = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

**Step 3:** Compute $(B^T B)^{-1}$:
$$(B^T B)^{-1} = \frac{1}{2(2) - 1(1)} \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix} = \frac{1}{3} \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}$$

**Step 4:** Compute $B^T \mathbf{x}$:
$$B^T \mathbf{x} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} \begin{bmatrix} 6 \\ 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 6 \\ 0 \end{bmatrix}$$

**Step 5:** Compute $(B^T B)^{-1} B^T \mathbf{x}$:
$$\frac{1}{3}\begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}\begin{bmatrix} 6 \\ 0 \end{bmatrix} = \frac{1}{3}\begin{bmatrix} 12 \\ -6 \end{bmatrix} = \begin{bmatrix} 4 \\ -2 \end{bmatrix}$$

**Step 6:** Compute the projection:
$$\pi_U(\mathbf{x}) = B \begin{bmatrix} 4 \\ -2 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}\begin{bmatrix} 4 \\ -2 \end{bmatrix} = \begin{bmatrix} 4 \\ -2 \\ 2 \end{bmatrix}$$

**Verify:** The residual $\mathbf{x} - \pi_U(\mathbf{x}) = \begin{bmatrix} 2 \\ 2 \\ -2 \end{bmatrix}$ should be orthogonal to both basis vectors:

- $\langle \begin{bmatrix} 2 \\ 2 \\ -2 \end{bmatrix}, \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} \rangle = 2 + 0 - 2 = 0$ ✓
- $\langle \begin{bmatrix} 2 \\ 2 \\ -2 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix} \rangle = 0 + 2 - 2 = 0$ ✓

---

## Part 9: Gram-Schmidt Process

### 9.1 The Algorithm

The **Gram-Schmidt process** takes any set of linearly independent vectors and produces an orthonormal set spanning the same subspace.

Given linearly independent vectors $\{\mathbf{b}_1, \mathbf{b}_2, \ldots, \mathbf{b}_k\}$:

**Step 1: Orthogonalize** (produce orthogonal vectors $\mathbf{u}_i$)

$$\mathbf{u}_1 = \mathbf{b}_1$$

$$\mathbf{u}_2 = \mathbf{b}_2 - \frac{\langle \mathbf{b}_2, \mathbf{u}_1 \rangle}{\langle \mathbf{u}_1, \mathbf{u}_1 \rangle} \mathbf{u}_1$$

$$\mathbf{u}_3 = \mathbf{b}_3 - \frac{\langle \mathbf{b}_3, \mathbf{u}_1 \rangle}{\langle \mathbf{u}_1, \mathbf{u}_1 \rangle} \mathbf{u}_1 - \frac{\langle \mathbf{b}_3, \mathbf{u}_2 \rangle}{\langle \mathbf{u}_2, \mathbf{u}_2 \rangle} \mathbf{u}_2$$

In general:
$$\mathbf{u}_i = \mathbf{b}_i - \sum_{j=1}^{i-1} \frac{\langle \mathbf{b}_i, \mathbf{u}_j \rangle}{\langle \mathbf{u}_j, \mathbf{u}_j \rangle} \mathbf{u}_j$$

**Step 2: Normalize** (produce unit vectors $\mathbf{e}_i$)

$$\mathbf{e}_i = \frac{\mathbf{u}_i}{\|\mathbf{u}_i\|}$$

> **Think of it as...** taking each new vector and "subtracting off" all the parts that point in the directions you have already handled. What remains is the genuinely new direction. Then you scale it to length 1.

### 9.2 Worked Example

Apply Gram-Schmidt to $\mathbf{b}_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}$ and $\mathbf{b}_2 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$.

**Step 1:** Set $\mathbf{u}_1 = \mathbf{b}_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}$.

**Step 2:** Compute the projection coefficient:
$$\frac{\langle \mathbf{b}_2, \mathbf{u}_1 \rangle}{\langle \mathbf{u}_1, \mathbf{u}_1 \rangle} = \frac{1(1) + 0(1) + 1(0)}{1^2 + 1^2 + 0^2} = \frac{1}{2}$$

**Step 3:** Subtract the projection:
$$\mathbf{u}_2 = \mathbf{b}_2 - \frac{1}{2}\mathbf{u}_1 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} - \frac{1}{2}\begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 1/2 \\ -1/2 \\ 1 \end{bmatrix}$$

**Verify orthogonality:**
$$\langle \mathbf{u}_1, \mathbf{u}_2 \rangle = 1\!\left(\tfrac{1}{2}\right) + 1\!\left(-\tfrac{1}{2}\right) + 0(1) = 0 \quad \checkmark$$

**Step 4:** Normalize:
$$\|\mathbf{u}_1\| = \sqrt{1+1+0} = \sqrt{2}, \quad \mathbf{e}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}$$

$$\|\mathbf{u}_2\| = \sqrt{\tfrac{1}{4}+\tfrac{1}{4}+1} = \sqrt{\tfrac{3}{2}} = \frac{\sqrt{6}}{2}, \quad \mathbf{e}_2 = \frac{2}{\sqrt{6}}\begin{bmatrix} 1/2 \\ -1/2 \\ 1 \end{bmatrix} = \frac{1}{\sqrt{6}}\begin{bmatrix} 1 \\ -1 \\ 2 \end{bmatrix}$$

**Result:** The orthonormal basis is:
$$\mathbf{e}_1 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \qquad \mathbf{e}_2 = \frac{1}{\sqrt{6}}\begin{bmatrix} 1 \\ -1 \\ 2 \end{bmatrix}$$

---

## Part 10: Rotations

### 10.1 Rotation Matrix in 2D

A **rotation** by angle $\theta$ (counter-clockwise) in $\mathbb{R}^2$ is given by:

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

> **Think of it as...** every point in the plane is swung around the origin by the angle $\theta$. The matrix encodes where the two standard basis vectors land after the rotation.

### 10.2 Properties of Rotation Matrices

| Property | Statement |
|----------|-----------|
| Orthogonal | $R(\theta)^T R(\theta) = I$ |
| Determinant | $\det(R(\theta)) = 1$ (no reflection) |
| Inverse is reverse rotation | $R(\theta)^{-1} = R(-\theta) = R(\theta)^T$ |
| Composition | $R(\alpha) R(\beta) = R(\alpha + \beta)$ |
| Preserves lengths | $\|R(\theta)\mathbf{x}\| = \|\mathbf{x}\|$ |
| Preserves angles | Angles between vectors are unchanged |

### 10.3 Worked Example

Rotate $\mathbf{x} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ by $\theta = 90^\circ$.

$$R(90^\circ) = \begin{bmatrix} \cos 90^\circ & -\sin 90^\circ \\ \sin 90^\circ & \cos 90^\circ \end{bmatrix} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$$

$$R(90^\circ)\mathbf{x} = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

This is exactly the unit vector pointing straight up — a $90^\circ$ counter-clockwise rotation of the unit vector pointing right. ✓

### 10.4 Rotations in 3D (Preview)

In $\mathbb{R}^3$, a rotation about the $z$-axis by angle $\theta$ is:

$$R_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

General 3D rotations can be composed from rotations about the three coordinate axes.

---

## Summary: Key Takeaways

### Norms and Inner Products
- The $\ell_1$, $\ell_2$, and $\ell_\infty$ norms each measure vector size differently
- An inner product $\langle \cdot, \cdot \rangle$ must satisfy symmetry, linearity, and positive definiteness
- Every inner product induces a norm: $\|\mathbf{x}\| = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle}$

### Geometry from Inner Products
- Angles: $\cos\theta = \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\|\|\mathbf{y}\|}$
- Orthogonality: $\langle \mathbf{x}, \mathbf{y} \rangle = 0$
- Cauchy-Schwarz: $|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\|\|\mathbf{y}\|$

### Orthogonal Structures
- Orthogonal matrices satisfy $A^{-1} = A^T$ and preserve geometry
- Orthogonal complements: $\ker(A) = \text{row}(A)^\perp$
- Gram-Schmidt converts any basis to an orthonormal basis

### Projections and Rotations
- Projection onto a line: $P_\pi = \frac{\mathbf{b}\mathbf{b}^T}{\mathbf{b}^T\mathbf{b}}$
- Projection onto a subspace: $P = B(B^TB)^{-1}B^T$
- 2D rotation: $R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$

---

## Practice Problems

### Problem 1
Compute the $\ell_1$, $\ell_2$, and $\ell_\infty$ norms of:
$$\mathbf{x} = \begin{bmatrix} -2 \\ 6 \\ -3 \end{bmatrix}$$

### Problem 2
Let $\mathbf{a} = \begin{bmatrix} 2 \\ 1 \\ -1 \end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix} 1 \\ -2 \\ 3 \end{bmatrix}$. Compute the angle $\theta$ between them.

### Problem 3
Verify that $A = \begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix}$ is an orthogonal matrix and determine whether it represents a rotation or a reflection.

### Problem 4
Project $\mathbf{x} = \begin{bmatrix} 4 \\ 3 \end{bmatrix}$ onto the line spanned by $\mathbf{b} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$.

### Problem 5
Apply the Gram-Schmidt process to the vectors $\mathbf{b}_1 = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$ and $\mathbf{b}_2 = \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix}$ to produce an orthonormal basis.

### Problem 6
Let $\mathbf{x} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$. Find $R(\theta)\mathbf{x}$ for $\theta = 60^\circ$ and verify that the result has the same norm as $\mathbf{x}$.

---

## Solutions

**Solution 1:**

$$\|\mathbf{x}\|_1 = |-2| + |6| + |-3| = 2 + 6 + 3 = 11$$

$$\|\mathbf{x}\|_2 = \sqrt{(-2)^2 + 6^2 + (-3)^2} = \sqrt{4 + 36 + 9} = \sqrt{49} = 7$$

$$\|\mathbf{x}\|_\infty = \max\{|-2|, |6|, |-3|\} = 6$$

---

**Solution 2:**

First compute the dot product:
$$\langle \mathbf{a}, \mathbf{b} \rangle = 2(1) + 1(-2) + (-1)(3) = 2 - 2 - 3 = -3$$

Then compute the norms:
$$\|\mathbf{a}\| = \sqrt{4 + 1 + 1} = \sqrt{6}, \quad \|\mathbf{b}\| = \sqrt{1 + 4 + 9} = \sqrt{14}$$

Therefore:
$$\cos\theta = \frac{-3}{\sqrt{6}\sqrt{14}} = \frac{-3}{\sqrt{84}} = \frac{-3}{2\sqrt{21}}$$

$$\theta = \arccos\!\left(\frac{-3}{2\sqrt{21}}\right) \approx \arccos(-0.327) \approx 109.1^\circ$$

---

**Solution 3:**

**Check $A^T A = I$:**
$$A^T A = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}\begin{bmatrix} 0 & 1 \\ -1 & 0 \end{bmatrix} = \begin{bmatrix} 0(0)+(-1)(-1) & 0(1)+(-1)(0) \\ 1(0)+0(-1) & 1(1)+0(0) \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I \quad \checkmark$$

So $A$ is orthogonal.

**Determine rotation vs. reflection:**
$$\det(A) = 0(0) - (1)(-1) = 1$$

Since $\det(A) = +1$, this is a **rotation** (not a reflection). Specifically, this is a rotation by $-90^\circ$ (or equivalently $270^\circ$ counter-clockwise).

---

**Solution 4:**

$$\pi_{\mathbf{b}}(\mathbf{x}) = \frac{\mathbf{x}^T\mathbf{b}}{\mathbf{b}^T\mathbf{b}} \mathbf{b} = \frac{4(1) + 3(1)}{1^2 + 1^2}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \frac{7}{2}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 7/2 \\ 7/2 \end{bmatrix}$$

**Verify:** The residual $\mathbf{x} - \pi_{\mathbf{b}}(\mathbf{x}) = \begin{bmatrix} 4 - 7/2 \\ 3 - 7/2 \end{bmatrix} = \begin{bmatrix} 1/2 \\ -1/2 \end{bmatrix}$ should be orthogonal to $\mathbf{b}$:

$$\left\langle \begin{bmatrix} 1/2 \\ -1/2 \end{bmatrix}, \begin{bmatrix} 1 \\ 1 \end{bmatrix} \right\rangle = \frac{1}{2} - \frac{1}{2} = 0 \quad \checkmark$$

---

**Solution 5:**

**Step 1:** Set $\mathbf{u}_1 = \mathbf{b}_1 = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$.

**Step 2:** Compute the projection coefficient:
$$\frac{\langle \mathbf{b}_2, \mathbf{u}_1 \rangle}{\langle \mathbf{u}_1, \mathbf{u}_1 \rangle} = \frac{0(1) + 1(1) + 2(1)}{1+1+1} = \frac{3}{3} = 1$$

**Step 3:** Subtract the projection:
$$\mathbf{u}_2 = \mathbf{b}_2 - 1 \cdot \mathbf{u}_1 = \begin{bmatrix} 0 \\ 1 \\ 2 \end{bmatrix} - \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} -1 \\ 0 \\ 1 \end{bmatrix}$$

**Check orthogonality:** $\langle \mathbf{u}_1, \mathbf{u}_2 \rangle = -1 + 0 + 1 = 0$ ✓

**Step 4:** Normalize:
$$\mathbf{e}_1 = \frac{\mathbf{u}_1}{\|\mathbf{u}_1\|} = \frac{1}{\sqrt{3}}\begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$$

$$\mathbf{e}_2 = \frac{\mathbf{u}_2}{\|\mathbf{u}_2\|} = \frac{1}{\sqrt{2}}\begin{bmatrix} -1 \\ 0 \\ 1 \end{bmatrix}$$

**Orthonormal basis:** $\left\{\frac{1}{\sqrt{3}}\begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix},\ \frac{1}{\sqrt{2}}\begin{bmatrix} -1 \\ 0 \\ 1 \end{bmatrix}\right\}$

---

**Solution 6:**

$$R(60^\circ) = \begin{bmatrix} \cos 60^\circ & -\sin 60^\circ \\ \sin 60^\circ & \cos 60^\circ \end{bmatrix} = \begin{bmatrix} 1/2 & -\sqrt{3}/2 \\ \sqrt{3}/2 & 1/2 \end{bmatrix}$$

$$R(60^\circ)\mathbf{x} = \begin{bmatrix} 1/2 & -\sqrt{3}/2 \\ \sqrt{3}/2 & 1/2 \end{bmatrix}\begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 1/2 \\ \sqrt{3}/2 \end{bmatrix}$$

**Verify the norm is preserved:**

$$\|\mathbf{x}\| = \sqrt{1^2 + 0^2} = 1$$

$$\|R(60^\circ)\mathbf{x}\| = \sqrt{\left(\frac{1}{2}\right)^2 + \left(\frac{\sqrt{3}}{2}\right)^2} = \sqrt{\frac{1}{4} + \frac{3}{4}} = \sqrt{1} = 1 \quad \checkmark$$

The norm is preserved, confirming that $R(60^\circ)$ is an orthogonal (length-preserving) transformation.

---

**Course:** Mathematics for Machine Learning
**Instructor:** Mohammed Alnemari

**Next:** Tutorial 3 - Matrix Decompositions
