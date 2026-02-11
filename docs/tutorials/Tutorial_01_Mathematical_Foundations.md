# Tutorial 1: Mathematical Foundations and Terminology

**Course:** Mathematics for Machine Learning
**Instructor:** Mohammed Alnemari

---

## 📚 Learning Objectives

By the end of this tutorial, you will understand:

1. Essential mathematical notation and terminology
2. Basic set theory concepts
3. Fundamental number systems
4. Vector and matrix terminology
5. Key mathematical operations

---

## Part 1: Mathematical Notation

### 1.1 Basic Symbols

| Symbol | Meaning | Example |
|--------|---------|---------|
| $\in$ | "is an element of" / "belongs to" | $x \in \mathbb{R}$ means "x is a real number" |
| $\notin$ | "is not an element of" | $i \notin \mathbb{R}$ |
| $\subset$ | "is a subset of" | $\mathbb{N} \subset \mathbb{Z}$ |
| $\cup$ | "union" | $A \cup B$ (all elements in A or B) |
| $\cap$ | "intersection" | $A \cap B$ (elements in both A and B) |
| $\emptyset$ | "empty set" | A set with no elements |
| $\forall$ | "for all" | $\forall x \in \mathbb{R}$ means "for all x in real numbers" |
| $\exists$ | "there exists" | $\exists x : x > 0$ means "there exists an x such that x > 0" |

### 1.2 Common Notation Conventions

**Scalars** (single numbers):
- Lower case letters: $a, b, c, x, y, z$
- Greek letters: $\alpha, \beta, \gamma, \lambda, \mu$

**Vectors** (ordered lists of numbers):
- Bold lower case: $\mathbf{x}, \mathbf{y}, \mathbf{v}, \mathbf{w}$
- Sometimes with arrows: $\vec{x}, \vec{y}$

**Matrices** (rectangular arrays of numbers):
- Bold upper case: $\mathbf{A}, \mathbf{B}, \mathbf{X}, \mathbf{Y}$
- Sometimes just upper case: $A, B, X, Y$

**Sets**:
- Upper case letters: $A, B, S, V$
- Number systems: $\mathbb{N}, \mathbb{Z}, \mathbb{Q}, \mathbb{R}, \mathbb{C}$

---

## Part 2: Number Systems

### 2.1 Common Number Systems

| Symbol | Name | Description | Examples |
|--------|------|-------------|----------|
| $\mathbb{N}$ | Natural numbers | Positive integers | $1, 2, 3, 4, \ldots$ |
| $\mathbb{Z}$ | Integers | Whole numbers | $\ldots, -2, -1, 0, 1, 2, \ldots$ |
| $\mathbb{Q}$ | Rational numbers | Fractions | $\frac{1}{2}, \frac{3}{4}, -\frac{5}{3}$ |
| $\mathbb{R}$ | Real numbers | All numbers on number line | $\pi, \sqrt{2}, -3.5$ |
| $\mathbb{C}$ | Complex numbers | Numbers with imaginary part | $2 + 3i, -1 + i$ |

### 2.2 Vector Spaces

**$\mathbb{R}^n$** - The set of all n-dimensional real vectors

- $\mathbb{R}^1$ = Real numbers (1D)
- $\mathbb{R}^2$ = Pairs of real numbers (2D plane)
- $\mathbb{R}^3$ = Triples of real numbers (3D space)
- $\mathbb{R}^n$ = n-tuples of real numbers (n-dimensional space)

**Example:**
$$\mathbf{x} = \begin{bmatrix} 2 \\ 3 \end{bmatrix} \in \mathbb{R}^2$$

This means $\mathbf{x}$ is a 2-dimensional vector with components 2 and 3.

---

## Part 3: Vectors

### 3.1 What is a Vector?

A **vector** is an ordered list of numbers representing:
- A point in space
- A direction and magnitude
- Features in machine learning

**Example in $\mathbb{R}^3$:**
$$\mathbf{v} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$$

### 3.2 Vector Terminology

**Dimension**: The number of components in a vector

**Notation:**
- Column vector (default): $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}$
- Row vector: $\mathbf{x}^T = \begin{bmatrix} x_1 & x_2 & x_3 \end{bmatrix}$

**Components/Entries**: The individual numbers in a vector
- $x_1$ is the first component
- $x_2$ is the second component
- $x_i$ is the i-th component

### 3.3 Basic Vector Operations

#### Addition
$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}$$

**Example:**
$$\begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 3 \\ 4 \end{bmatrix} = \begin{bmatrix} 4 \\ 6 \end{bmatrix}$$

#### Scalar Multiplication
$$c \cdot \mathbf{a} = c \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} = \begin{bmatrix} c \cdot a_1 \\ c \cdot a_2 \end{bmatrix}$$

**Example:**
$$3 \cdot \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 3 \\ 6 \end{bmatrix}$$

#### Dot Product (Inner Product)
$$\mathbf{a} \cdot \mathbf{b} = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n = \sum_{i=1}^{n} a_i b_i$$

**Example:**
$$\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \cdot \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} = 1(4) + 2(5) + 3(6) = 4 + 10 + 18 = 32$$

---

## Part 4: Matrices

### 4.1 What is a Matrix?

A **matrix** is a rectangular array of numbers arranged in rows and columns.

**General form:**
$$\mathbf{A} = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix}$$

### 4.2 Matrix Terminology

**Dimension/Size**: $m \times n$ where
- $m$ = number of rows
- $n$ = number of columns

**Example:**
$$\mathbf{A} = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}_{2 \times 3}$$

This is a $2 \times 3$ matrix (2 rows, 3 columns).

**Entry/Element**: $a_{ij}$ is the element in row $i$, column $j$

**Square Matrix**: A matrix where $m = n$ (same number of rows and columns)

**Example:**
$$\mathbf{B} = \begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}_{2 \times 2}$$

### 4.3 Special Matrices

#### Identity Matrix ($\mathbf{I}$)
A square matrix with 1s on the diagonal and 0s elsewhere.

$$\mathbf{I}_3 = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}$$

**Property:** $\mathbf{A} \mathbf{I} = \mathbf{I} \mathbf{A} = \mathbf{A}$

#### Zero Matrix
A matrix where all elements are zero.

$$\mathbf{0} = \begin{bmatrix}
0 & 0 \\
0 & 0
\end{bmatrix}$$

#### Diagonal Matrix
A square matrix where all off-diagonal elements are zero.

$$\mathbf{D} = \begin{bmatrix}
d_1 & 0 & 0 \\
0 & d_2 & 0 \\
0 & 0 & d_3
\end{bmatrix}$$

#### Transpose ($\mathbf{A}^T$)
Flip rows and columns.

**Example:**
$$\mathbf{A} = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}, \quad \mathbf{A}^T = \begin{bmatrix}
1 & 4 \\
2 & 5 \\
3 & 6
\end{bmatrix}$$

**Rule:** $(a_{ij})^T = a_{ji}$

---

## Part 5: Matrix Operations

### 5.1 Matrix Addition
Add corresponding elements (matrices must have same dimensions).

$$\mathbf{A} + \mathbf{B} = \begin{bmatrix}
a_{11} + b_{11} & a_{12} + b_{12} \\
a_{21} + b_{21} & a_{22} + b_{22}
\end{bmatrix}$$

**Example:**
$$\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix} + \begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix} = \begin{bmatrix}
6 & 8 \\
10 & 12
\end{bmatrix}$$

### 5.2 Scalar Multiplication
Multiply every element by a scalar.

$$c \mathbf{A} = \begin{bmatrix}
c \cdot a_{11} & c \cdot a_{12} \\
c \cdot a_{21} & c \cdot a_{22}
\end{bmatrix}$$

**Example:**
$$2 \begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix} = \begin{bmatrix}
2 & 4 \\
6 & 8
\end{bmatrix}$$

### 5.3 Matrix Multiplication
**Rule:** To multiply $\mathbf{A}_{m \times n}$ and $\mathbf{B}_{n \times p}$:
- Number of columns in $\mathbf{A}$ must equal number of rows in $\mathbf{B}$
- Result is $\mathbf{C}_{m \times p}$

**Formula:**
$$c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

**Example:**
$$\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix} \begin{bmatrix}
5 & 6 \\
7 & 8
\end{bmatrix} = \begin{bmatrix}
1(5) + 2(7) & 1(6) + 2(8) \\
3(5) + 4(7) & 3(6) + 4(8)
\end{bmatrix} = \begin{bmatrix}
19 & 22 \\
43 & 50
\end{bmatrix}$$

**Important:** Matrix multiplication is **NOT commutative**
- $\mathbf{AB} \neq \mathbf{BA}$ (in general)

---

## Part 6: Important Concepts

### 6.1 Linear Combination
A sum of scalar multiples of vectors.

$$\mathbf{v} = c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_n \mathbf{v}_n$$

**Example:**
$$\mathbf{v} = 2\begin{bmatrix} 1 \\ 0 \end{bmatrix} + 3\begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$$

### 6.2 Linear Independence
Vectors are **linearly independent** if no vector can be written as a linear combination of the others.

**Example (Independent):**
$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

**Example (Dependent):**
$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$$
(Here $\mathbf{v}_2 = 2\mathbf{v}_1$)

### 6.3 Span
The **span** of a set of vectors is all possible linear combinations of those vectors.

$$\text{span}\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n\} = \{c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_n\mathbf{v}_n : c_i \in \mathbb{R}\}$$

### 6.4 Basis
A **basis** for a vector space is a set of linearly independent vectors that span the space.

**Standard basis for $\mathbb{R}^3$:**
$$\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad \mathbf{e}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

### 6.5 Dimension
The **dimension** of a vector space is the number of vectors in any basis.

- $\mathbb{R}^1$ has dimension 1
- $\mathbb{R}^2$ has dimension 2
- $\mathbb{R}^3$ has dimension 3
- $\mathbb{R}^n$ has dimension n

---

## Part 7: Functions and Mappings

### 7.1 Function Notation
$$f: A \to B$$

Means: "function $f$ maps from set $A$ to set $B$"
- $A$ is the **domain** (input set)
- $B$ is the **codomain** (output set)
- $f(x)$ is the **image** of $x$

**Example:**
$$f: \mathbb{R} \to \mathbb{R}, \quad f(x) = x^2$$

### 7.2 Linear Transformation
A function $T: \mathbb{R}^n \to \mathbb{R}^m$ is **linear** if:

1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$ (additivity)
2. $T(c\mathbf{u}) = cT(\mathbf{u})$ (homogeneity)

**Key fact:** Every linear transformation can be represented by a matrix!

---

## Part 8: Norms and Distance

### 8.1 Vector Norm
A **norm** measures the "length" or "magnitude" of a vector.

**Euclidean Norm (L2 norm):**
$$\|\mathbf{x}\|_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2} = \sqrt{\sum_{i=1}^{n} x_i^2}$$

**Example:**
$$\left\|\begin{bmatrix} 3 \\ 4 \end{bmatrix}\right\|_2 = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

**Other norms:**
- L1 norm: $\|\mathbf{x}\|_1 = |x_1| + |x_2| + \cdots + |x_n|$
- L∞ norm: $\|\mathbf{x}\|_\infty = \max\{|x_1|, |x_2|, \ldots, |x_n|\}$

### 8.2 Distance
The **distance** between two vectors:
$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|$$

**Example:**
$$d\left(\begin{bmatrix} 1 \\ 2 \end{bmatrix}, \begin{bmatrix} 4 \\ 6 \end{bmatrix}\right) = \left\|\begin{bmatrix} -3 \\ -4 \end{bmatrix}\right\| = \sqrt{9 + 16} = 5$$

---

## Part 9: Summation and Product Notation

### 9.1 Summation ($\Sigma$)
$$\sum_{i=1}^{n} a_i = a_1 + a_2 + a_3 + \cdots + a_n$$

**Examples:**
$$\sum_{i=1}^{5} i = 1 + 2 + 3 + 4 + 5 = 15$$

$$\sum_{i=1}^{3} i^2 = 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14$$

### 9.2 Product ($\Pi$)
$$\prod_{i=1}^{n} a_i = a_1 \times a_2 \times a_3 \times \cdots \times a_n$$

**Example:**
$$\prod_{i=1}^{4} i = 1 \times 2 \times 3 \times 4 = 24$$

---

## Part 10: Common Functions

### 10.1 Exponential Function
$$f(x) = e^x \text{ where } e \approx 2.71828$$

**Properties:**
- $e^0 = 1$
- $e^{a+b} = e^a \cdot e^b$
- $(e^x)' = e^x$

### 10.2 Logarithm
$$y = \log_b(x) \text{ means } b^y = x$$

**Natural logarithm:** $\ln(x) = \log_e(x)$

**Properties:**
- $\ln(1) = 0$
- $\ln(ab) = \ln(a) + \ln(b)$
- $\ln(a^b) = b\ln(a)$

### 10.3 Sigmoid Function
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Properties:**
- Range: $(0, 1)$
- $\sigma(0) = 0.5$
- Used in logistic regression and neural networks

---

## Summary: Key Takeaways

### Essential Notation
- Vectors: $\mathbf{x} \in \mathbb{R}^n$
- Matrices: $\mathbf{A} \in \mathbb{R}^{m \times n}$
- Sets: $A \subset B$, $x \in S$

### Fundamental Operations
- Vector addition, scalar multiplication, dot product
- Matrix addition, multiplication, transpose
- Norms and distances

### Core Concepts
- Linear independence and span
- Basis and dimension
- Linear transformations
- Functions and mappings

---

## Practice Problems

### Problem 1
Calculate the following:
$$\begin{bmatrix} 2 \\ 3 \\ 1 \end{bmatrix} \cdot \begin{bmatrix} 1 \\ 4 \\ 2 \end{bmatrix}$$

### Problem 2
Find the norm:
$$\left\|\begin{bmatrix} 5 \\ 12 \end{bmatrix}\right\|_2$$

### Problem 3
Multiply these matrices:
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 1 & 3 \end{bmatrix}$$

### Problem 4
Are these vectors linearly independent?
$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 3 \\ 6 \end{bmatrix}$$

---

## Solutions

**Solution 1:**
$$2(1) + 3(4) + 1(2) = 2 + 12 + 2 = 16$$

**Solution 2:**
$$\sqrt{5^2 + 12^2} = \sqrt{25 + 144} = \sqrt{169} = 13$$

**Solution 3:**
$$\begin{bmatrix} 1(2) + 2(1) & 1(0) + 2(3) \\ 3(2) + 4(1) & 3(0) + 4(3) \end{bmatrix} = \begin{bmatrix} 4 & 6 \\ 10 & 12 \end{bmatrix}$$

**Solution 4:**
No, they are **dependent** because $\mathbf{v}_2 = 3\mathbf{v}_1$

---

**Course:** Mathematics for Machine Learning
**Instructor:** Mohammed Alnemari

**Next:** Tutorial 2 - Vector Spaces and Linear Transformations
