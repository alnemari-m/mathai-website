# Tutorial 2: Linear Algebra

**Course:** Mathematics for Machine Learning
**Instructor:** Mohammed Alnemari

---

## 📚 Learning Objectives

By the end of this tutorial, you will understand:

1. Systems of linear equations and the three types of solutions
2. Matrices, their types, and core operations
3. Matrix inverses and how to compute them
4. Gaussian elimination and row echelon form
5. Vector spaces, subspaces, and the subspace test
6. Linear independence and how to check it
7. Span, generating sets, basis, and dimension
8. Rank of a matrix and its role in solvability
9. Linear mappings, kernel, and image
10. The Rank-Nullity Theorem
11. Change of basis
12. Affine spaces and their connection to solution sets

---

## Part 1: Systems of Linear Equations

### 1.1 What is a System of Linear Equations?

A **system of linear equations** is a collection of equations where each equation is linear (no powers, no products of variables). We want to find the values of the unknowns that satisfy *all* the equations at the same time.

**General form** (two equations, two unknowns):

$$a_{11}x_1 + a_{12}x_2 = b_1$$

$$a_{21}x_1 + a_{22}x_2 = b_2$$

In plain English: we have some unknowns ($x_1, x_2$), each multiplied by known constants ($a_{ij}$), and we want the results to equal known values ($b_1, b_2$).

> **Think of it as...** each equation describes a line (in 2D) or a plane (in 3D). Solving the system means finding where all the lines or planes intersect.

### 1.2 The Three Types of Solutions

Every system of linear equations has exactly one of three outcomes:

| Type | Description | Geometric Picture (2D) | Example |
|------|-------------|------------------------|---------|
| **Unique solution** | Exactly one solution | Two lines cross at a single point | $x + y = 3$, $x - y = 1$ gives $x=2, y=1$ |
| **Infinitely many solutions** | A family of solutions | Two lines lie on top of each other | $x + y = 2$, $2x + 2y = 4$ (same line) |
| **No solution** | The system is inconsistent | Two lines are parallel and never meet | $x + y = 1$, $x + y = 3$ (parallel lines) |

**Worked Example:** Determine the type of solution.

System 1: $x + y = 5$ and $x - y = 1$

Add the equations: $2x = 6$, so $x = 3$, $y = 2$. **Unique solution.**

System 2: $x + y = 2$ and $2x + 2y = 4$

The second equation is just $2 \times$ the first. Every point on the line $x + y = 2$ is a solution. **Infinitely many solutions.**

System 3: $x + y = 1$ and $x + y = 3$

These say the same expression $x+y$ equals two different things. Impossible. **No solution.**

### 1.3 Matrix Notation: $A\mathbf{x} = \mathbf{b}$

We can write any system of linear equations compactly using matrices:

$$\underbrace{\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}}_{A} \underbrace{\begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}}_{\mathbf{x}} = \underbrace{\begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}}_{\mathbf{b}}$$

- $A$ is the **coefficient matrix** ($m \times n$)
- $\mathbf{x}$ is the **unknown vector** ($n \times 1$)
- $\mathbf{b}$ is the **right-hand side vector** ($m \times 1$)

### 1.4 The Augmented Matrix

The **augmented matrix** $[A \mid \mathbf{b}]$ combines the coefficient matrix and the right-hand side into one object, which is convenient for Gaussian elimination:

$$[A \mid \mathbf{b}] = \left[\begin{array}{ccc|c} a_{11} & a_{12} & \cdots & b_1 \\ a_{21} & a_{22} & \cdots & b_2 \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & b_m \end{array}\right]$$

**Example:** The system $2x + 3y = 7$ and $x - y = 1$ becomes:

$$\left[\begin{array}{cc|c} 2 & 3 & 7 \\ 1 & -1 & 1 \end{array}\right]$$

> **Think of it as...** packing all the important numbers from your system into one neat table, with a vertical line separating the left side from the right side of the equals sign.

---

## Part 2: Matrices

### 2.1 Definition

A **matrix** is a rectangular array of numbers with $m$ rows and $n$ columns. We say it has dimensions $m \times n$ (read "m by n").

$$A \in \mathbb{R}^{m \times n} \quad \text{means } A \text{ has } m \text{ rows and } n \text{ columns of real numbers.}$$

The entry in row $i$ and column $j$ is written $a_{ij}$.

### 2.2 Special Matrices

| Matrix | Definition | Example |
|--------|-----------|---------|
| **Identity** $I_n$ | Square matrix with 1s on the diagonal, 0s everywhere else | $I_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$ |
| **Zero** $\mathbf{0}$ | All entries are zero | $\begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}$ |
| **Diagonal** | Only the diagonal entries can be nonzero | $\begin{bmatrix} 3 & 0 \\ 0 & 7 \end{bmatrix}$ |
| **Symmetric** | $A = A^T$ (equal to its own transpose) | $\begin{bmatrix} 1 & 4 \\ 4 & 5 \end{bmatrix}$ |
| **Square** | Number of rows equals number of columns ($m = n$) | Any $n \times n$ matrix |

**Key property of the identity matrix:** For any matrix $A$ of compatible size:

$$AI = IA = A$$

> **Think of it as...** the identity matrix is like multiplying by 1. It does nothing to whatever it touches.

### 2.3 Matrix Operations

#### Matrix Addition

Add corresponding entries. Both matrices must have the same dimensions.

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}$$

#### Scalar Multiplication

Multiply every entry by a number.

$$3 \begin{bmatrix} 1 & 2 \\ 4 & 5 \end{bmatrix} = \begin{bmatrix} 3 & 6 \\ 12 & 15 \end{bmatrix}$$

#### Matrix Multiplication

To compute $C = AB$ where $A$ is $m \times n$ and $B$ is $n \times p$, each entry of $C$ is:

$$c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj} = (\text{row } i \text{ of } A) \cdot (\text{column } j \text{ of } B)$$

The result $C$ has dimensions $m \times p$.

**Requirement:** The number of columns of $A$ must equal the number of rows of $B$.

**Worked Example:**

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1(5)+2(7) & 1(6)+2(8) \\ 3(5)+4(7) & 3(6)+4(8) \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

**Critical fact:** Matrix multiplication is **NOT commutative** in general:

$$AB \neq BA$$

> **Think of it as...** putting on socks then shoes is not the same as putting on shoes then socks. The order matters.

### 2.4 Transpose

The **transpose** $A^T$ flips a matrix over its diagonal: rows become columns and columns become rows.

If $A$ is $m \times n$, then $A^T$ is $n \times m$, and $(A^T)_{ij} = A_{ji}$.

**Example:**

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \quad A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

**Transpose properties:**

| Property | Formula |
|----------|---------|
| Double transpose | $(A^T)^T = A$ |
| Sum | $(A + B)^T = A^T + B^T$ |
| Scalar | $(cA)^T = cA^T$ |
| Product | $(AB)^T = B^T A^T$ (note the reversed order!) |

---

## Part 3: Matrix Inverse

### 3.1 Definition

The **inverse** of a square matrix $A$ is a matrix $A^{-1}$ such that:

$$AA^{-1} = A^{-1}A = I$$

In plain English: $A^{-1}$ "undoes" whatever $A$ does. If $A$ transforms a vector, $A^{-1}$ transforms it back.

A matrix that has an inverse is called **invertible** (or **non-singular** or **regular**). A matrix that does not have an inverse is called **singular**.

> **Think of it as...** the inverse is like an "undo button." If $A$ scrambles your data, $A^{-1}$ unscrambles it perfectly.

### 3.2 When Does the Inverse Exist?

A square matrix $A$ is invertible if and only if:

- $\det(A) \neq 0$
- The columns of $A$ are linearly independent
- The only solution to $A\mathbf{x} = \mathbf{0}$ is $\mathbf{x} = \mathbf{0}$
- $\text{rank}(A) = n$ (full rank)

These are all equivalent conditions -- if one is true, they are all true.

### 3.3 Computing the Inverse of a 2x2 Matrix

For $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the inverse is:

$$A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

The number $ad - bc$ is the **determinant** $\det(A)$. If $\det(A) = 0$, the inverse does not exist.

**Worked Example:**

$$A = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix}$$

$$\det(A) = 4(6) - 7(2) = 24 - 14 = 10$$

$$A^{-1} = \frac{1}{10}\begin{bmatrix} 6 & -7 \\ -2 & 4 \end{bmatrix} = \begin{bmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{bmatrix}$$

**Verify:** $AA^{-1} = \begin{bmatrix} 4 & 7 \\ 2 & 6 \end{bmatrix}\begin{bmatrix} 0.6 & -0.7 \\ -0.2 & 0.4 \end{bmatrix} = \begin{bmatrix} 2.4-1.4 & -2.8+2.8 \\ 1.2-1.2 & -1.4+2.4 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I \quad \checkmark$

### 3.4 Properties of the Inverse

| Property | Formula |
|----------|---------|
| Inverse of inverse | $(A^{-1})^{-1} = A$ |
| Inverse of product | $(AB)^{-1} = B^{-1}A^{-1}$ (note the reversed order!) |
| Inverse of transpose | $(A^T)^{-1} = (A^{-1})^T$ |
| Inverse of scalar multiple | $(cA)^{-1} = \frac{1}{c}A^{-1}$ for $c \neq 0$ |

The property $(AB)^{-1} = B^{-1}A^{-1}$ is very important. Notice the order reverses -- just like with the transpose of a product.

> **Think of it as...** if you put on socks then shoes, to undo it you take off shoes first, then socks. The same "reverse order" logic applies to matrix inverses.

---

## Part 4: Solving Systems -- Gaussian Elimination

### 4.1 The Idea

**Gaussian elimination** is a systematic method for solving $A\mathbf{x} = \mathbf{b}$. The idea is simple: use allowed row operations to transform the augmented matrix $[A \mid \mathbf{b}]$ into a simpler form from which the solution is easy to read off.

### 4.2 Elementary Row Operations

There are three operations you are allowed to perform on rows. None of them change the solution set of the system.

| Operation | Description | Example |
|-----------|-------------|---------|
| **Swap** | Exchange two rows: $R_i \leftrightarrow R_j$ | Swap row 1 and row 2 |
| **Scale** | Multiply a row by a nonzero constant: $R_i \to cR_i$ | Multiply row 2 by $\frac{1}{3}$ |
| **Add** | Add a multiple of one row to another: $R_i \to R_i + cR_j$ | Add $(-2) \times$ row 1 to row 2 |

> **Think of it as...** you are doing the same algebra you would do by hand (adding equations, multiplying both sides by a constant), but in a more organized table format.

### 4.3 Row Echelon Form (REF)

A matrix is in **row echelon form** if:

1. All rows that are entirely zero are at the bottom
2. The first nonzero entry in each row (called the **pivot**) is to the right of the pivot in the row above
3. All entries below each pivot are zero

**Example of REF:**

$$\begin{bmatrix} 2 & 1 & -1 \\ 0 & 3 & 5 \\ 0 & 0 & 4 \end{bmatrix}$$

The pivots are 2, 3, and 4. Each is to the right and below the previous one, forming a "staircase" pattern.

### 4.4 Reduced Row Echelon Form (RREF)

A matrix is in **reduced row echelon form** if it is in REF and additionally:

4. Every pivot is 1
5. Every pivot is the only nonzero entry in its column

**Example of RREF:**

$$\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

When the augmented matrix $[A \mid \mathbf{b}]$ is in RREF, you can read the solution directly.

### 4.5 Full Worked Example: Solving a System via Gaussian Elimination

Solve the system:

$$x_1 + 2x_2 - x_3 = 3$$

$$2x_1 + 5x_2 - 2x_3 = 7$$

$$-x_1 - x_2 + 3x_3 = 2$$

**Step 1:** Write the augmented matrix.

$$\left[\begin{array}{ccc|c} 1 & 2 & -1 & 3 \\ 2 & 5 & -2 & 7 \\ -1 & -1 & 3 & 2 \end{array}\right]$$

**Step 2:** Eliminate below the first pivot (the 1 in position $(1,1)$).

$R_2 \to R_2 - 2R_1$: Replace row 2 with (row 2 minus 2 times row 1).

$R_3 \to R_3 + R_1$: Replace row 3 with (row 3 plus row 1).

$$\left[\begin{array}{ccc|c} 1 & 2 & -1 & 3 \\ 0 & 1 & 0 & 1 \\ 0 & 1 & 2 & 5 \end{array}\right]$$

**Step 3:** Eliminate below the second pivot (the 1 in position $(2,2)$).

$R_3 \to R_3 - R_2$:

$$\left[\begin{array}{ccc|c} 1 & 2 & -1 & 3 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 2 & 4 \end{array}\right]$$

This is now in **row echelon form**. We have three pivots for three unknowns, so there is a unique solution.

**Step 4:** Scale row 3 to make the pivot equal to 1.

$R_3 \to \frac{1}{2}R_3$:

$$\left[\begin{array}{ccc|c} 1 & 2 & -1 & 3 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 2 \end{array}\right]$$

**Step 5:** Back substitution (or continue to RREF). Let us continue to RREF.

$R_1 \to R_1 + R_3$ (eliminate the $-1$ in position $(1,3)$):

$$\left[\begin{array}{ccc|c} 1 & 2 & 0 & 5 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 2 \end{array}\right]$$

$R_1 \to R_1 - 2R_2$ (eliminate the $2$ in position $(1,2)$):

$$\left[\begin{array}{ccc|c} 1 & 0 & 0 & 3 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 2 \end{array}\right]$$

**Step 6:** Read the solution from RREF.

$$x_1 = 3, \quad x_2 = 1, \quad x_3 = 2$$

**Verify:** Plug back into the original equations:

- $1(3) + 2(1) - 1(2) = 3 + 2 - 2 = 3$ $\checkmark$
- $2(3) + 5(1) - 2(2) = 6 + 5 - 4 = 7$ $\checkmark$
- $-1(3) - 1(1) + 3(2) = -3 - 1 + 6 = 2$ $\checkmark$

---

## Part 5: Vector Spaces

### 5.1 What is a Vector Space?

A **vector space** is a set $V$ of objects (called "vectors") together with two operations -- vector addition and scalar multiplication -- that satisfy certain rules (axioms).

In plain English: a vector space is a collection of things that you can add together and scale, and the result always stays inside the same collection.

> **Think of it as...** a playground with fences. You can run around (add vectors, scale them) however you like, but you can never leave the playground. Everything you create stays inside.

### 5.2 The Vector Space Axioms

For all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and all scalars $\alpha, \beta \in \mathbb{R}$:

| Axiom | Statement | In Plain English |
|-------|-----------|-----------------|
| Closure under addition | $\mathbf{u} + \mathbf{v} \in V$ | Adding two vectors keeps you in $V$ |
| Closure under scalar multiplication | $\alpha \mathbf{u} \in V$ | Scaling a vector keeps you in $V$ |
| Associativity of addition | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ | Grouping does not matter for addition |
| Commutativity of addition | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ | Order does not matter for addition |
| Zero vector exists | $\exists \mathbf{0} \in V : \mathbf{u} + \mathbf{0} = \mathbf{u}$ | There is a "do-nothing" element |
| Additive inverse | $\forall \mathbf{u} \in V, \exists (-\mathbf{u}) : \mathbf{u} + (-\mathbf{u}) = \mathbf{0}$ | Every vector has a "negative" |
| Distributivity (vector) | $\alpha(\mathbf{u} + \mathbf{v}) = \alpha\mathbf{u} + \alpha\mathbf{v}$ | Scaling distributes over vector addition |
| Distributivity (scalar) | $(\alpha + \beta)\mathbf{u} = \alpha\mathbf{u} + \beta\mathbf{u}$ | Scaling distributes over scalar addition |
| Associativity of scaling | $\alpha(\beta \mathbf{u}) = (\alpha\beta)\mathbf{u}$ | Order of scaling does not matter |
| Identity of scaling | $1 \cdot \mathbf{u} = \mathbf{u}$ | Scaling by 1 does nothing |

### 5.3 Common Examples of Vector Spaces

- $\mathbb{R}^n$: the set of all $n$-tuples of real numbers (the most common example)
- $\mathbb{R}^{m \times n}$: the set of all $m \times n$ real matrices
- The set of all polynomials of degree at most $n$
- The set of all continuous functions $f: \mathbb{R} \to \mathbb{R}$

### 5.4 Subspaces

A **subspace** $U$ of a vector space $V$ is a subset $U \subseteq V$ that is itself a vector space under the same operations.

#### The 3-Step Subspace Test

To check whether a subset $U \subseteq V$ is a subspace, verify three things:

| Step | Check | Why |
|------|-------|-----|
| 1 | $\mathbf{0} \in U$ (the zero vector is in $U$) | Every vector space must contain the zero vector |
| 2 | $\mathbf{u} + \mathbf{v} \in U$ for all $\mathbf{u}, \mathbf{v} \in U$ (closed under addition) | Adding members should not kick you out |
| 3 | $\alpha \mathbf{u} \in U$ for all $\alpha \in \mathbb{R}$, $\mathbf{u} \in U$ (closed under scalar multiplication) | Scaling should not kick you out |

If all three pass, $U$ is a subspace. If any one fails, it is not.

> **Think of it as...** a "mini playground" inside the bigger playground. As long as the mini playground contains the origin and any combination of its members stays inside, it qualifies as a subspace.

**Example (is a subspace):** Let $U = \left\{\begin{bmatrix} x \\ 0 \end{bmatrix} : x \in \mathbb{R}\right\} \subseteq \mathbb{R}^2$ (the $x$-axis).

- Zero vector: $\begin{bmatrix} 0 \\ 0 \end{bmatrix} \in U$ $\checkmark$
- Closed under addition: $\begin{bmatrix} a \\ 0 \end{bmatrix} + \begin{bmatrix} b \\ 0 \end{bmatrix} = \begin{bmatrix} a+b \\ 0 \end{bmatrix} \in U$ $\checkmark$
- Closed under scalar multiplication: $c\begin{bmatrix} a \\ 0 \end{bmatrix} = \begin{bmatrix} ca \\ 0 \end{bmatrix} \in U$ $\checkmark$

**Example (not a subspace):** Let $W = \left\{\begin{bmatrix} x \\ 1 \end{bmatrix} : x \in \mathbb{R}\right\} \subseteq \mathbb{R}^2$ (a horizontal line at height 1).

- Zero vector: $\begin{bmatrix} 0 \\ 0 \end{bmatrix} \notin W$ because the second component is always 1, not 0. $\times$

$W$ fails the very first check, so it is not a subspace.

---

## Part 6: Linear Independence

### 6.1 Definition

A set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ is **linearly independent** if the only way to combine them to get the zero vector is with all coefficients equal to zero:

$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0} \implies c_1 = c_2 = \cdots = c_k = 0$$

If there is a nontrivial combination (some $c_i \neq 0$) that gives $\mathbf{0}$, the vectors are **linearly dependent**.

In plain English: vectors are independent when none of them is "redundant" -- you cannot build any one of them from the others.

> **Think of it as...** independent vectors each bring something genuinely new to the table. Dependent vectors are freeloaders -- at least one of them could be replaced by a combination of the rest.

### 6.2 How to Check Linear Independence

**Method:** Form a matrix with the vectors as columns and row-reduce to REF. Count the pivots.

- If every column has a pivot, the vectors are **linearly independent**.
- If any column lacks a pivot, the vectors are **linearly dependent**.

**Worked Example:** Are the following vectors linearly independent?

$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}, \quad \mathbf{v}_3 = \begin{bmatrix} 1 \\ 3 \\ 4 \end{bmatrix}$$

Form the matrix $[\mathbf{v}_1 \mid \mathbf{v}_2 \mid \mathbf{v}_3]$ and row reduce:

$$\begin{bmatrix} 1 & 0 & 1 \\ 2 & 1 & 3 \\ 3 & 1 & 4 \end{bmatrix} \xrightarrow{R_2 - 2R_1} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 3 & 1 & 4 \end{bmatrix} \xrightarrow{R_3 - 3R_1} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 1 & 1 \end{bmatrix} \xrightarrow{R_3 - R_2} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{bmatrix}$$

Only 2 pivots for 3 vectors. Column 3 has no pivot, so $\mathbf{v}_3$ is a linear combination of the others. The vectors are **linearly dependent**.

Indeed, $\mathbf{v}_3 = \mathbf{v}_1 + \mathbf{v}_2 = \begin{bmatrix} 1+0 \\ 2+1 \\ 3+1 \end{bmatrix} = \begin{bmatrix} 1 \\ 3 \\ 4 \end{bmatrix}$.

### 6.3 Geometric Meaning

| Number of vectors | Independent means... | Dependent means... |
|-------------------|---------------------|---------------------|
| 2 vectors in $\mathbb{R}^2$ | They point in different directions | They lie on the same line |
| 3 vectors in $\mathbb{R}^3$ | They span all of 3D space | They all lie in the same plane (or line) |
| $k$ vectors in $\mathbb{R}^n$ | They span a $k$-dimensional subspace | They span something less than $k$-dimensional |

### 6.4 Maximum Number of Independent Vectors

In $\mathbb{R}^n$, you can have **at most $n$** linearly independent vectors. If you have more than $n$ vectors in $\mathbb{R}^n$, they are guaranteed to be linearly dependent.

---

## Part 7: Generating Sets, Span, and Basis

### 7.1 Span

The **span** of a set of vectors is the set of all possible linear combinations of those vectors:

$$\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\} = \left\{c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k : c_1, \ldots, c_k \in \mathbb{R}\right\}$$

In plain English: the span is everything you can "reach" by adding and scaling the given vectors.

### 7.2 Generating Set

A set of vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is a **generating set** (or spanning set) for a vector space $V$ if:

$$V = \text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$$

This means every vector in $V$ can be written as a linear combination of $\mathbf{v}_1, \ldots, \mathbf{v}_k$.

### 7.3 Basis

A **basis** for a vector space $V$ is a set of vectors that is:

1. **Linearly independent** (no redundancy)
2. **Spanning** (reaches everything in $V$)

A basis is the most "efficient" generating set -- it spans the whole space with no wasted vectors.

**Standard basis for $\mathbb{R}^3$:**

$$\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad \mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad \mathbf{e}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

These three vectors are linearly independent and every vector in $\mathbb{R}^3$ can be written as $\mathbf{v} = v_1\mathbf{e}_1 + v_2\mathbf{e}_2 + v_3\mathbf{e}_3$.

> **Think of it as...** a basis is like a set of building blocks. You need enough blocks to build anything in the space, but you do not want any block that is just a copy of others glued together. The standard basis in $\mathbb{R}^3$ uses the three coordinate axes as blocks.

### 7.4 Dimension

The **dimension** of a vector space $V$ is the number of vectors in any basis of $V$. This number is always the same no matter which basis you choose.

$$\dim(\mathbb{R}^n) = n$$

**Examples:**

- A line through the origin in $\mathbb{R}^3$ has dimension 1
- A plane through the origin in $\mathbb{R}^3$ has dimension 2
- $\mathbb{R}^3$ itself has dimension 3

### 7.5 Finding a Basis for a Set of Vectors

Given vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$, to find a basis for $\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$:

1. Form the matrix $A = [\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_k]$
2. Row reduce to REF
3. The original vectors corresponding to pivot columns form a basis

---

## Part 8: Rank

### 8.1 Definition

The **rank** of a matrix $A$ is the number of linearly independent columns (equivalently, the number of linearly independent rows).

A fundamental fact: **column rank = row rank**. This means you get the same number whether you count independent columns or independent rows.

### 8.2 Computing Rank via RREF

To find the rank of $A$:

1. Row reduce $A$ to REF or RREF
2. Count the number of pivots (the nonzero leading entries)

$$\text{rank}(A) = \text{number of pivots}$$

**Worked Example:**

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 0 & 1 & 1 \end{bmatrix}$$

Row reduce:

$$\xrightarrow{R_2 - 2R_1} \begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 0 & 1 & 1 \end{bmatrix} \xrightarrow{R_2 \leftrightarrow R_3} \begin{bmatrix} 1 & 2 & 3 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{bmatrix}$$

There are **2 pivots** (in columns 1 and 2), so $\text{rank}(A) = 2$.

### 8.3 Rank and Solvability of $A\mathbf{x} = \mathbf{b}$

The rank tells you about the solvability of $A\mathbf{x} = \mathbf{b}$:

| Condition | What it means |
|-----------|---------------|
| $\text{rank}(A) = \text{rank}([A \mid \mathbf{b}])$ | The system $A\mathbf{x} = \mathbf{b}$ has at least one solution |
| $\text{rank}(A) < \text{rank}([A \mid \mathbf{b}])$ | The system $A\mathbf{x} = \mathbf{b}$ has no solution |
| $\text{rank}(A) = n$ (number of unknowns) | If a solution exists, it is unique |
| $\text{rank}(A) < n$ | If a solution exists, there are infinitely many |

> **Think of it as...** rank tells you "how many genuinely useful equations" you have. If the rank equals the number of unknowns, you have enough information to pin down a unique answer. If the rank is less, you have leftover freedom, meaning infinitely many solutions.

---

## Part 9: Linear Mappings

### 9.1 Definition

A function $\Phi: V \to W$ between vector spaces is a **linear mapping** (or linear transformation) if it preserves both addition and scalar multiplication:

1. $\Phi(\mathbf{u} + \mathbf{v}) = \Phi(\mathbf{u}) + \Phi(\mathbf{v})$ for all $\mathbf{u}, \mathbf{v} \in V$ (preserves addition)
2. $\Phi(\alpha \mathbf{u}) = \alpha \Phi(\mathbf{u})$ for all $\alpha \in \mathbb{R}$, $\mathbf{u} \in V$ (preserves scalar multiplication)

Equivalently, in one combined condition:

$$\Phi(\alpha \mathbf{u} + \beta \mathbf{v}) = \alpha \Phi(\mathbf{u}) + \beta \Phi(\mathbf{v})$$

In plain English: a linear mapping respects the structure of the vector space. It does not "break" addition or scaling.

> **Think of it as...** a linear mapping is a "well-behaved" transformation. It keeps straight lines straight, keeps the origin fixed, and does not do anything "nonlinear" like bending or shifting.

### 9.2 Transformation Matrix

Every linear mapping $\Phi: \mathbb{R}^n \to \mathbb{R}^m$ can be represented by a matrix $A \in \mathbb{R}^{m \times n}$ such that:

$$\Phi(\mathbf{x}) = A\mathbf{x}$$

To find $A$, compute what $\Phi$ does to each standard basis vector $\mathbf{e}_j$ and place the result as the $j$-th column of $A$:

$$A = [\Phi(\mathbf{e}_1) \mid \Phi(\mathbf{e}_2) \mid \cdots \mid \Phi(\mathbf{e}_n)]$$

### 9.3 Kernel (Null Space)

The **kernel** (or null space) of a linear mapping $\Phi: V \to W$ (or equivalently of its matrix $A$) is the set of all vectors that map to the zero vector:

$$\ker(\Phi) = \{\mathbf{x} \in V : \Phi(\mathbf{x}) = \mathbf{0}\} = \{\mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{0}\}$$

In plain English: the kernel is everything that $\Phi$ "kills" (sends to zero).

**How to compute:** Solve the homogeneous system $A\mathbf{x} = \mathbf{0}$ by row reducing $A$ and expressing the free variables.

The kernel is always a **subspace** of $V$.

### 9.4 Image (Column Space)

The **image** (or range, or column space) of a linear mapping $\Phi$ is the set of all possible outputs:

$$\text{Im}(\Phi) = \{\Phi(\mathbf{x}) : \mathbf{x} \in V\} = \{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\}$$

This is the same as the **column space** of $A$ -- the span of the columns of $A$:

$$\text{Im}(A) = \text{span}\{\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_n\}$$

where $\mathbf{a}_j$ are the columns of $A$.

The image is always a **subspace** of $W$.

### 9.5 Injective, Surjective, Bijective

| Property | Meaning | Condition |
|----------|---------|-----------|
| **Injective** (one-to-one) | Different inputs give different outputs | $\ker(\Phi) = \{\mathbf{0}\}$ |
| **Surjective** (onto) | Every element of $W$ is an output | $\text{Im}(\Phi) = W$ |
| **Bijective** | Both injective and surjective | The mapping has an inverse |

> **Think of it as...** Injective means no two inputs "collide" at the same output. Surjective means every target is "hit" by some input. Bijective means there is a perfect one-to-one pairing between inputs and outputs.

### 9.6 Worked Example: Finding the Kernel and Image

Let $A = \begin{bmatrix} 1 & 2 & 0 \\ 0 & 0 & 1 \end{bmatrix}$.

**Kernel:** Solve $A\mathbf{x} = \mathbf{0}$:

$$\begin{bmatrix} 1 & 2 & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

From the matrix (already in RREF): $x_1 + 2x_2 = 0$ and $x_3 = 0$. Variable $x_2$ is free. Set $x_2 = t$:

$$\mathbf{x} = \begin{bmatrix} -2t \\ t \\ 0 \end{bmatrix} = t\begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix}$$

$$\ker(A) = \text{span}\left\{\begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix}\right\}$$

Dimension of kernel: $\dim(\ker(A)) = 1$.

**Image:** The image is the column space of $A$. Since $A$ has 2 pivots (in columns 1 and 3), the pivot columns form a basis for the image:

$$\text{Im}(A) = \text{span}\left\{\begin{bmatrix} 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \end{bmatrix}\right\} = \mathbb{R}^2$$

Dimension of image: $\dim(\text{Im}(A)) = 2$.

---

## Part 10: The Rank-Nullity Theorem

### 10.1 Statement

For a linear mapping $\Phi: V \to W$ with transformation matrix $A \in \mathbb{R}^{m \times n}$:

$$\dim(\ker(\Phi)) + \dim(\text{Im}(\Phi)) = \dim(V)$$

Or equivalently, using the matrix $A$:

$$\text{nullity}(A) + \text{rank}(A) = n$$

where $n$ is the number of columns of $A$ (the dimension of the domain).

In plain English: the number of dimensions that get "killed" (mapped to zero) plus the number of dimensions that "survive" (form the output) always add up to the total number of input dimensions.

> **Think of it as...** you start with $n$ dimensions of freedom. Some get collapsed to zero (the nullity), and the rest get mapped to actual outputs (the rank). Nothing is lost or created -- everything is accounted for.

### 10.2 Example

From the worked example in Part 9:

- $A \in \mathbb{R}^{2 \times 3}$, so $n = 3$
- $\dim(\ker(A)) = 1$ (the nullity)
- $\dim(\text{Im}(A)) = 2$ (the rank)

Check: $1 + 2 = 3 = n$ $\checkmark$

### 10.3 Using the Theorem

The Rank-Nullity Theorem is useful for:

| If you know... | You can deduce... |
|----------------|-------------------|
| Rank and $n$ | Nullity = $n - \text{rank}$ |
| Nullity and $n$ | Rank = $n - \text{nullity}$ |
| Rank $= n$ | Nullity = 0, so the mapping is injective |
| Rank $= m$ (rows) | The mapping is surjective |

---

## Part 11: Change of Basis

### 11.1 Coordinate Vectors

Every vector $\mathbf{x}$ in a vector space $V$ can be represented as a linear combination of basis vectors. The **coordinate vector** of $\mathbf{x}$ with respect to a basis $B = \{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ is the vector of coefficients:

If $\mathbf{x} = \alpha_1 \mathbf{b}_1 + \alpha_2 \mathbf{b}_2 + \cdots + \alpha_n \mathbf{b}_n$, then the coordinate vector is:

$$[\mathbf{x}]_B = \begin{bmatrix} \alpha_1 \\ \alpha_2 \\ \vdots \\ \alpha_n \end{bmatrix}$$

In plain English: the same vector can look different depending on which coordinate system (basis) you use.

> **Think of it as...** describing a location. You can say "3 blocks east and 2 blocks north" or "3.6 blocks northeast." Same point, different descriptions depending on your coordinate system.

### 11.2 The Change of Basis Matrix

Suppose you have two bases $B = \{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$ and $\tilde{B} = \{\tilde{\mathbf{b}}_1, \ldots, \tilde{\mathbf{b}}_n\}$ for the same vector space. The **change of basis matrix** $S$ from $\tilde{B}$ to $B$ satisfies:

$$[\mathbf{x}]_B = S \, [\mathbf{x}]_{\tilde{B}}$$

To construct $S$: write each new basis vector $\tilde{\mathbf{b}}_j$ in terms of the old basis $B$. The coefficients form the $j$-th column of $S$.

### 11.3 Transformation Under a Change of Basis

If a linear mapping $\Phi$ has transformation matrix $A$ with respect to basis $B$, then with respect to basis $\tilde{B}$, it has transformation matrix:

$$\tilde{A} = S^{-1} A S$$

where $S$ is the change of basis matrix from $\tilde{B}$ to $B$.

**Worked Example:**

Let $B = \left\{\begin{bmatrix} 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \end{bmatrix}\right\}$ (the standard basis) and $\tilde{B} = \left\{\begin{bmatrix} 1 \\ 1 \end{bmatrix}, \begin{bmatrix} 1 \\ -1 \end{bmatrix}\right\}$.

The change of basis matrix $S$ from $\tilde{B}$ to $B$ is formed by writing the new basis vectors in terms of the standard basis (which is trivial since $B$ is the standard basis):

$$S = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}$$

Now if $\mathbf{x}$ has coordinates $[\mathbf{x}]_{\tilde{B}} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}$ in the new basis, the standard coordinates are:

$$[\mathbf{x}]_B = S \begin{bmatrix} 3 \\ 2 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}\begin{bmatrix} 3 \\ 2 \end{bmatrix} = \begin{bmatrix} 5 \\ 1 \end{bmatrix}$$

This means $\mathbf{x} = 3\begin{bmatrix} 1 \\ 1 \end{bmatrix} + 2\begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 5 \\ 1 \end{bmatrix}$.

---

## Part 12: Affine Spaces

### 12.1 Affine Subspaces vs. Vector Subspaces

A **vector subspace** must pass through the origin. An **affine subspace** is a "shifted" version of a vector subspace -- it may or may not pass through the origin.

**Formal definition:** An affine subspace $L$ of a vector space $V$ is a set of the form:

$$L = \mathbf{x}_0 + U = \{\mathbf{x}_0 + \mathbf{u} : \mathbf{u} \in U\}$$

where $\mathbf{x}_0 \in V$ is a fixed point and $U \subseteq V$ is a vector subspace.

In plain English: take a subspace $U$ (which passes through the origin) and slide it so it passes through the point $\mathbf{x}_0$ instead. The result is an affine subspace.

> **Think of it as...** a vector subspace is a flat surface passing through the origin (a line through zero, a plane through zero, etc.). An affine subspace is that same flat surface slid to a different location. A line that does not pass through the origin is affine but not a vector subspace.

### 12.2 Examples

| Geometric object | Vector subspace? | Affine subspace? |
|-----------------|------------------|------------------|
| A line through the origin | Yes | Yes (with $\mathbf{x}_0 = \mathbf{0}$) |
| A line NOT through the origin | No | Yes |
| A plane through the origin | Yes | Yes |
| A plane NOT through the origin | No | Yes |
| A single point | No (unless it is the origin) | Yes (a 0-dimensional affine subspace) |

### 12.3 Connection to Solutions of $A\mathbf{x} = \mathbf{b}$

This is the key insight: **the solution set of $A\mathbf{x} = \mathbf{b}$ (when $\mathbf{b} \neq \mathbf{0}$) is an affine subspace, not a vector subspace.**

Here is why. Suppose $\mathbf{x}_p$ is one particular solution to $A\mathbf{x} = \mathbf{b}$. Then every solution can be written as:

$$\mathbf{x} = \mathbf{x}_p + \mathbf{v}, \quad \text{where } \mathbf{v} \in \ker(A)$$

In other words, the full solution set is:

$$\{\mathbf{x}_p + \mathbf{v} : \mathbf{v} \in \ker(A)\} = \mathbf{x}_p + \ker(A)$$

This is exactly an affine subspace: a particular solution $\mathbf{x}_p$ plus the kernel (null space) of $A$.

**Why this works:**

- If $A\mathbf{x}_p = \mathbf{b}$ and $A\mathbf{v} = \mathbf{0}$, then $A(\mathbf{x}_p + \mathbf{v}) = A\mathbf{x}_p + A\mathbf{v} = \mathbf{b} + \mathbf{0} = \mathbf{b}$ $\checkmark$

**Worked Example:**

Solve $A\mathbf{x} = \mathbf{b}$ where $A = \begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix} 3 \\ 6 \end{bmatrix}$.

**Step 1:** Find a particular solution $\mathbf{x}_p$.

The augmented matrix is $\left[\begin{array}{cc|c} 1 & 2 & 3 \\ 2 & 4 & 6 \end{array}\right]$.

$R_2 \to R_2 - 2R_1$: $\left[\begin{array}{cc|c} 1 & 2 & 3 \\ 0 & 0 & 0 \end{array}\right]$

So $x_1 + 2x_2 = 3$. Setting $x_2 = 0$ gives $x_1 = 3$. A particular solution is $\mathbf{x}_p = \begin{bmatrix} 3 \\ 0 \end{bmatrix}$.

**Step 2:** Find the kernel $\ker(A)$.

Solve $A\mathbf{x} = \mathbf{0}$: $x_1 + 2x_2 = 0$, so $x_1 = -2x_2$. Setting $x_2 = t$:

$$\ker(A) = \text{span}\left\{\begin{bmatrix} -2 \\ 1 \end{bmatrix}\right\}$$

**Step 3:** Write the full solution as an affine subspace.

$$\mathbf{x} = \begin{bmatrix} 3 \\ 0 \end{bmatrix} + t\begin{bmatrix} -2 \\ 1 \end{bmatrix}, \quad t \in \mathbb{R}$$

This is a line in $\mathbb{R}^2$ that passes through $\begin{bmatrix} 3 \\ 0 \end{bmatrix}$ with direction $\begin{bmatrix} -2 \\ 1 \end{bmatrix}$. It does not pass through the origin, so it is affine but not a vector subspace.

---

## Summary: Key Takeaways

### Systems and Matrices
- A system $A\mathbf{x} = \mathbf{b}$ has 0, 1, or infinitely many solutions
- The augmented matrix $[A \mid \mathbf{b}]$ is the starting point for Gaussian elimination
- The inverse $A^{-1}$ exists when $\det(A) \neq 0$; for $2 \times 2$: $A^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$

### Vector Spaces and Subspaces
- A vector space satisfies closure under addition and scalar multiplication
- Subspace test: contains $\mathbf{0}$, closed under $+$, closed under scalar multiplication
- Basis = linearly independent + spanning; dimension = number of basis vectors

### Rank, Kernel, and Image
- $\text{rank}(A) =$ number of pivots in REF
- $\ker(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$; $\text{Im}(A) = $ column space of $A$
- Rank-Nullity Theorem: $\text{nullity}(A) + \text{rank}(A) = n$

### Linear Mappings and Affine Spaces
- Linear mappings preserve addition and scalar multiplication
- Change of basis: $\tilde{A} = S^{-1}AS$
- Solution set of $A\mathbf{x} = \mathbf{b}$ is an affine subspace: $\mathbf{x}_p + \ker(A)$

---

## Practice Problems

### Problem 1: Solving a System via Gaussian Elimination

Solve the following system using Gaussian elimination:

$$x_1 + x_2 + 2x_3 = 9$$

$$2x_1 + 4x_2 - 3x_3 = 1$$

$$3x_1 + 6x_2 - 5x_3 = 0$$

### Problem 2: Finding the Inverse of a Matrix

Find the inverse of:

$$A = \begin{bmatrix} 3 & 5 \\ 1 & 2 \end{bmatrix}$$

Verify your answer by checking that $AA^{-1} = I$.

### Problem 3: Checking if a Set is a Subspace

Determine whether the following subset of $\mathbb{R}^3$ is a subspace:

$$U = \left\{\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} \in \mathbb{R}^3 : x_1 + 2x_2 - x_3 = 0\right\}$$

### Problem 4: Testing for Linear Independence

Determine whether the following vectors are linearly independent:

$$\mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \\ 2 \end{bmatrix}, \quad \mathbf{v}_2 = \begin{bmatrix} 0 \\ 1 \\ -1 \end{bmatrix}, \quad \mathbf{v}_3 = \begin{bmatrix} 2 \\ 1 \\ 3 \end{bmatrix}$$

### Problem 5: Finding a Basis and Dimension

Find a basis for the column space of the following matrix and state its dimension:

$$A = \begin{bmatrix} 1 & 3 & 5 \\ 2 & 6 & 10 \\ 1 & 1 & 3 \end{bmatrix}$$

### Problem 6: Computing the Rank

Compute the rank of:

$$M = \begin{bmatrix} 1 & 2 & 1 & 0 \\ 2 & 4 & 3 & 1 \\ 3 & 6 & 4 & 1 \end{bmatrix}$$

### Problem 7: Kernel and Image

For the matrix $A = \begin{bmatrix} 1 & 0 & -1 \\ 2 & 1 & 0 \\ -1 & 1 & 2 \end{bmatrix}$:

(a) Find the kernel of $A$.

(b) Find a basis for the image of $A$.

(c) Verify the Rank-Nullity Theorem.

### Problem 8: Affine Space -- Parametric Form of Solution Set

Find the general solution (in affine/parametric form) of:

$$\begin{bmatrix} 1 & 1 & 2 \\ 2 & 2 & 4 \end{bmatrix}\mathbf{x} = \begin{bmatrix} 4 \\ 8 \end{bmatrix}$$

Identify the particular solution, the kernel, and explain why the solution set is an affine subspace.

---

## Solutions

**Solution 1:**

Write the augmented matrix:

$$\left[\begin{array}{ccc|c} 1 & 1 & 2 & 9 \\ 2 & 4 & -3 & 1 \\ 3 & 6 & -5 & 0 \end{array}\right]$$

$R_2 \to R_2 - 2R_1$:

$$\left[\begin{array}{ccc|c} 1 & 1 & 2 & 9 \\ 0 & 2 & -7 & -17 \\ 3 & 6 & -5 & 0 \end{array}\right]$$

$R_3 \to R_3 - 3R_1$:

$$\left[\begin{array}{ccc|c} 1 & 1 & 2 & 9 \\ 0 & 2 & -7 & -17 \\ 0 & 3 & -11 & -27 \end{array}\right]$$

$R_3 \to R_3 - \frac{3}{2}R_2$:

$$\left[\begin{array}{ccc|c} 1 & 1 & 2 & 9 \\ 0 & 2 & -7 & -17 \\ 0 & 0 & -\frac{1}{2} & -\frac{3}{2} \end{array}\right]$$

$R_3 \to -2R_3$:

$$\left[\begin{array}{ccc|c} 1 & 1 & 2 & 9 \\ 0 & 2 & -7 & -17 \\ 0 & 0 & 1 & 3 \end{array}\right]$$

Back substitution: From row 3: $x_3 = 3$.

From row 2: $2x_2 - 7(3) = -17 \implies 2x_2 = 4 \implies x_2 = 2$.

From row 1: $x_1 + 2 + 2(3) = 9 \implies x_1 = 9 - 2 - 6 = 1$.

$$x_1 = 1, \quad x_2 = 2, \quad x_3 = 3$$

**Verify:** $1+2+6=9$ $\checkmark$, $2+8-9=1$ $\checkmark$, $3+12-15=0$ $\checkmark$.

---

**Solution 2:**

$$\det(A) = 3(2) - 5(1) = 6 - 5 = 1$$

$$A^{-1} = \frac{1}{1}\begin{bmatrix} 2 & -5 \\ -1 & 3 \end{bmatrix} = \begin{bmatrix} 2 & -5 \\ -1 & 3 \end{bmatrix}$$

**Verify:**

$$AA^{-1} = \begin{bmatrix} 3 & 5 \\ 1 & 2 \end{bmatrix}\begin{bmatrix} 2 & -5 \\ -1 & 3 \end{bmatrix} = \begin{bmatrix} 3(2)+5(-1) & 3(-5)+5(3) \\ 1(2)+2(-1) & 1(-5)+2(3) \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = I \quad \checkmark$$

---

**Solution 3:**

We apply the 3-step subspace test.

**Step 1 (Zero vector):** Let $x_1 = x_2 = x_3 = 0$. Then $0 + 2(0) - 0 = 0$. So $\mathbf{0} \in U$. $\checkmark$

**Step 2 (Closed under addition):** Let $\mathbf{u} = \begin{bmatrix} u_1 \\ u_2 \\ u_3 \end{bmatrix}$ and $\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}$ both be in $U$, so $u_1 + 2u_2 - u_3 = 0$ and $v_1 + 2v_2 - v_3 = 0$.

Then $\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1+v_1 \\ u_2+v_2 \\ u_3+v_3 \end{bmatrix}$ and:

$$(u_1+v_1) + 2(u_2+v_2) - (u_3+v_3) = (u_1+2u_2-u_3) + (v_1+2v_2-v_3) = 0 + 0 = 0$$

So $\mathbf{u} + \mathbf{v} \in U$. $\checkmark$

**Step 3 (Closed under scalar multiplication):** Let $\alpha \in \mathbb{R}$ and $\mathbf{u} \in U$.

$$\alpha u_1 + 2(\alpha u_2) - \alpha u_3 = \alpha(u_1 + 2u_2 - u_3) = \alpha \cdot 0 = 0$$

So $\alpha \mathbf{u} \in U$. $\checkmark$

**Conclusion:** $U$ is a subspace of $\mathbb{R}^3$. (Geometrically, it is a plane through the origin.)

---

**Solution 4:**

Form the matrix and row reduce:

$$\begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \\ 2 & -1 & 3 \end{bmatrix} \xrightarrow{R_3 - 2R_1} \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \\ 0 & -1 & -1 \end{bmatrix} \xrightarrow{R_3 + R_2} \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{bmatrix}$$

There are 2 pivots for 3 vectors. Column 3 has no pivot, meaning $\mathbf{v}_3$ depends on the others.

The vectors are **linearly dependent**.

From the RREF: $\mathbf{v}_3 = 2\mathbf{v}_1 + 1\mathbf{v}_2 = \begin{bmatrix} 2 \\ 1 \\ 3 \end{bmatrix}$. $\checkmark$

---

**Solution 5:**

Row reduce $A$:

$$\begin{bmatrix} 1 & 3 & 5 \\ 2 & 6 & 10 \\ 1 & 1 & 3 \end{bmatrix} \xrightarrow{R_2 - 2R_1} \begin{bmatrix} 1 & 3 & 5 \\ 0 & 0 & 0 \\ 1 & 1 & 3 \end{bmatrix} \xrightarrow{R_3 - R_1} \begin{bmatrix} 1 & 3 & 5 \\ 0 & 0 & 0 \\ 0 & -2 & -2 \end{bmatrix} \xrightarrow{R_2 \leftrightarrow R_3} \begin{bmatrix} 1 & 3 & 5 \\ 0 & -2 & -2 \\ 0 & 0 & 0 \end{bmatrix}$$

Pivots are in columns 1 and 2. A basis for the column space consists of the corresponding original columns of $A$:

$$\text{Basis} = \left\{\begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}, \begin{bmatrix} 3 \\ 6 \\ 1 \end{bmatrix}\right\}$$

**Dimension:** $\dim(\text{Col}(A)) = \text{rank}(A) = 2$.

---

**Solution 6:**

Row reduce $M$:

$$\begin{bmatrix} 1 & 2 & 1 & 0 \\ 2 & 4 & 3 & 1 \\ 3 & 6 & 4 & 1 \end{bmatrix} \xrightarrow{R_2 - 2R_1} \begin{bmatrix} 1 & 2 & 1 & 0 \\ 0 & 0 & 1 & 1 \\ 3 & 6 & 4 & 1 \end{bmatrix} \xrightarrow{R_3 - 3R_1} \begin{bmatrix} 1 & 2 & 1 & 0 \\ 0 & 0 & 1 & 1 \\ 0 & 0 & 1 & 1 \end{bmatrix} \xrightarrow{R_3 - R_2} \begin{bmatrix} 1 & 2 & 1 & 0 \\ 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

There are **2 pivots** (in columns 1 and 3).

$$\text{rank}(M) = 2$$

---

**Solution 7:**

**(a) Kernel:** Solve $A\mathbf{x} = \mathbf{0}$.

$$\begin{bmatrix} 1 & 0 & -1 \\ 2 & 1 & 0 \\ -1 & 1 & 2 \end{bmatrix} \xrightarrow{R_2 - 2R_1} \begin{bmatrix} 1 & 0 & -1 \\ 0 & 1 & 2 \\ -1 & 1 & 2 \end{bmatrix} \xrightarrow{R_3 + R_1} \begin{bmatrix} 1 & 0 & -1 \\ 0 & 1 & 2 \\ 0 & 1 & 1 \end{bmatrix} \xrightarrow{R_3 - R_2} \begin{bmatrix} 1 & 0 & -1 \\ 0 & 1 & 2 \\ 0 & 0 & -1 \end{bmatrix}$$

Three pivots for three unknowns, so the only solution to $A\mathbf{x} = \mathbf{0}$ is $\mathbf{x} = \mathbf{0}$.

$$\ker(A) = \{\mathbf{0}\}$$

**(b) Image:** Since $\text{rank}(A) = 3$ and $A \in \mathbb{R}^{3 \times 3}$, all three columns are linearly independent. A basis for the image is:

$$\text{Basis for Im}(A) = \left\{\begin{bmatrix} 1 \\ 2 \\ -1 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}, \begin{bmatrix} -1 \\ 0 \\ 2 \end{bmatrix}\right\}$$

So $\text{Im}(A) = \mathbb{R}^3$ (the mapping is surjective).

**(c) Verify Rank-Nullity:**

$$\dim(\ker(A)) + \dim(\text{Im}(A)) = 0 + 3 = 3 = n \quad \checkmark$$

Since $\ker(A) = \{\mathbf{0}\}$ (injective) and $\text{Im}(A) = \mathbb{R}^3$ (surjective), the mapping is **bijective** and $A$ is invertible.

---

**Solution 8:**

Write the augmented matrix and row reduce:

$$\left[\begin{array}{ccc|c} 1 & 1 & 2 & 4 \\ 2 & 2 & 4 & 8 \end{array}\right] \xrightarrow{R_2 - 2R_1} \left[\begin{array}{ccc|c} 1 & 1 & 2 & 4 \\ 0 & 0 & 0 & 0 \end{array}\right]$$

There is 1 pivot (in column 1) and 2 free variables ($x_2$ and $x_3$).

From the first row: $x_1 + x_2 + 2x_3 = 4$, so $x_1 = 4 - x_2 - 2x_3$.

**Particular solution:** Set the free variables to zero ($x_2 = 0, x_3 = 0$):

$$\mathbf{x}_p = \begin{bmatrix} 4 \\ 0 \\ 0 \end{bmatrix}$$

**Kernel:** Solve $A\mathbf{x} = \mathbf{0}$: $x_1 = -x_2 - 2x_3$, with $x_2 = s, x_3 = t$ free:

$$\ker(A) = \text{span}\left\{\begin{bmatrix} -1 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} -2 \\ 0 \\ 1 \end{bmatrix}\right\}$$

**General solution (affine/parametric form):**

$$\mathbf{x} = \underbrace{\begin{bmatrix} 4 \\ 0 \\ 0 \end{bmatrix}}_{\mathbf{x}_p} + s\underbrace{\begin{bmatrix} -1 \\ 1 \\ 0 \end{bmatrix}}_{\text{kernel direction 1}} + t\underbrace{\begin{bmatrix} -2 \\ 0 \\ 1 \end{bmatrix}}_{\text{kernel direction 2}}, \quad s, t \in \mathbb{R}$$

**Why is this an affine subspace?** The solution set does not pass through the origin (when $s = t = 0$, the solution is $\begin{bmatrix} 4 \\ 0 \\ 0 \end{bmatrix} \neq \mathbf{0}$). It is formed by taking one particular solution $\mathbf{x}_p$ and adding all elements of the kernel, which is a 2-dimensional vector subspace. The result is a 2-dimensional plane in $\mathbb{R}^3$, shifted away from the origin -- a classic affine subspace.

---

**Course:** Mathematics for Machine Learning
**Instructor:** Mohammed Alnemari

**Next:** Tutorial - Analytic Geometry
