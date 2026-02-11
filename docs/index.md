<div class="hero-section" markdown="1">

# MATHEMATICS FOR MACHINE LEARNING

**Graduate Course • Spring 2026**

Mathematical foundations of artificial intelligence and machine learning - from theory to implementation.

**Prepared by: Mohammed Alnemari**

</div>

---

## 📢 ANNOUNCEMENTS

<div class="announcement-box" markdown="1">

### **Important Updates**

- **Week 1:** Course begins **January 20, 2026** - Welcome!
- **Office Hours:** Remember to schedule appointments via email
- **First Quiz:** Scheduled for **Week 3** - Linear Algebra fundamentals
- **Notebooks:** All Python implementations available in the Notebooks section

</div>

---

## TEXTBOOK AND REFERENCES

<div class="info-card" markdown="1">

### PRIMARY

**Mathematics for Machine Learning**
Deisenroth, Faisal, and Ong
Cambridge University Press
[mml-book.github.io](https://mml-book.github.io/)

**Convex Optimization**
Boyd and Vandenberghe
Cambridge University Press
[stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/)

**Introduction to Probability**
Bertsekas and Tsitsiklis
Athena Scientific (2nd Ed.)

</div>

---

## COURSE MATERIALS

- **[📄 LECTURES](lectures.md)** - Lecture slides and notes in PDF format
- **[📐 MATH TUTORIALS](tutorials.md)** - Detailed mathematical derivations and notes
- **[💻 NOTEBOOKS](notebooks.md)** - Python/Jupyter implementations and exercises

---

<div class="instructor-info" markdown="1">

## INSTRUCTOR & COURSE INFORMATION

**INSTRUCTOR**
Mohammed Alnemari

**TEACHING APPROACH**
- Part 1: Concepts & Explanation
- Part 2: Mathematical Examples & Tutorials
- Part 3: Python Implementation

**PHILOSOPHY**
Building strong mathematical foundations combined with practical coding skills to prepare students for real-world machine learning applications.

**OFFICE HOURS**
Monday & Wednesday
11:00 AM – 1:00 PM

**BY APPOINTMENT**
To schedule a meeting outside regular office hours, please contact via email.

<div class="contact-box" markdown="1">
**EMAIL:** mnemari@gmail.com
**COURSE WEBSITE:** ut.edu.sa/mathml
</div>

</div>

---

## COURSE STRUCTURE & ASSESSMENT

### 6 Core Chapters (Focus)

| # | Topic | Description |
|---|-------|-------------|
| **1** | **Linear Algebra** | Vectors, matrices, and operations |
| **2** | **Analytic Geometry** | Geometric interpretations |
| **3** | **Matrix Decomposition** | Eigendecomposition & SVD |
| **4** | **Vector Calculus** | Gradients & optimization |
| **5** | **Probability & Distributions** | Statistical foundations |
| **6** | **Optimization** | Model training & parameter estimation |

### Assessment Breakdown

| Component | Weight | Details |
|-----------|--------|---------|
| **Quizzes** | **40%** | 4-5 quizzes throughout course |
| **Midterm Examination** | **20%** | - |
| **Final Examination** | **30%** | - |
| **Reading & Review Papers** | **10%** | 4-5 papers |
| **TOTAL** | **100%** | - |

---

## EXAM QUESTION PHILOSOPHY & EXAMPLES

<div class="info-card" markdown="1">

### **Deep Understanding Through Challenging Problems**

Exam questions are designed to test **deep conceptual understanding** rather than memorization. Each question builds progressively, connecting theory, computation, and insight.

**Question Structure:**
- Multi-part problems that build on each other
- Require synthesis of multiple concepts
- Mix analytical work with computational techniques
- Based strictly on material covered in lectures

</div>

### Sample Exam Question

This question exemplifies the depth and rigor expected in exams, requiring synthesis of multiple concepts and progressive problem-solving.

---

#### **Linear Independence, Basis, and Rank**

Consider the matrix $\mathbf{A} \in \mathbb{R}^{4 \times 4}$ with column vectors:
$$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 & 5 \\ 2 & 1 & 3 & 4 \\ 1 & 0 & 1 & 1 \\ 0 & 1 & 1 & 2 \end{bmatrix}$$

**(a)** Use Gaussian elimination to determine $\text{rank}(\mathbf{A})$ by reducing to row echelon form. Identify which columns form a basis for the column space $\mathcal{C}(\mathbf{A})$ and express each remaining column as a linear combination of these basis columns.

**(b)** Find a basis for the null space $\mathcal{N}(\mathbf{A}) = \{\mathbf{x} \in \mathbb{R}^4 : \mathbf{Ax} = \mathbf{0}\}$ by solving the homogeneous system. Verify that $\text{rank}(\mathbf{A}) + \dim(\mathcal{N}(\mathbf{A})) = 4$.

**(c)** Prove the following general statement: If $\mathbf{A} \in \mathbb{R}^{m \times n}$ has rank $r$, then any set of $r+1$ columns must be linearly dependent. Apply this to show that columns 1, 2, and 4 of your matrix cannot all be part of a linearly independent set if $r < 3$.

**(d)** Compute the row space $\mathcal{R}(\mathbf{A})$ by finding a basis for the row space from the row echelon form. Show that $\dim(\mathcal{R}(\mathbf{A})) = \dim(\mathcal{C}(\mathbf{A}))$.

**(e)** Demonstrate that $\mathcal{N}(\mathbf{A})$ and $\mathcal{R}(\mathbf{A})$ are orthogonal subspaces by verifying that every vector in $\mathcal{N}(\mathbf{A})$ is orthogonal to every row of $\mathbf{A}$. What does this tell you about the decomposition $\mathbb{R}^4 = \mathcal{R}(\mathbf{A}^T) \oplus \mathcal{N}(\mathbf{A})$?

**(f)** Consider the augmented matrix $[\mathbf{A} | \mathbf{b}]$ where $\mathbf{b} = [1, 1, 0, 1]^T$. Without solving the system, use your knowledge of $\mathcal{C}(\mathbf{A})$ to determine whether $\mathbf{b} \in \mathcal{C}(\mathbf{A})$. If $\mathbf{b} \notin \mathcal{C}(\mathbf{A})$, decompose $\mathbf{b}$ as $\mathbf{b} = \mathbf{b}_{\parallel} + \mathbf{b}_{\perp}$ where $\mathbf{b}_{\parallel} \in \mathcal{C}(\mathbf{A})$ and $\mathbf{b}_{\perp} \perp \mathcal{C}(\mathbf{A})$.

---

## LECTURE STRUCTURE: THREE-PART APPROACH

### PART 1: Concepts & Explanation
Theoretical foundations and intuitive understanding of the topic.
- Key definitions
- Conceptual framework
- Intuitive explanations
- Real-world context

### PART 2: Mathematical Examples & Tutorials
Hands-on mathematical work with students through guided examples.
- Worked examples
- Step-by-step solutions
- Interactive tutorials
- Mathematical derivations

### PART 3: Python Implementation
Practical coding examples implementing concepts from the lecture.
- Code examples
- Implementation details
- Practical applications
- Coding exercises

---

## NOTATION CONVENTIONS

| Category | Notation |
|----------|----------|
| **SCALARS** | $a, b, c, \alpha, \beta, \gamma$ |
| **VECTORS** | $\mathbf{x}, \mathbf{y}, \mathbf{z}$ |
| **MATRICES** | $\mathbf{X}, \mathbf{Y}, \mathbf{Z}$ |
| **SETS** | $A, B, C$ |
| **NUMBER SYSTEMS** | $\mathbb{R}, \mathbb{C}, \mathbb{Z}, \mathbb{N}, \mathbb{R}^n$ |
| **PROBABILITY** | $p(\cdot), P[\cdot]$ |

---

<div class="signature" markdown="1">

**MOHAMMED ALNEMARI**
*MATHEMATICS FOR MACHINE LEARNING • SPRING 2026*

**ENJOY YOUR LEARNING JOURNEY**

Welcome to Mathematics for Machine Learning.
Building the foundation for your future algorithms.

</div>

---

<div class="last-updated" markdown="1">
**Last Updated:** February 10, 2026
</div>
