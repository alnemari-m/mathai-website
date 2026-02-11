# Additional Exam Questions Archive

This file contains additional challenging exam questions for the Mathematics for Machine Learning course. Each question follows a multi-part progressive structure requiring deep conceptual understanding.

---

## Question 1: Systems of Linear Equations and Matrices

Consider the system of linear equations $\mathbf{Ax} = \mathbf{b}$ where $\mathbf{A} \in \mathbb{R}^{m \times n}$.

**(a)** Prove that if $\mathbf{Ax} = \mathbf{b}$ has two distinct solutions $\mathbf{x}_1$ and $\mathbf{x}_2$, then it has infinitely many solutions. Show that the solution set forms an affine space and determine its dimension in terms of $\text{rank}(\mathbf{A})$.

**(b)** Consider the parametric system:
$$\begin{cases} 2x_1 + 4x_2 - 2x_3 = \alpha \\ 4x_1 + 9x_2 + 3x_3 = 8 \\ 2x_1 + 3x_2 + x_3 = 1 \end{cases}$$

For which values of $\alpha$ does this system have: (i) no solution, (ii) unique solution, (iii) infinitely many solutions? Use Gaussian elimination to fully analyze the structure, showing all row operations explicitly.

**(c)** For the case where infinitely many solutions exist, construct the complete solution set in the form $\mathbf{x} = \mathbf{x}_p + \text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$. Prove that your vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ are linearly independent and form a basis for the null space of $\mathbf{A}$.

**(d)** Compute $\mathbf{A}^T\mathbf{A}$ and determine its null space. Show that $\text{null}(\mathbf{A}^T\mathbf{A}) = \text{null}(\mathbf{A})$ and explain why $\mathbf{A}^T\mathbf{A}$ is invertible if and only if $\mathbf{A}$ has full column rank.

**(e)** Now consider the modified system $\mathbf{A}^T\mathbf{Ax} = \mathbf{A}^T\mathbf{b}$ (the normal equations). Prove that this system is always consistent regardless of whether $\mathbf{Ax} = \mathbf{b}$ has a solution. What is the geometric interpretation of the solution to the normal equations when $\mathbf{Ax} = \mathbf{b}$ is inconsistent?

**(f)** Using your results from parts (b)-(e), compute the least-squares solution for the inconsistent case you identified in part (b). Verify your answer by computing the residual $\|\mathbf{Ax} - \mathbf{b}\|^2$ and showing it is minimized.

---

## Question 2: Vector Spaces and Subspaces

Let $V = \mathbb{R}^3$ and consider two sets:
$$U_1 = \{(x, y, z) \in \mathbb{R}^3 : 2x - y + z = 0\}, \quad U_2 = \{(x, y, z) \in \mathbb{R}^3 : x + y - z = 0\}$$

**(a)** Prove that both $U_1$ and $U_2$ are subspaces of $\mathbb{R}^3$ by verifying the subspace axioms. For each subspace, find a basis and determine the dimension.

**(b)** Define the intersection $U_1 \cap U_2$ and the sum $U_1 + U_2 = \{\mathbf{u}_1 + \mathbf{u}_2 : \mathbf{u}_1 \in U_1, \mathbf{u}_2 \in U_2\}$. Prove that both are subspaces of $\mathbb{R}^3$. Find explicit bases for $U_1 \cap U_2$ and $U_1 + U_2$.

**(c)** Verify the dimension formula: $\dim(U_1 + U_2) = \dim(U_1) + \dim(U_2) - \dim(U_1 \cap U_2)$ using your results from part (b). Explain geometrically what this formula represents in terms of planes and lines in $\mathbb{R}^3$.

**(d)** Now consider the affine spaces $L_1 = \{(x, y, z) : 2x - y + z = 3\}$ and $L_2 = \{(x, y, z) : x + y - z = 1\}$. Express each as $L_i = \mathbf{x}_i + U_i$ for appropriate support points and direction spaces.

**(e)** Determine whether $L_1$ and $L_2$ intersect. If they do, find all points of intersection. If they don't, find the minimum distance between the two affine spaces and the closest points on each.

**(f)** Prove that if $L = \mathbf{x}_0 + U$ is an affine subspace with direction space $U$, then the choice of support point $\mathbf{x}_0$ is not unique, but any two support points differ by a vector in $U$. Show that the direction space $U$ is uniquely determined by $L$.

---

## Question 4: Linear Mappings and Transformation Matrices

Consider two linear mappings: $\Phi : \mathbb{R}^3 \to \mathbb{R}^3$ defined by $\Phi(x, y, z) = (x + 2y, y + z, x + y + z)$ and $\Psi : \mathbb{R}^3 \to \mathbb{R}^2$ defined by $\Psi(x, y, z) = (x - y, 2y - z)$.

**(a)** For the mapping $\Phi$, find the transformation matrix $\mathbf{A}_\Phi$ with respect to the standard basis. Determine $\text{ker}(\Phi)$, $\text{Im}(\Phi)$, and state whether $\Phi$ is injective, surjective, or bijective. Justify each conclusion.

**(b)** If $\Phi$ is invertible, compute $\Phi^{-1}$ by finding the inverse matrix $\mathbf{A}_\Phi^{-1}$. Verify your answer by showing that $\Phi^{-1}(\Phi(\mathbf{v})) = \mathbf{v}$ for an arbitrary test vector.

**(c)** For the composition $\Psi \circ \Phi : \mathbb{R}^3 \to \mathbb{R}^2$, find the transformation matrix $\mathbf{A}_{\Psi \circ \Phi}$ in two ways: (i) directly from the definition $(\Psi \circ \Phi)(\mathbf{x}) = \Psi(\Phi(\mathbf{x}))$, and (ii) using matrix multiplication $\mathbf{A}_\Psi \mathbf{A}_\Phi$. Verify they produce the same result.

**(d)** Prove the following: $\text{ker}(\Psi \circ \Phi) = \Phi^{-1}(\text{ker}(\Psi))$. Use this to find a basis for $\text{ker}(\Psi \circ \Phi)$ from your knowledge of $\text{ker}(\Psi)$ and $\Phi$.

**(e)** Show that $\text{Im}(\Psi \circ \Phi) \subseteq \text{Im}(\Psi)$ and determine when equality holds. For this specific case, is $\text{Im}(\Psi \circ \Phi) = \text{Im}(\Psi)$? Justify your answer using rank considerations.

**(f)** Consider the restricted mapping $\Phi|_U : U \to \mathbb{R}^3$ where $U = \text{span}\{(1,0,0), (0,1,1)\}$. Find $\text{Im}(\Phi|_U)$ and determine whether $\Phi|_U$ is injective. Compare $\dim(\text{Im}(\Phi|_U))$ with $\dim(U)$ and explain the relationship.

---

## Question 5: Basis Change and Coordinate Transformations

In $\mathbb{R}^3$, consider three bases:
$$B = \{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\} \text{ (standard)}, \quad B' = \left\{\begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}, \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}\right\}, \quad B'' = \left\{\begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}, \begin{bmatrix} 2 \\ 1 \\ 1 \end{bmatrix}, \begin{bmatrix} 1 \\ 1 \\ 2 \end{bmatrix}\right\}$$

**(a)** Verify that $B'$ and $B''$ are bases for $\mathbb{R}^3$. Find the change of basis matrices $\mathbf{S}_{B \to B'}$ and $\mathbf{S}_{B \to B''}$ such that $[\mathbf{v}]_B = \mathbf{S}_{B \to B'}[\mathbf{v}]_{B'}$ for any vector $\mathbf{v}$.

**(b)** Prove that $(\mathbf{S}_{B \to B'})^{-1} = \mathbf{S}_{B' \to B}$ and compute $\mathbf{S}_{B' \to B}$ explicitly. Then find the change of basis matrix $\mathbf{S}_{B' \to B''}$ using the composition property: $\mathbf{S}_{B' \to B''} = \mathbf{S}_{B \to B''}\mathbf{S}_{B' \to B}$.

**(c)** Consider the linear transformation $\Phi : \mathbb{R}^3 \to \mathbb{R}^3$ defined by $\Phi(x, y, z) = (2x + y, x + 2y + z, y + 2z)$. Find the matrix representations $[\Phi]_B$, $[\Phi]_{B'}$, and $[\Phi]_{B''}$.

**(d)** Verify the similarity transformation formula: $[\Phi]_{B'} = \mathbf{S}_{B' \to B}[\Phi]_B\mathbf{S}_{B \to B'}$. Show that $\text{det}([\Phi]_B) = \text{det}([\Phi]_{B'}) = \text{det}([\Phi]_{B''})$ and explain why the determinant is basis-independent.

**(e)** Prove that if two matrices $\mathbf{A}$ and $\tilde{\mathbf{A}}$ represent the same linear transformation in different bases, then they have the same rank, nullity, and eigenvalues (eigenvalues introduced here conceptually). Verify this for $[\Phi]_B$ and $[\Phi]_{B'}$.

**(f)** Find a basis $B^*$ in which the matrix representation $[\Phi]_{B^*}$ is diagonal or as simple as possible. Describe your strategy and compute $[\Phi]_{B^*}$ explicitly. What geometric insight does this simplified representation provide about $\Phi$?

---

## Question 6: Affine Spaces and Geometric Applications

Consider three affine subspaces in $\mathbb{R}^4$:
$$L_1: x_1 + x_2 - x_3 + x_4 = 2, \quad L_2: 2x_1 - x_2 + x_3 = 1, \quad L_3: x_1 + 3x_2 - 2x_3 + x_4 = 5$$

**(a)** For each affine space $L_i$, express it in the parametric form $L_i = \mathbf{a}_i + U_i$ where $\mathbf{a}_i$ is a support point and $U_i$ is the direction space. Find a basis for each direction space and determine $\dim(U_i)$.

**(b)** Investigate the intersection $L_1 \cap L_2$. Determine whether the intersection is empty, a point, a line, or a higher-dimensional affine space. If non-empty, express $L_1 \cap L_2$ in parametric form and find its dimension.

**(c)** Prove the following general result: If $L = \mathbf{a} + U$ and $L' = \mathbf{b} + V$ are affine subspaces, then $L \cap L'$ is either empty or an affine subspace with direction space contained in $U \cap V$. Apply this to find the dimension of $L_1 \cap L_2 \cap L_3$ without computing it explicitly.

**(d)** For two non-intersecting affine spaces $L_1$ and $L_2$, define the distance $d(L_1, L_2) = \inf\{\|\mathbf{x} - \mathbf{y}\| : \mathbf{x} \in L_1, \mathbf{y} \in L_2\}$. If $L_1 \cap L_2 = \emptyset$, find the minimum distance between $L_1$ and $L_2$ and the closest points $\mathbf{p}_1 \in L_1$ and $\mathbf{p}_2 \in L_2$ achieving this distance.

**(e)** Consider the affine mapping $f : \mathbb{R}^4 \to \mathbb{R}^4$ defined by $f(\mathbf{x}) = \mathbf{Ax} + \mathbf{b}$ where
$$\mathbf{A} = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 1 \\ 1 \\ 0 \\ 0 \end{bmatrix}$$

Show that $f$ maps affine subspaces to affine subspaces. Compute $f(L_1)$ and determine its dimension. Verify that $\dim(f(L_1)) \leq \dim(L_1)$ with equality if and only if $\mathbf{A}$ is injective when restricted to the direction space $U_1$.

**(f)** Prove that for any affine space $L = \mathbf{a} + U$ in $\mathbb{R}^n$ and any point $\mathbf{p} \in \mathbb{R}^n$, there exists a unique point $\mathbf{q} \in L$ closest to $\mathbf{p}$ (i.e., minimizing $\|\mathbf{p} - \mathbf{q}\|$). Show that $\mathbf{q}$ is characterized by the condition $(\mathbf{p} - \mathbf{q}) \perp U$. Apply this to find the point on $L_1$ closest to the origin.

---

**Note:** These questions are designed for deep conceptual understanding and require synthesis of multiple topics from linear algebra. They follow a progressive multi-part structure where each part builds on previous results.
