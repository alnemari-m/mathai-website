<div id="password-gate">
  <div style="min-height: 100vh; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #0052D4 0%, #00D9A3 100%);">
    <div style="background: white; padding: 3em; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.2); max-width: 400px; text-align: center;">
      <h2 style="color: #0052D4; margin-top: 0; border: none; padding: 0;">🔒 Protected Content</h2>
      <p style="color: #666; margin-bottom: 2em;">Enter password to access mathematical animations and visualizations</p>
      <input type="password" id="password-input" placeholder="Enter password" style="width: 100%; padding: 12px; border: 2px solid #0052D4; border-radius: 6px; font-size: 1em; margin-bottom: 1em;" />
      <button onclick="checkPassword()" style="width: 100%; padding: 12px; background: #FF5733; color: white; border: none; border-radius: 6px; font-size: 1em; font-weight: 600; cursor: pointer; transition: all 0.3s;">
        Unlock Content
      </button>
      <p id="error-message" style="color: #FF5733; margin-top: 1em; display: none;">❌ Incorrect password. Please try again.</p>
    </div>
  </div>
</div>

<div id="protected-content" style="display: none;" markdown="1">

# Mathematical Animations & Visualizations

**Advanced Learning Tools for Deep Mathematical Understanding**

---

## 🎬 About This Section

This section provides interactive mathematical animations and visualizations to help you develop **intuitive understanding** of abstract concepts covered in the course.

**Tools Used:**
- **Manim (Mathematical Animation Engine)** - Created by Grant Sanderson (3Blue1Brown)
- **GeoGebra** - Interactive geometry and algebra
- **Desmos** - Graphing calculator
- **Python Visualizations** - Custom interactive plots

---

## 📚 Why Mathematical Animations?

<div class="info-card" markdown="1">

### **The Power of Visualization**

Mathematical animations help you:
- **See** abstract concepts in action
- **Understand** geometric interpretations
- **Remember** complex theorems through visual memory
- **Build** intuition before diving into proofs
- **Connect** different mathematical concepts

*"Mathematics is not about numbers, equations, or algorithms: it is about understanding."* - William Paul Thurston

</div>

---

## 🎯 Available Animations

### **Topic 1: Linear Algebra Fundamentals**

#### **Animation 1.1: Vector Addition and Scaling**
- Visualizing vector addition (tip-to-tail method)
- Scalar multiplication and direction
- Linear combinations in 2D and 3D

**📹 Watch:** [Vector Operations Animation](animations/videos/01_vector_operations.mp4)
**💻 Code:** [Manim Source Code](animations/code/01_vector_operations.py)
**📝 Exercise:** Try different scalars and see what happens!

---

#### **Animation 1.2: Matrix Transformations**
- Matrices as linear transformations
- Understanding det(A) geometrically (area scaling)
- Rotation, scaling, shearing, and reflection matrices

**📹 Watch:** [Matrix Transformations](animations/videos/02_matrix_transforms.mp4)
**💻 Interactive:** [GeoGebra - Matrix Transformations](https://www.geogebra.org/m/matrix-transforms)

---

#### **Animation 1.3: Column Space and Null Space**
- Visualizing column space as "reachable" vectors
- Null space as vectors that get squashed to zero
- Rank-nullity theorem in action

**📹 Watch:** [Column Space & Null Space](animations/videos/03_spaces.mp4)

---

### **Topic 2: Analytic Geometry**

#### **Animation 2.1: Inner Products and Projections**
- Geometric interpretation of dot product
- Vector projections animated
- Orthogonality visualization

**📹 Watch:** [Inner Products](animations/videos/04_inner_products.mp4)

---

#### **Animation 2.2: Norms and Distances**
- Different norms (L1, L2, L∞) visualized
- Unit balls in different norms
- Distance metrics comparison

**📹 Watch:** [Norms and Metrics](animations/videos/05_norms.mp4)

---

### **Topic 3: Matrix Decomposition**

#### **Animation 3.1: Eigenvalues and Eigenvectors**
- What eigenvalues and eigenvectors really mean
- Visualizing Av = λv
- Diagonalization process animated

**📹 Watch:** [Eigendecomposition](animations/videos/06_eigenvalues.mp4)
**🌟 Featured:** This is one of the most important visualizations in the course!

---

#### **Animation 3.2: Singular Value Decomposition (SVD)**
- Breaking down any matrix into rotation + scaling + rotation
- Geometric interpretation of SVD
- Applications to image compression

**📹 Watch:** [SVD Visualization](animations/videos/07_svd.mp4)
**💻 Interactive:** [SVD Image Compression Demo](animations/demos/svd_compression.html)

---

### **Topic 4: Vector Calculus**

#### **Animation 4.1: Gradients and Directional Derivatives**
- Gradient as "direction of steepest ascent"
- Visualizing gradient descent
- Contour plots and gradient fields

**📹 Watch:** [Gradient Visualization](animations/videos/08_gradients.mp4)

---

#### **Animation 4.2: The Chain Rule**
- Chain rule in action with function composition
- Backpropagation visualization
- Computational graphs animated

**📹 Watch:** [Chain Rule & Backprop](animations/videos/09_chain_rule.mp4)

---

### **Topic 5: Probability & Distributions**

#### **Animation 5.1: Common Probability Distributions**
- Gaussian distribution evolving
- Central Limit Theorem animated
- Comparing different distributions

**📹 Watch:** [Probability Distributions](animations/videos/10_distributions.mp4)

---

#### **Animation 5.2: Bayesian Inference**
- Prior to posterior animation
- Bayes' theorem visualized
- Updating beliefs with evidence

**📹 Watch:** [Bayesian Inference](animations/videos/11_bayes.mp4)

---

### **Topic 6: Optimization**

#### **Animation 6.1: Gradient Descent Variants**
- Batch, mini-batch, and stochastic gradient descent
- Momentum and acceleration visualized
- Adam optimizer animation

**📹 Watch:** [Optimization Algorithms](animations/videos/12_optimization.mp4)
**💻 Interactive:** [Gradient Descent Playground](animations/demos/gd_playground.html)

---

#### **Animation 6.2: Convex vs Non-Convex Optimization**
- Convex functions and global minima
- Non-convex landscapes and local minima
- Saddle points visualization

**📹 Watch:** [Loss Landscapes](animations/videos/13_landscapes.mp4)

---

## 🛠️ Create Your Own Animations

### **Getting Started with Manim**

Manim is a powerful Python library for creating mathematical animations.

#### **Installation**

```bash
# Install Manim Community Edition
pip install manim

# Install dependencies
pip install numpy scipy matplotlib
```

#### **Your First Animation**

```python
from manim import *

class VectorScene(Scene):
    def construct(self):
        # Create coordinate plane
        plane = NumberPlane()

        # Create vectors
        v1 = Vector([2, 1], color=BLUE)
        v2 = Vector([1, 2], color=RED)
        v3 = Vector([3, 3], color=GREEN)

        # Labels
        v1_label = MathTex("\\vec{v}", color=BLUE).next_to(v1, RIGHT)
        v2_label = MathTex("\\vec{w}", color=RED).next_to(v2, LEFT)
        v3_label = MathTex("\\vec{v} + \\vec{w}", color=GREEN).next_to(v3, UP)

        # Animate
        self.add(plane)
        self.play(Create(v1), Write(v1_label))
        self.play(Create(v2), Write(v2_label))
        self.wait()
        self.play(Transform(v1.copy(), v3), Create(v3), Write(v3_label))
        self.wait(2)

# Render with: manim -pql your_file.py VectorScene
```

#### **Manim Resources**

- **Official Docs:** [https://docs.manim.community/](https://docs.manim.community/)
- **3Blue1Brown Channel:** [YouTube](https://www.youtube.com/c/3blue1brown)
- **Manim Tutorial:** [Community Guide](https://docs.manim.community/en/stable/tutorials.html)

---

### **Alternative Tools**

#### **1. GeoGebra** (Interactive Geometry)
- Great for exploring transformations
- No coding required
- [Download GeoGebra](https://www.geogebra.org/download)

#### **2. Desmos** (Graphing Calculator)
- Excellent for function visualization
- Web-based, no installation
- [Desmos Calculator](https://www.desmos.com/calculator)

#### **3. Python + Matplotlib** (Custom Visualizations)

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Example: Visualizing matrix transformation
def plot_transformation(A):
    # Create unit square
    square = np.array([[0, 1, 1, 0, 0],
                       [0, 0, 1, 1, 0]])

    # Transform
    transformed = A @ square

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(square[0], square[1], 'b-', linewidth=2)
    ax1.set_title('Original')
    ax1.grid(True)
    ax1.axis('equal')

    ax2.plot(transformed[0], transformed[1], 'r-', linewidth=2)
    ax2.set_title('After Transformation')
    ax2.grid(True)
    ax2.axis('equal')

    plt.show()

# Example usage
A = np.array([[2, 0],
              [0, 1]])  # Scaling matrix
plot_transformation(A)
```

---

## 📖 Recommended Video Series

### **3Blue1Brown - Essence of Linear Algebra**
The best visual introduction to linear algebra ever made.

1. **Vectors, what even are they?** - [Watch](https://www.youtube.com/watch?v=fNk_zzaMoSs)
2. **Linear combinations, span, and basis vectors** - [Watch](https://www.youtube.com/watch?v=k7RM-ot2NWY)
3. **Linear transformations and matrices** - [Watch](https://www.youtube.com/watch?v=kYB8IZa5AuE)
4. **Matrix multiplication as composition** - [Watch](https://www.youtube.com/watch?v=XkY2DOUCWMU)
5. **The determinant** - [Watch](https://www.youtube.com/watch?v=Ip3X9LOh2dk)
6. **Inverse matrices, column space and null space** - [Watch](https://www.youtube.com/watch?v=uQhTuRlWMxw)
7. **Dot products and duality** - [Watch](https://www.youtube.com/watch?v=LyGKycYT2v0)
8. **Cross products** - [Watch](https://www.youtube.com/watch?v=eu6i7WJeinw)
9. **Change of basis** - [Watch](https://www.youtube.com/watch?v=P2LTAUO1TdA)
10. **Eigenvectors and eigenvalues** - [Watch](https://www.youtube.com/watch?v=PFDu9oVAE-g)

### **3Blue1Brown - Essence of Calculus**
Visual approach to understanding derivatives, integrals, and gradients.

### **StatQuest** - Statistics and ML Concepts
Great for probability and statistical learning visualizations.

---

## 💡 How to Use This Section

### **Study Strategy:**

1. **Before Lecture:** Watch the relevant animation to build intuition
2. **During Lecture:** Connect the visuals to mathematical definitions
3. **After Lecture:** Create your own animations to test understanding
4. **Before Exam:** Review animations for quick intuitive recall

### **Active Learning Tips:**

- **Pause and Predict:** Stop the animation and guess what happens next
- **Vary Parameters:** Change values and observe the effects
- **Code Your Own:** Try recreating animations from scratch
- **Teach Someone:** Explain the visual to a classmate

---

## 🎓 Student Projects

Want to earn bonus points? Create your own mathematical animation!

### **Project Ideas:**

1. Visualize the Jacobi or Gauss-Seidel iteration process
2. Animate the convergence of gradient descent on different functions
3. Show how PCA finds principal components
4. Visualize the EM algorithm for Gaussian Mixture Models
5. Animate SVM decision boundaries with different kernels

### **Submission:**

- Email your Manim code to: mnemari@gmail.com
- Include a 1-page explanation of what you visualized
- Best submissions will be featured in this section!
- **Bonus:** +3% on final grade for exceptional work

---

## 📁 Download All Animations

**Full Animation Pack (2.5 GB):**
[📥 Download ZIP](animations/MathML_Animations_Complete.zip)

**Individual Topics:**
- [Topic 1: Linear Algebra](animations/Topic1_LinearAlgebra.zip) - 450 MB
- [Topic 2: Analytic Geometry](animations/Topic2_Geometry.zip) - 320 MB
- [Topic 3: Matrix Decomposition](animations/Topic3_Decomposition.zip) - 380 MB
- [Topic 4: Vector Calculus](animations/Topic4_Calculus.zip) - 290 MB
- [Topic 5: Probability](animations/Topic5_Probability.zip) - 340 MB
- [Topic 6: Optimization](animations/Topic6_Optimization.zip) - 420 MB

---

## 🔐 Access Information

**This section is restricted to enrolled students only.**

If you need the password:
- Attend the first lecture
- Check your email (sent after enrollment)
- Contact during office hours

**Keep the password confidential** - sharing it violates academic integrity policies.

---

<div class="signature" markdown="1">

**MOHAMMED ALNEMARI**
*MATHEMATICS FOR MACHINE LEARNING • SPRING 2026*

**Visualize. Understand. Master.**

</div>

---

<div class="last-updated" markdown="1">
**Last Updated:** January 26, 2026
**Next Animation Upload:** Week 3 (SVD Deep Dive)
</div>

</div>

<script>
(function() {
  // Simple password protection
  const CORRECT_PASSWORD = "malak2024";

  window.checkPassword = function() {
    const input = document.getElementById('password-input');
    const errorMsg = document.getElementById('error-message');

    if (!input) return;

    if (input.value === CORRECT_PASSWORD) {
      // Store authentication in session
      sessionStorage.setItem('animations_auth', 'true');
      showContent();
    } else {
      errorMsg.style.display = 'block';
      input.value = '';
      input.focus();
    }
  };

  function showContent() {
    const gate = document.getElementById('password-gate');
    const content = document.getElementById('protected-content');

    if (gate) gate.style.display = 'none';
    if (content) content.style.display = 'block';
  }

  function checkAuth() {
    if (sessionStorage.getItem('animations_auth') === 'true') {
      showContent();
    }
  }

  // Run immediately
  checkAuth();

  // Also run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
      checkAuth();

      const input = document.getElementById('password-input');
      if (input) {
        input.addEventListener('keypress', function(e) {
          if (e.key === 'Enter') {
            window.checkPassword();
          }
        });

        // Focus input after a small delay
        setTimeout(function() {
          input.focus();
        }, 100);
      }
    });
  } else {
    checkAuth();

    const input = document.getElementById('password-input');
    if (input) {
      input.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
          window.checkPassword();
        }
      });
      input.focus();
    }
  }
})();
</script>

<style>
#password-input:focus {
  outline: none;
  border-color: #00D9A3;
  box-shadow: 0 0 0 3px rgba(0, 217, 163, 0.2);
}

button:hover {
  background: #E64A2E !important;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(255, 87, 51, 0.3);
}
</style>
