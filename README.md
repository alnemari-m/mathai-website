# Mathematics for Machine Learning

**Course Website for Graduate-Level Mathematics of AI**

**Instructor:** Mohammed Alnemari
**Website:** https://alnemari-m.github.io/mathai-website/

---

## 📚 About This Course

This course provides a comprehensive mathematical foundation for understanding and implementing modern machine learning algorithms. It bridges the gap between theoretical mathematics and practical machine learning applications, equipping students with both rigorous mathematical understanding and hands-on coding skills.

### Course Philosophy

Building strong mathematical foundations combined with practical coding skills to prepare students for real-world machine learning applications. Each lecture follows a three-part approach:

1. **Concepts & Explanation** - Theoretical foundations and intuitive understanding
2. **Mathematical Examples & Tutorials** - Hands-on work through guided examples
3. **Python Implementation** - Practical coding exercises

---

## 🎯 What You'll Learn

### Part I: Mathematical Foundations (6 Topics)

**Topic 1: Linear Algebra**
Vectors, matrices, and their operations form the computational foundation for all machine learning algorithms.

**Topic 2: Analytic Geometry**
Understanding geometric interpretations of algebraic concepts provides intuition for high-dimensional data.

**Topic 3: Matrix Decomposition**
Techniques like eigendecomposition and SVD reveal underlying structure in data and enable dimensionality reduction.

**Topic 4: Vector Calculus**
Gradients and optimization concepts are critical for training machine learning models.

**Topic 5: Probability & Distributions**
Probabilistic reasoning and statistical foundations underpin modern machine learning theory.

**Topic 6: Optimization**
Methods for finding optimal solutions are central to model training and parameter estimation.

### Part II: Machine Learning Applications (4 Topics)

**Application 1: When Models Meet Data**
Introduction to the practical aspects of applying mathematical models to real-world datasets.

**Application 2: Dimensionality Reduction**
Using Principal Component Analysis (PCA) to reduce data complexity while preserving essential information.

**Application 3: Density Estimation**
Probabilistic approaches to understanding data distributions and clustering using Gaussian Mixture Models (GMM).

**Application 4: Classification**
Geometric and optimization-based methods for supervised learning using Support Vector Machines (SVM).

---

## 📊 Assessment Structure

- **Quizzes:** 40% (4-5 quizzes throughout course)
- **Midterm Examination:** 20% (Covers Part I: Mathematical Foundations)
- **Final Examination:** 30% (Comprehensive)
- **Reading & Review Papers:** 10% (4-5 research papers)

---

## 🎓 Target Audience

**Level:** Undergraduate/Graduate

**Prerequisites:**
- Basic linear algebra and vector calculus recommended
- Course structure allows for adjustment based on student background
- Suitable for students with varying levels of mathematical preparation

---

## 📖 Primary Textbooks

1. **Mathematics for Machine Learning**
   Deisenroth, Faisal, and Ong
   Cambridge University Press
   [mml-book.github.io](https://mml-book.github.io/)

2. **Convex Optimization**
   Boyd and Vandenberghe
   Cambridge University Press
   [stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/)

3. **Introduction to Probability**
   Bertsekas and Tsitsiklis
   Athena Scientific (2nd Ed.)

---

## 🌐 Website Structure

This repository contains the complete course website with:

- **Lectures:** Complete slides and course materials
- **Math Tutorials:** Detailed mathematical notes and derivations
- **Python Notebooks:** Hands-on coding exercises and implementations
- **Resources:** Textbook references and additional materials

```
├── docs/
│   ├── index.md              # Homepage
│   ├── lectures.md           # Lecture organization and materials
│   ├── tutorials.md          # Math tutorials
│   ├── notebooks.md          # Python notebooks
│   ├── pdfs/                # PDF lecture slides
│   ├── notebooks/           # Jupyter notebooks (.ipynb)
│   ├── tutorials/           # Tutorial markdown files
│   └── stylesheets/         # Custom CSS styling
├── mkdocs.yml               # MkDocs configuration
└── README.md               # This file
```

---

## 🚀 Quick Start for Students

### Access Course Materials

**Website:** https://alnemari-m.github.io/mathai-website/

1. **Lectures** - Download PDF slides for all topics
2. **Math Tutorials** - Read detailed mathematical explanations
3. **Python Notebooks** - Open in Google Colab or download for local use

### Using Jupyter Notebooks

**Option 1: Google Colab (Recommended - No Setup Required)**
- Click any notebook link on the website
- Open in Google Colab
- Run code directly in your browser

**Option 2: Local Jupyter**
```bash
pip install jupyter numpy scipy pandas matplotlib
jupyter notebook
```

---

## 💻 For Instructors: Local Development

### Setup

```bash
# Clone repository
git clone https://github.com/alnemari-m/mathai-website.git
cd mathai-website

# Install dependencies
pip install mkdocs pymdown-extensions

# Run local server
mkdocs serve
```

Visit http://127.0.0.1:8000

### Adding Content

**Add Lecture PDFs:**
```bash
# Place PDF in docs/pdfs/
cp lecture01.pdf docs/pdfs/
git add docs/pdfs/lecture01.pdf
git commit -m "Add Lecture 1 PDF"
git push
```

**Add Jupyter Notebooks:**
```bash
# Place notebook in docs/notebooks/
cp my_notebook.ipynb docs/notebooks/
git add docs/notebooks/my_notebook.ipynb
git commit -m "Add new notebook"
git push
```

**Add Math Tutorials:**
```bash
# Create markdown file in docs/tutorials/
nano docs/tutorials/Tutorial_02_Example.md
git add docs/tutorials/Tutorial_02_Example.md
git commit -m "Add Tutorial 2"
git push
```

### Deploy Updates

```bash
# Make changes to any files
git add .
git commit -m "Update course materials"
git push
```

GitHub Actions will automatically rebuild and deploy the website to GitHub Pages within 1-2 minutes.

---

## 🎨 Customization

The website uses a custom color scheme matching the course slides:
- **Primary Color:** Navy Blue (#003D6B)
- **Accent Color:** Bright Yellow (#FFC107)
- **Theme:** ReadTheDocs with custom CSS

To modify styling, edit `docs/stylesheets/custom.css`.

---

## 📧 Contact

**Instructor:** Mohammed Alnemari
**Email:** mnemari@gmail.com
**Office Hours:** Monday & Wednesday, 10:00 AM – 12:00 PM

---

## 📄 License

Educational use only. Course materials prepared by Mohammed Alnemari.

Some materials adapted from:
- Mathematics for Machine Learning (Deisenroth, Faisal, Ong)
- Yi, Yung (KAIST EE) course materials

---

## 🌟 Features

- ✅ Modern, responsive design
- ✅ Mobile-friendly navigation
- ✅ MathJax support for LaTeX equations
- ✅ Google Colab integration for notebooks
- ✅ Automatic deployment via GitHub Actions
- ✅ Professional styling matching course slides
- ✅ Fast loading and optimized for students
- ✅ Print-friendly CSS for offline reading

---

**Visit the course website:** https://alnemari-m.github.io/mathai-website/
