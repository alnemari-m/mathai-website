# Quick Start Guide

## 1. Add Your PDFs

Convert your markdown lectures to PDF:

```bash
cd ~/Documents/mathai

# Install pandoc if needed
sudo apt install pandoc texlive-latex-base

# Convert each lecture
pandoc LECTURE_01_Vector_Spaces_And_Data_Manifolds.md -o lecture01.pdf
pandoc LECTURE_02_Linear_Maps_And_Operators.md -o lecture02.pdf
pandoc LECTURE_03_Matrix_Decompositions_SVD.md -o lecture03.pdf
# ... etc

# Copy to website
cp lecture*.pdf ~/mathai-website/docs/pdfs/
```

## 2. Add Your Notebooks

If you have Jupyter notebooks:

```bash
# Copy notebooks to website
cp your_notebook.ipynb ~/mathai-website/docs/notebooks/
```

## 3. Test Locally

```bash
cd ~/mathai-website
mkdocs serve
# Open http://127.0.0.1:8000
```

## 4. Deploy to GitHub

```bash
# Create repo on GitHub first at: https://github.com/new
# Name it: mathai-website

# Then push
cd ~/mathai-website
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/mathai-website.git
git push -u origin main
```

Done! Your site will be live at:
https://YOUR_USERNAME.github.io/mathai-website/

## Directory Structure

```
mathai-website/
├── docs/
│   ├── pdfs/              ← Put your PDF files here
│   │   ├── lecture01.pdf
│   │   ├── lecture02.pdf
│   │   └── ...
│   ├── notebooks/         ← Put your .ipynb files here
│   │   ├── week1.ipynb
│   │   └── ...
│   ├── index.md           ← Homepage
│   ├── lectures.md        ← Lists all PDFs
│   ├── tutorials.md       ← Math tutorials
│   └── notebooks.md       ← Lists all notebooks
└── mkdocs.yml            ← Configuration
```
