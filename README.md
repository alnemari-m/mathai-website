# Mathematics of AI - Course Website

Simple course website for Mathematics of AI graduate course.

## Live Site

https://YOUR_GITHUB_USERNAME.github.io/mathai-website/

## Structure

```
├── docs/
│   ├── index.md          # Homepage
│   ├── lectures.md       # Lecture PDFs
│   ├── tutorials.md      # Math tutorials
│   ├── notebooks.md      # Python notebooks
│   ├── pdfs/            # Store PDF files here
│   └── notebooks/       # Store .ipynb files here
└── mkdocs.yml           # Configuration
```

## Adding Content

### Add Lecture PDFs

1. Put PDF files in `docs/pdfs/`
2. They'll be accessible at `pdfs/lecture01.pdf`

### Add Notebooks

1. Put `.ipynb` files in `docs/notebooks/`
2. They'll be accessible at `notebooks/week1_vectors.ipynb`

### Convert Markdown to PDF

```bash
# Install pandoc
sudo apt install pandoc texlive-latex-base

# Convert lecture markdown to PDF
cd ~/Documents/mathai
pandoc LECTURE_01_Vector_Spaces_And_Data_Manifolds.md -o lecture01.pdf
```

## Local Development

```bash
pip install mkdocs pymdown-extensions
mkdocs serve
```

Visit http://127.0.0.1:8000

## Deploy to GitHub

```bash
# First time setup
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/mathai-website.git
git push -u origin main

# Updates
git add .
git commit -m "Update lectures"
git push
```

GitHub Actions will automatically deploy to GitHub Pages.

## License

Educational use only.
