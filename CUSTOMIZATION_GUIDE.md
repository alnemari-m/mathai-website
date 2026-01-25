# Customization Guide

Your website now has **custom styling** with your personal branding!

## 🎨 Your Custom Design

### Color Scheme
- **Primary Color:** Dark blue (#2c3e50) - Professional and academic
- **Accent Color:** Bright blue (#3498db) - Modern and engaging
- **Background:** Light gray (#ecf0f1) - Clean and readable

### Custom Elements

1. **Hero Section** - Gradient banner on homepage
2. **Info Cards** - Highlighted content blocks
3. **Instructor Section** - Personal branding area
4. **Signature Footer** - Your name on every page
5. **Custom Tables** - Styled schedule and grading tables

---

## 📝 Personalizing Content

### Update Your Information

Edit `docs/index.md`:

```markdown
**Mohammed Alnemari, Ph.D.**

**Email:** your.email@university.edu
**Office Hours:** Tuesday 2-4 PM
**Location:** Building X, Room Y
```

### Change Colors

Edit `docs/stylesheets/custom.css`:

```css
:root {
  --primary-color: #YOUR_COLOR;  /* Main color */
  --accent-color: #YOUR_COLOR;   /* Accent color */
}
```

**Color Ideas:**
- Academic Blue: `#2c3e50` + `#3498db` (current)
- Forest Green: `#27ae60` + `#2ecc71`
- Deep Purple: `#8e44ad` + `#9b59b6`
- Warm Orange: `#d35400` + `#e67e22`

---

## 🖼️ Adding Your Photo/Logo

Create `docs/images/profile.jpg` and add to homepage:

```markdown
<div class="instructor-info" markdown="1">

## Instructor

![Mohammed Alnemari](images/profile.jpg){ width="150" }

**Mohammed Alnemari, Ph.D.**
...
</div>
```

---

## 📄 Converting Lectures to PDF

Use the provided script:

```bash
cd ~/mathai-website
./convert_lectures.sh
```

Or manually:

```bash
cd ~/Documents/mathai
pandoc LECTURE_01_Vector_Spaces_And_Data_Manifolds.md -o lecture01.pdf
cp lecture01.pdf ~/mathai-website/docs/pdfs/
```

---

## 🎯 Adding Your Notebooks

1. Create Jupyter notebooks
2. Save to `docs/notebooks/`
3. Update `docs/notebooks.md` with links

Example:

```markdown
**My Custom Notebook**
Description here
[📓 View](notebooks/my_notebook.ipynb) | [⬇️ Download](notebooks/my_notebook.ipynb)
```

---

## 🌐 Deploy to GitHub

### First Time

```bash
cd ~/mathai-website

# Update YOUR_GITHUB_USERNAME in mkdocs.yml
nano mkdocs.yml

# Initialize git
git init
git add .
git commit -m "Initial commit - Mohammed Alnemari's course site"

# Create repo on GitHub (mathai-website)
# Then:
git remote add origin https://github.com/YOUR_USERNAME/mathai-website.git
git push -u origin main
```

### Enable GitHub Pages

1. Go to repository Settings → Pages
2. Source will auto-deploy via GitHub Actions
3. Site will be live at: `https://YOUR_USERNAME.github.io/mathai-website/`

### Updates

```bash
# Make changes to any .md files
git add .
git commit -m "Update lectures"
git push

# Site automatically rebuilds!
```

---

## 🎨 Advanced Customization

### Add Custom Font

Edit `docs/stylesheets/custom.css`:

```css
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

body {
  font-family: 'Roboto', sans-serif;
}
```

### Add Badges

```html
<span class="badge">New</span>
<span class="badge badge-updated">Updated</span>
```

### Custom Admonitions

```markdown
!!! note "Important Note"
    This is a highlighted note.

!!! warning "Attention"
    This needs your attention.
```

---

## 📱 Preview Changes

Before deploying:

```bash
cd ~/mathai-website
mkdocs serve
# Visit http://127.0.0.1:8000
```

---

## 🚀 Quick Checklist

Before going live:

- [ ] Update email and office hours
- [ ] Add your PDFs to `docs/pdfs/`
- [ ] Add notebooks to `docs/notebooks/`
- [ ] Update GitHub username in `mkdocs.yml`
- [ ] Test all links locally
- [ ] Review all pages for typos
- [ ] Update course schedule
- [ ] Add your photo (optional)

---

## 💡 Tips

**Keep it Updated:**
- Add new lectures weekly
- Update schedule regularly
- Post announcements on homepage
- Keep office hours current

**Student-Friendly:**
- Clear navigation
- Working download links
- Mobile-responsive design
- Fast loading times

**Professional:**
- Consistent styling
- No broken links
- Regular content updates
- Active maintenance

---

**Your website is now personalized and ready to deploy!**

Mohammed Alnemari • Mathematics of AI • Spring 2026
