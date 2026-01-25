#!/bin/bash

# Script to convert lecture markdown files to PDF and copy to website

echo "Converting lectures to PDF..."
cd ~/Documents/mathai

# Check if pandoc is installed
if ! command -v pandoc &> /dev/null; then
    echo "Installing pandoc..."
    sudo apt install -y pandoc texlive-latex-base
fi

# Convert each lecture
for i in {01..09}; do
    input="LECTURE_${i}"*.md
    output="lecture${i}.pdf"
    
    if [ -f $input ]; then
        echo "Converting $input to $output..."
        pandoc "$input" -o "$output"
    fi
done

# Copy PDFs to website
echo "Copying PDFs to website..."
mkdir -p ~/mathai-website/docs/pdfs
cp lecture*.pdf ~/mathai-website/docs/pdfs/

# Also convert syllabus and problem set
if [ -f "SYLLABUS.md" ]; then
    pandoc SYLLABUS.md -o syllabus.pdf
    cp syllabus.pdf ~/mathai-website/docs/pdfs/
fi

if [ -f "assignments/PROBLEM_SET_1.md" ]; then
    pandoc assignments/PROBLEM_SET_1.md -o pset1.pdf
    cp pset1.pdf ~/mathai-website/docs/pdfs/
fi

echo "Done! PDFs are in ~/mathai-website/docs/pdfs/"
