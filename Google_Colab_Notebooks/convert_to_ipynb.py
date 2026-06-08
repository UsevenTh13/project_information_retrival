import json
import os

def create_notebook(filename, code_content, title):
    notebook = {
     "cells": [
      {
       "cell_type": "markdown",
       "metadata": {},
       "source": [
        f"# {title}\n",
        "Jalankan sel di bawah ini untuk mengeksekusi kode."
       ]
      },
      {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": code_content.splitlines(True)
      }
     ],
     "metadata": {
      "kernelspec": {
       "display_name": "Python 3",
       "language": "python",
       "name": "python3"
      },
      "language_info": {
       "name": "python"
      }
     },
     "nbformat": 4,
     "nbformat_minor": 4
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

# 1. Pipeline
with open('pipeline.py', 'r', encoding='utf-8') as f:
    pip_code = "%pip install cloudscraper beautifulsoup4 pandas Sastrawi nltk PyMuPDF\n\n" + f.read() + "\nrun_pipeline()\n"
create_notebook('1_Pipeline_Crawling_Preprocessing.ipynb', pip_code, "Tahap 1: Crawling & Preprocessing")

# 2. Train Word2Vec
with open('train_model.py', 'r', encoding='utf-8') as f:
    w2v_code = "%pip install gensim\n\n" + f.read() + "\ntrain_and_save_model()\n"
create_notebook('2_Train_Word2Vec_Thesaurus.ipynb', w2v_code, "Tahap 2: Pelatihan Model Word2Vec (Thesaurus)")

# 3. Train TFIDF
with open('train_tfidf.py', 'r', encoding='utf-8') as f:
    tfidf_code = "%pip install scikit-learn pandas\n\n" + f.read() + "\ntrain_and_save_tfidf()\n"
create_notebook('3_Train_TFIDF_Model.ipynb', tfidf_code, "Tahap 3: Pelatihan Vector Space Model (TF-IDF)")

print("Semua file .py berhasil dikonversi ke .ipynb!")
