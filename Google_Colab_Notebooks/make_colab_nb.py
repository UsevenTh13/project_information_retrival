import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Sistem Temu Kembali - Versi Google Colab\n",
    "Jalankan sel ini untuk menginstal pustaka yang dibutuhkan dan memuat model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "%pip install pandas scikit-learn nltk gensim Sastrawi\n",
    "\n",
    "import pandas as pd\n",
    "import pickle\n",
    "import re\n",
    "import nltk\n",
    "from gensim.models import Word2Vec\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "\n",
    "nltk.download('stopwords', quiet=True)\n",
    "stop_words = set(nltk.corpus.stopwords.words('indonesian'))\n",
    "stop_words.update({'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'pada', 'dengan', 'adalah', 'ini', 'itu', 'atau', 'juga', 'jadi'})\n",
    "\n",
    "# Pastikan Anda sudah mengunggah (upload) file-file ini ke direktori Google Colab:\n",
    "# 1. cnbc_tokenized.csv\n",
    "# 2. word2vec_model.bin\n",
    "# 3. tfidf_model.pkl\n",
    "\n",
    "try:\n",
    "    df_corpus = pd.read_csv('cnbc_tokenized.csv')\n",
    "    df_corpus['Tokens_Str'] = df_corpus['Tokens_Str'].fillna('')\n",
    "    df_corpus['Text'] = df_corpus['Text'].fillna('')\n",
    "\n",
    "    w2v_model = Word2Vec.load('word2vec_model.bin')\n",
    "\n",
    "    with open('tfidf_model.pkl', 'rb') as f:\n",
    "        tfidf_data = pickle.load(f)\n",
    "        tfidf_vectorizer = tfidf_data['vectorizer']\n",
    "        tfidf_matrix = tfidf_data['tfidf_matrix']\n",
    "\n",
    "    print('Semua data dan model berhasil dimuat!')\n",
    "except FileNotFoundError:\n",
    "    print('ERROR: Anda belum mengunggah file CSV atau Model (.bin / .pkl) ke Colab!')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Fungsi Pencarian\n",
    "Fungsi ini menggantikan logika `app.py` untuk menerima query dan mengembalikan hasil beserta metrik evaluasinya."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def search_engine(query, top_n=5):\n",
    "    query = query.lower().strip()\n",
    "    if not query:\n",
    "        return 'Query kosong'\n",
    "        \n",
    "    synonyms = []\n",
    "    if query in w2v_model.wv.key_to_index:\n",
    "        similar_words = w2v_model.wv.most_similar(query, topn=30)\n",
    "        for w, score in similar_words:\n",
    "            if w not in stop_words and len(w) > 2:\n",
    "                synonyms.append(w)\n",
    "                if len(synonyms) >= 10:\n",
    "                    break\n",
    "                    \n",
    "    search_terms = [query] + synonyms\n",
    "    print(f'Mencari kata: {query}')\n",
    "    print(f'Sinonim (Word2Vec): {synonyms}\\n')\n",
    "    \n",
    "    relevant_docs = []\n",
    "    retrieved_docs = []\n",
    "    results = []\n",
    "    \n",
    "    for index, row in df_corpus.iterrows():\n",
    "        tokens_str = row['Tokens_Str']\n",
    "        tokens_list = tokens_str.split(',') if tokens_str else []\n",
    "        if query in tokens_list:\n",
    "            relevant_docs.append(index)\n",
    "            \n",
    "    query_string = ' '.join(search_terms)\n",
    "    query_vec = tfidf_vectorizer.transform([query_string])\n",
    "    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()\n",
    "    \n",
    "    related_docs_indices = cosine_similarities.argsort()[::-1]\n",
    "    \n",
    "    for index in related_docs_indices:\n",
    "        score = cosine_similarities[index]\n",
    "        if score > 0:\n",
    "            retrieved_docs.append(int(index))\n",
    "            if len(results) < top_n:\n",
    "                row = df_corpus.iloc[index]\n",
    "                results.append({\n",
    "                    'title': row['Title'],\n",
    "                    'score': round(score, 4),\n",
    "                    'text': row['Text'][:200] + '...'\n",
    "                })\n",
    "                \n",
    "    total_docs = len(df_corpus)\n",
    "    tp = len(set(relevant_docs).intersection(retrieved_docs))\n",
    "    fp = len(set(retrieved_docs) - set(relevant_docs))\n",
    "    fn = len(set(relevant_docs) - set(retrieved_docs))\n",
    "    tn = total_docs - (tp + fp + fn)\n",
    "    \n",
    "    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0\n",
    "    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0\n",
    "    accuracy = ((tp + tn) / total_docs) * 100 if total_docs > 0 else 0.0\n",
    "    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0\n",
    "    \n",
    "    print('--- METRIK EVALUASI ---')\n",
    "    print(f'Precision: {precision:.2f}% | Recall: {recall:.2f}%')\n",
    "    print(f'Accuracy:  {accuracy:.2f}% | F1 Score: {f1_score:.2f}%')\n",
    "    print(f'TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}\\n')\n",
    "    \n",
    "    print('--- HASIL ARTIKEL TERATAS ---')\n",
    "    if not results:\n",
    "        print('Tidak ada artikel yang cocok.')\n",
    "    for i, res in enumerate(results, 1):\n",
    "        print(f\"{i}. {res['title']} (Skor TF-IDF: {res['score']})\")\n",
    "        print(f\"   {res['text']}\\n\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Silakan ganti kata 'pajak' dengan kata yang ingin Anda cari\n",
    "search_engine('pajak', top_n=5)\n"
   ]
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

with open('App_Colab_Version.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
