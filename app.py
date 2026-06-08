from flask import Flask, render_template, request, jsonify
from gensim.models import Word2Vec
import pandas as pd
import os
import re
import pickle
import mimetypes
from sklearn.metrics.pairwise import cosine_similarity

# Fix MIME type issue on Windows untuk file CSS
mimetypes.add_type('text/css', '.css')

app = Flask(__name__)

# Load model globally
MODEL_PATH = './word2vec_model.bin'
CSV_PATH = './cnbc_tokenized.csv'
TFIDF_PATH = './tfidf_model.pkl'

model = None
df_corpus = None
tfidf_vectorizer = None
tfidf_matrix = None

def load_data():
    global model, df_corpus, tfidf_vectorizer, tfidf_matrix
    # Load Word2Vec
    if os.path.exists(MODEL_PATH):
        try:
            model = Word2Vec.load(MODEL_PATH)
            print("Model Thesaurus berhasil dimuat.")
        except Exception as e:
            print(f"Error memuat model: {e}")
    else:
        print("Model tidak ditemukan. Pastikan Anda sudah menjalankan train_model.py")

    # Load CSV Corpus
    if os.path.exists(CSV_PATH):
        try:
            df_corpus = pd.read_csv(CSV_PATH)
            print(f"Corpus Berita berhasil dimuat: {len(df_corpus)} artikel.")
            # Handle NaN values
            df_corpus['Tokens_Str'] = df_corpus['Tokens_Str'].fillna('')
            df_corpus['Text'] = df_corpus['Text'].fillna('')
        except Exception as e:
            print(f"Error memuat CSV: {e}")
    else:
        print("CSV tidak ditemukan. Harap pastikan pipeline.py sudah dijalankan.")

    # Load TF-IDF Model
    if os.path.exists(TFIDF_PATH):
        try:
            with open(TFIDF_PATH, 'rb') as f:
                tfidf_data = pickle.load(f)
                tfidf_vectorizer = tfidf_data['vectorizer']
                tfidf_matrix = tfidf_data['tfidf_matrix']
            print("Model TF-IDF berhasil dimuat.")
        except Exception as e:
            print(f"Error memuat model TF-IDF: {e}")
    else:
        print("Model TF-IDF tidak ditemukan. Pastikan Anda sudah menjalankan train_tfidf.py")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    if model is None or df_corpus is None:
        return jsonify({"error": "Data sistem belum siap. Pastikan model dan corpus sudah dibuat."}), 500
    
    data = request.get_json()
    query = data.get('word', '').lower().strip()
    
    if not query:
        return jsonify({"error": "Kata tidak boleh kosong."}), 400
        
    try:
        synonyms = []
        if query in model.wv.key_to_index:
            # We import nltk stopwords to filter synonyms
            import nltk
            try:
                stop_words = set(nltk.corpus.stopwords.words('indonesian'))
            except:
                nltk.download('stopwords', quiet=True)
                stop_words = set(nltk.corpus.stopwords.words('indonesian'))
            
            additional_stopwords = {"yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "adalah", "ini", "itu", "atau", "juga", "jadi"}
            stop_words.update(additional_stopwords)
            
            # Request more synonyms initially in case we filter out many
            similar_words = model.wv.most_similar(query, topn=30)
            for w, score in similar_words:
                if w not in stop_words and len(w) > 2:
                    synonyms.append(w)
                    if len(synonyms) >= 10:
                        break

        search_terms = [query] + synonyms

        # Lists for metrics calculation
        relevant_docs = [] # Contains exact query
        retrieved_docs = [] # Contains query OR synonyms (score > 0)

        results = []

        # Find relevant docs (Exact match) for evaluation
        for index, row in df_corpus.iterrows():
            tokens_str = row['Tokens_Str']
            tokens_list = tokens_str.split(',') if tokens_str else []
            if query in tokens_list:
                relevant_docs.append(index)

        # Transform query to TF-IDF vector
        query_string = ' '.join(search_terms)
        query_vec = tfidf_vectorizer.transform([query_string])
        
        # Calculate Cosine Similarity
        cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Get documents with similarity > 0
        related_docs_indices = cosine_similarities.argsort()[::-1]
        
        for index in related_docs_indices:
            score = cosine_similarities[index]
            if score > 0:
                retrieved_docs.append(int(index))
                
                row = df_corpus.iloc[index]
                original_text = row['Text']
                title = row['Title']

                # Prepare Highlighted Text
                highlighted_text = original_text
                if search_terms:
                    # Sort by length descending so longer words match first
                    sorted_terms = sorted(search_terms, key=len, reverse=True)
                    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_terms)) + r')\b', re.IGNORECASE)
                    highlighted_text = pattern.sub(r'<mark>\1</mark>', highlighted_text)

                results.append({
                    "title": title,
                    "text": highlighted_text,
                    "score": round(score, 4)
                })

        # Calculate Metrics
        total_docs = len(df_corpus)
        
        tp = len(set(relevant_docs).intersection(retrieved_docs))
        fp = len(set(retrieved_docs) - set(relevant_docs))
        fn = len(set(relevant_docs) - set(retrieved_docs))
        tn = total_docs - (tp + fp + fn)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / total_docs if total_docs > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return jsonify({
            "metrics": {
                "precision": round(precision * 100, 2),
                "recall": round(recall * 100, 2),
                "accuracy": round(accuracy * 100, 2),
                "f1_score": round(f1_score * 100, 2),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn
            },
            "synonyms": synonyms,
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Mencegah input() terpanggil dua kali saat Flask Auto-Reloader aktif
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        print("="*50)
        pilihan = input("Apakah Anda ingin merayapi (crawling) dan melatih ulang seluruh model data dari awal? \n(Ketik 'y' untuk menimpa data lama, atau 'n' untuk gunakan data yang sudah ada): ")
        
        if pilihan.strip().lower() == 'y':
            import pipeline
            import train_model
            import train_tfidf
            
            print("\n--- [1/3] Memulai Pipeline Crawling & Tokenisasi ---")
            pipeline.run_pipeline()
            
            print("\n--- [2/3] Memulai Pelatihan Word2Vec (Thesaurus) ---")
            train_model.train_and_save_model()
            
            print("\n--- [3/3] Memulai Pelatihan Vector Space Model (TF-IDF) ---")
            train_tfidf.train_and_save_tfidf()
            
            print("\n[SELESAI] Data lama berhasil ditimpa dengan data baru yang sudah dilatih!\n")
        else:
            print("Baik, menggunakan model dan corpus yang sudah ada (tidak menimpa).\n")

    load_data()
    app.run(debug=True, port=5000)
