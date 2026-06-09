from flask import Flask, render_template, request, jsonify
import os
import re
import json
import math
import mimetypes

# Fix MIME type issue on Windows untuk file CSS
mimetypes.add_type('text/css', '.css')

app = Flask(__name__)

# Load model globally (Zero Dependency)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYNONYMS_PATH = os.path.join(BASE_DIR, 'synonyms.json')
CORPUS_PATH = os.path.join(BASE_DIR, 'corpus_data.json')

synonyms_dict = {}
corpus_data = {}

def load_data():
    global synonyms_dict, corpus_data
    
    if os.path.exists(SYNONYMS_PATH):
        try:
            with open(SYNONYMS_PATH, 'r') as f:
                synonyms_dict = json.load(f)
            print("Kamus Sinonim JSON berhasil dimuat.")
        except Exception as e:
            print(f"Error memuat kamus sinonim: {e}")
    else:
        print("synonyms.json tidak ditemukan. Harap jalankan train_model.py")

    if os.path.exists(CORPUS_PATH):
        try:
            with open(CORPUS_PATH, 'r') as f:
                corpus_data = json.load(f)
            print(f"Corpus Data JSON berhasil dimuat: {len(corpus_data.get('docs', []))} artikel.")
        except Exception as e:
            print(f"Error memuat corpus data: {e}")
    else:
        print("corpus_data.json tidak ditemukan. Harap jalankan train_tfidf.py")

load_data()

@app.route('/')
def index():
    return render_template('index.html')

def cosine_similarity_dict(vec1, vec2):
    dot = sum(vec1.get(k, 0) * vec2.get(k, 0) for k in set(vec1) & set(vec2))
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    if mag1 == 0 or mag2 == 0: return 0.0
    return dot / (mag1 * mag2)

@app.route('/api/search', methods=['POST'])
def search():
    if not synonyms_dict or not corpus_data:
        return jsonify({"error": "Data sistem belum siap. Pastikan model JSON sudah dibuat."}), 500
    
    data = request.get_json()
    raw_query = data.get('word', '').lower().strip()
    
    if not raw_query:
        return jsonify({"error": "Kata tidak boleh kosong."}), 400
        
    try:
        # Pecah query menjadi beberapa kata (Multi-word query support)
        query_tokens = raw_query.split()
        
        # Saring kata hubung dasar agar tidak ikut dicari sinonimnya
        basic_stopwords = {"yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "adalah", "ini", "itu", "atau", "juga", "jadi", "sebagai", "dalam", "bahwa", "tersebut"}
        clean_tokens = [w for w in query_tokens if w not in basic_stopwords and len(w) > 2]
        
        if not clean_tokens:
            clean_tokens = query_tokens # Fallback jika user hanya mengetik kata hubung
            
        # 1. Get Synonyms untuk SETIAP kata kunci
        synonyms = []
        for token in clean_tokens:
            token_syns = synonyms_dict.get(token, [])
            synonyms.extend(token_syns)
            
        # Hapus duplikat dan pastikan kata asli tidak masuk di list sinonim
        synonyms = list(set(synonyms) - set(clean_tokens))
        
        search_terms = clean_tokens + synonyms

        # 2. Compute Query TF-IDF Vector
        idf_dict = corpus_data.get('idf', {})
        query_vec = {}
        for term in search_terms:
            if term in idf_dict:
                # Menghitung bobot (TF-IDF query sederhana)
                tf = search_terms.count(term)
                query_vec[term] = tf * idf_dict[term]
                
        # L2 Normalize the query vector (like scikit-learn does)
        mag = math.sqrt(sum(v**2 for v in query_vec.values()))
        if mag > 0:
            for term in query_vec:
                query_vec[term] /= mag

        # 3. Find Matches & Cosine Similarity
        docs = corpus_data.get('docs', [])
        
        relevant_docs = [] # Exact match (for metrics)
        retrieved_docs = []
        results = []
        
        for idx, doc in enumerate(docs):
            tokens = doc.get("tokens", [])
            
            # Dokumen dianggap relevan (Exact Match) jika mengandung SEMUA kata asli dari user
            if all(qt in tokens for qt in clean_tokens):
                relevant_docs.append(idx)
                
            doc_vec = doc.get("vector", {})
            score = cosine_similarity_dict(query_vec, doc_vec)
            
            if score > 0:
                retrieved_docs.append(idx)
                
                highlighted_text = doc["text"]
                if search_terms:
                    sorted_terms = sorted(search_terms, key=len, reverse=True)
                    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, sorted_terms)) + r')\b', re.IGNORECASE)
                    highlighted_text = pattern.sub(r'<mark>\1</mark>', highlighted_text)

                results.append({
                    "title": doc["title"],
                    "text": highlighted_text,
                    "score": round(score, 4),
                    "_score": score # hidden field for precise sorting
                })
                
        # Sort results by score descending
        results = sorted(results, key=lambda x: x["_score"], reverse=True)
        # Clean up hidden field
        for r in results:
            del r["_score"]

        # 4. Metrics calculation
        total_docs = len(docs)
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
        import traceback
        traceback.print_exc()
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
            
            print("\n[SELESAI] Data lama berhasil ditimpa dengan data JSON baru!\n")
            load_data()
        else:
            print("Baik, menggunakan model JSON yang sudah ada.\n")

    app.run(debug=True, port=5000)

