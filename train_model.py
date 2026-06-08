import pickle
import pandas as pd
from gensim.models import Word2Vec
import os

def train_and_save_model():
    pickle_path = './thesaurus_dict_tokens.pkl'
    csv_path = './cnbc_tokenized.csv'
    model_path = './word2vec_model.bin'

    if not os.path.exists(pickle_path) or not os.path.exists(csv_path):
        print(f"Error: {pickle_path} atau {csv_path} tidak ditemukan.")
        return

    print(f"Memuat data Kamus dari {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        loaded_tokens = pickle.load(f)

    print(f"Memuat data Berita dari {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Tokens_Str'] = df['Tokens_Str'].fillna('')
    cnbc_sentences = []
    for index, row in df.iterrows():
        if row['Tokens_Str']:
            cnbc_sentences.append(row['Tokens_Str'].split(','))

    print("Menggabungkan kedua dataset...")
    combined_sentences = loaded_tokens + cnbc_sentences

    print(f"Melatih model Word2Vec dengan {len(combined_sentences)} kalimat gabungan...")
    # Latih model Thesaurus (Word2Vec)
    model = Word2Vec(sentences=combined_sentences, vector_size=300, window=5, min_count=5, workers=4, sg=1)

    print(f"Menyimpan model ke {model_path}...")
    model.save(model_path)
    
    print("Mengekstrak JSON Synonyms (Zero Dependency) untuk Vercel...")
    import json
    import nltk
    try:
        stop_words = set(nltk.corpus.stopwords.words('indonesian'))
    except:
        nltk.download('stopwords', quiet=True)
        stop_words = set(nltk.corpus.stopwords.words('indonesian'))
        
    additional_stopwords = {"yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "adalah", "ini", "itu", "atau", "juga", "jadi"}
    stop_words.update(additional_stopwords)
    
    synonyms_dict = {}
    for word in model.wv.key_to_index:
        raw_syns = model.wv.most_similar(word, topn=30)
        clean_syns = [w for w, score in raw_syns if w not in stop_words and len(w) > 2]
        if clean_syns:
            synonyms_dict[word] = clean_syns[:10]
            
    with open('synonyms.json', 'w') as f:
        json.dump(synonyms_dict, f)
    print("Selesai mengekstrak synonyms.json!")
    
    print("Selesai! Model Word2Vec siap digunakan.")

if __name__ == "__main__":
    train_and_save_model()
