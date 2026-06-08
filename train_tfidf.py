import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def train_and_save_tfidf():
    csv_path = './cnbc_tokenized.csv'
    model_path = './tfidf_model.pkl'

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} tidak ditemukan. Silakan jalankan pipeline.py terlebih dahulu.")
        return

    print(f"Memuat data CSV dari {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Fill NaN values with empty string
    df['Tokens_Str'] = df['Tokens_Str'].fillna('')
    
    # We will use Tokens_Str (comma separated tokens) directly
    # Since they are already stemmed and cleaned, we can just replace commas with spaces
    documents = df['Tokens_Str'].apply(lambda x: x.replace(',', ' ')).tolist()

    print("Melatih model TF-IDF...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    print(f"Menyimpan model ke {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'vectorizer': vectorizer,
            'tfidf_matrix': tfidf_matrix
        }, f)
        
    print("Selesai! Model TF-IDF siap digunakan.")

if __name__ == "__main__":
    train_and_save_tfidf()
