import fitz # PyMuPDF
import re
import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
import os

nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)

def run_pipeline():
    # 1. Ekstraksi URL dari PDF
    pdf_path = './50 Link CNBC Indonesia.pdf'
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} tidak ditemukan.")
        return

    print("Membaca PDF dan mengekstrak URL...")
    doc = fitz.open(pdf_path)
    full_text = ''
    for page in doc:
        full_text += page.get_text()

    pattern = r'\S*cnbcindonesia\.com/\S+'
    raw_matches = re.findall(pattern, full_text)
    target_urls = []
    for match in raw_matches:
        clean_url = re.sub(r'^.*?://', 'https://', match) if '://' in match else 'https://' + match.split('cnbcindonesia.com')[-1]
        if not clean_url.startswith('http'):
            clean_url = 'https://www.cnbcindonesia.com' + match.split('cnbcindonesia.com')[-1]
        clean_url = clean_url.rstrip('.,;()"\' ')
        if clean_url not in target_urls:
            target_urls.append(clean_url)

    target_urls = target_urls[:50]
    print(f"Berhasil menemukan {len(target_urls)} URL target.")

    # 2. Crawling
    scraper = cloudscraper.create_scraper()
    scraped_data = []
    print("Memulai Crawling...")
    for i, url in enumerate(target_urls, 1):
        try:
            response = scraper.get(url, timeout=20)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.find('h1').get_text(strip=True) if soup.find('h1') else 'N/A'
                content_div = soup.find('div', class_='detail_text') or soup.find('div', class_='detail-text')
                text = " ".join([p.get_text(strip=True) for p in content_div.find_all('p')]) if content_div else 'N/A'
                scraped_data.append({'URL': url, 'Title': title, 'Text': text})
                print(f"[{i}] Berhasil: {title[:30]}...")
            else:
                print(f"[{i}] Gagal HTTP {response.status_code} pada {url}")
        except Exception as e:
            print(f"Error pada {url}: {e}")

    df = pd.DataFrame(scraped_data)

    if df.empty:
        print("Gagal Crawling: Data kosong! Coba periksa koneksi internet.")
        return

    # 3. Preprocessing
    print("Memulai Preprocessing (Sastrawi Stemming, memakan waktu)...")
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    def preprocess_text(text):
        if text == 'N/A': return ""
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        return stemmer.stem(text)

    df['Clean_Text'] = df['Text'].apply(preprocess_text)

    # 4. Tokenisasi
    print("Memulai Tokenisasi...")
    df['Tokens'] = df['Clean_Text'].apply(word_tokenize)

    # 5. Simpan ke CSV
    df['Tokens_Str'] = df['Tokens'].apply(lambda x: ','.join(x))
    csv_path = './cnbc_tokenized.csv'
    df.to_csv(csv_path, index=False)
    print(f"Data Tahap 1 (CSV) tersimpan secara lokal di {csv_path}")

    # 6. Simpan ke Pickle
    all_tokens = df['Tokens'].tolist()
    pickle_path = './cnbc_tokens.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_tokens, f)
    print(f"Data berhasil diproses dan disimpan sebagai Pickle di {pickle_path}")

if __name__ == "__main__":
    run_pipeline()
