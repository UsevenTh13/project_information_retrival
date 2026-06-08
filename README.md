# Sistem Temu Kembali (Information Retrieval) - Mesin Pencari Berita CNBC Hybrid

Proyek ini adalah implementasi lengkap dari **Sistem Temu Kembali Informasi (Information Retrieval)** berupa Mesin Pencari (Search Engine) khusus untuk berita-berita ekonomi dari portal CNBC. 

Sistem ini dirancang sangat cerdas karena menggabungkan algoritma pencarian teks klasik dengan Kecerdasan Buatan (AI) pemahaman bahasa untuk memperluas kueri pencarian (Query Expansion) menggunakan sinonim dari Kamus Thesaurus.

---

## ⚙️ Alur Kerja Sistem (Workflow)

Sistem ini bekerja melalui serangkaian tahapan pemrosesan data (Pipeline) yang sangat terstruktur:

1. **Pengumpulan Tautan (Link Extraction):** Sistem membaca file `50 Link CNBC Indonesia.pdf` lalu mengekstrak seluruh URL artikel berita yang ada di dalamnya secara otomatis.
2. **Perayapan Web (Web Crawling):** Sistem mengunjungi setiap URL berita CNBC tersebut dan menyedot isi teks beritanya dengan menembus sistem anti-bot.
3. **Pembersihan Teks (Preprocessing & Stemming):** Teks mentah dibersihkan dari angka, tanda baca, dan kata sambung (Stopwords). Kemudian, seluruh kata diubah menjadi kata dasar (Stemming) menggunakan pustaka Sastrawi. Hasil akhirnya dipecah menjadi kumpulan kata (Tokenization).
4. **Ekstraksi Kamus Thesaurus:** Sistem mengekstrak buku `KamusThesaurus.pdf` dengan sangat presisi dengan cara mendeteksi **Font Bold (Tebal)** sebagai kata utama, dan kata di sebelahnya sebagai sinonim. 
5. **Pelatihan AI (Word2Vec):** Data kamus dan data berita digabung (Hybrid). Mesin AI kemudian dilatih untuk mempelajari kedekatan makna matematis dari gabungan dataset ini.
6. **Pembuatan Vector Space Model (TF-IDF):** Keseluruhan teks berita diubah menjadi matriks bobot angka. Kata yang penting akan mendapat bobot tinggi, kata yang umum mendapat bobot rendah.
7. **Pencarian & Evaluasi:** Saat pengguna mencari kata (misal: "pajak"), sistem akan mencari sinonimnya ("cukai", "bea"), lalu mencari berita yang paling relevan dengan kumpulan kata tersebut. Sistem kemudian menghitung metrik performanya (Precision, Recall, F1-Score) dengan membandingkannya terhadap Tabel Kebenaran (Ground Truth).

---

## 🧠 Algoritma & Konsep yang Digunakan

* **Word2Vec (Gensim):** Algoritma *Deep Learning* (Jaringan Saraf Tiruan) yang mengubah kata menjadi vektor matematika multi-dimensi. Digunakan untuk menemukan persamaan makna kata (Sinonim) berdasarkan kedekatan jarak vektornya (*Query Expansion*).
* **TF-IDF (Term Frequency-Inverse Document Frequency):** Algoritma pembobotan teks. Digunakan untuk menentukan seberapa penting sebuah kata di dalam suatu artikel dibandingkan dengan di seluruh artikel lainnya.
* **Cosine Similarity:** Rumus trigonometri yang menghitung sudut kemiringan antara dua vektor. Digunakan untuk meranking hasil pencarian (menghitung tingkat kemiripan antara "kata kunci pencarian" dengan "isi teks berita").
* **Confusion Matrix & IR Metrics:** Algoritma evaluasi yang menghitung True Positive (TP), False Positive (FP), dsb., untuk mendapatkan nilai **Precision** (Akurasi Pencarian), **Recall** (Daya Tangkap), **F1-Score** (Keseimbangan), dan **Accuracy**.

---

## 📂 Struktur File Utama & Fungsinya

Berikut adalah penjelasan detail mengenai kode utama penyusun sistem ini dan artefak (*file*) yang dihasilkannya:

### 1. `pipeline.py` (Script Perayap Berita)
* **Fungsi:** Bertugas melakukan tahap 1 sampai 3 (Ekstraksi URL, Crawling CNBC, Preprocessing Sastrawi, dan Tokenisasi).
* **File yang Dihasilkan:** 
  * `cnbc_tokenized.csv`: Tabel data mentah teks berita yang sudah dibersihkan dan di-stemming.
  * `cnbc_tokens.pkl`: File kompresi (Pickle) berisi list token artikel.

### 2. `process_pdf_to_pickle.py` (Script Ekstraktor Kamus)
* **Fungsi:** Membaca `KamusThesaurus.pdf`, mendeteksi *font* teks berhuruf tebal (Bold) untuk memisahkan antara Kata Utama dan Sinonimnya secara rapi.
* **File yang Dihasilkan:**
  * `thesaurus_dict_tokens.pkl`: Kamus terstruktur berisi puluhan ribu pasang kata dan sinonimnya.

### 3. `train_model.py` (Script Pelatih AI Thesaurus)
* **Fungsi:** Menggabungkan dataset dari `cnbc_tokenized.csv` dan `thesaurus_dict_tokens.pkl` (Pendekatan Hybrid), lalu menyuapkannya ke algoritma Word2Vec agar mesin belajar bahasa ekonomi dan bahasa kamus sekaligus.
* **File yang Dihasilkan:**
  * `word2vec_model.bin`: "Otak AI" yang sudah dibekukan dan siap digunakan kapan saja untuk menebak sinonim (*Query Expansion*).

### 4. `train_tfidf.py` (Script Pembuat Matriks Pencarian)
* **Fungsi:** Membaca ulang korpus berita CNBC, menghitung rumus matematika TF-IDF untuk setiap kata, dan mencocokkannya dengan `Tabel_Kebenaran_...csv` untuk keperluan perhitungan metrik.
* **File yang Dihasilkan:**
  * `tfidf_model.pkl`: Model Vector Space dan Matriks dokumen berukuran raksasa yang memungkinkan pencarian berlangsung dalam hitungan milidetik.

### 5. `app.py` (Main Server / Frontend)
* **Fungsi:** Ini adalah **Kode Utama** aplikasi (*Entry Point*). Script ini membangun *server website* menggunakan Flask. Bertugas memuat (*load*) seluruh model yang sudah dilatih (`word2vec_model.bin` dan `tfidf_model.pkl`), menerima *input* pencarian dari antarmuka web, melakukan perhitungan *Cosine Similarity* secara *real-time*, dan menampilkannya secara elegan di layar beserta metrik evaluasinya.

---

## 🛠 Pustaka (Library) yang Digunakan

* **`fitz` (PyMuPDF):** Pustaka andalan untuk membedah file PDF hingga ke level properti *font* (untuk mendeteksi cetak tebal/Bold).
* **`cloudscraper` & `BeautifulSoup4`:** Duet maut pencuri data website. Cloudscraper digunakan untuk mengecoh Cloudflare (Anti-Bot) website CNBC, sementara BeautifulSoup bertugas menyaring HTML untuk mengambil teks beritanya saja.
* **`Sastrawi`:** Mesin kecerdasan linguistik Bahasa Indonesia untuk memotong imbuhan (misal: "mempertanggungjawabkan" $\rightarrow$ "tanggung jawab").
* **`nltk`:** Pustaka standar global untuk NLP (*Natural Language Processing*) yang digunakan untuk memotong kalimat (Tokenisasi).
* **`pandas`:** Mengubah data mentah menjadi format tabel CSV rapi yang mudah dimanipulasi.
* **`gensim`:** Pustaka Machine Learning khusus untuk pemodelan topik dan NLP. Bertugas murni mengeksekusi algoritma Word2Vec.
* **`scikit-learn` (sklearn):** Pustaka Machine Learning untuk mengeksekusi algoritma TF-IDF (TfidfVectorizer) dan menghitung jarak Cosine Similarity.
* **`flask`:** *Framework* backend ringan pembuat *server* HTTP yang menjembatani antara algoritma Python dengan tampilan visual di Web (HTML/CSS).

---

## 🚀 Cara Menjalankan Aplikasi

1. Buka Terminal / Command Prompt.
2. Lakukan instalasi seluruh pustaka di atas menggunakan perintah:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan *server* web utama:
   ```bash
   python app.py
   ```
4. Jika ini adalah pertama kalinya, pada terminal akan muncul pertanyaan apakah Anda ingin merayapi ulang web dan melatih data dari awal. Ketik `y` jika Anda belum memiliki file model biner. Jika model sudah tersedia, ketik `n`.
5. Buka `http://127.0.0.1:5000` di *browser* Anda dan nikmati pencariannya!
