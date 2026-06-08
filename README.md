# Sistem Temu Kembali (Information Retrieval) - Ekstraksi & Thesaurus Artikel CNBC

Proyek ini adalah implementasi sistematis dari **Sistem Temu Kembali Informasi (Information Retrieval)** yang bertujuan untuk mengumpulkan artikel berita berbahasa Indonesia dari CNBC, membersihkannya, melakukan *tokenisasi*, dan pada akhirnya membangun kamus kata (Thesaurus) berdasarkan data yang didapatkan.

---

## 📂 Struktur Proyek & Tahapan
Proses pengolahan data dibagi menjadi 3 tahapan yang masing-masing dijalankan melalui *Jupyter Notebook* terpisah agar lebih mudah dipahami:

### 1. `BS_article_html.ipynb` (Tahap 1: Crawling & Preprocessing)
Ini adalah tahap awal dimana pengumpulan data dilakukan.
* **Ekstraksi PDF:** Membaca dokumen `50 Link CNBC Indonesia.pdf` lalu mengambil seluruh tautan web secara otomatis.
* **Crawling Berita:** Mengakses seluruh tautan tersebut dan mengekstrak isi teks beritanya (tanpa iklan atau menu website).
* **Pembersihan Teks (Preprocessing):** Menyeragamkan seluruh huruf menjadi huruf kecil (*lowercase*), menghapus angka, serta membersihkan tanda baca.
* **Stemming:** Membuang semua imbuhan (awalan, sisipan, akhiran) dari setiap kata menggunakan Sastrawi, sehingga menyisakan akar katanya saja (contoh: *meningkatkan* $\rightarrow$ *tingkat*).
* **Tokenisasi:** Memecah paragraf panjang menjadi potongan satuan kata (token).
* **Output:** Mengekspor seluruh data yang telah bersih ke dalam file `cnbc_tokenized.csv`.

### 2. `PickleConverter.ipynb` (Tahap 2: Konversi ke Data Biner)
Tahap jembatan untuk mengamankan format data asli.
* Membaca kembali data CSV dari Tahap 1.
* Mengembalikan bentuk *string* token yang tercetak di CSV kembali menjadi format himpunan struktur data murni (`List`).
* **Output:** Mengonversinya ke dalam file biner Pickle (`cnbc_tokens.pkl`) agar data tersebut terkompresi secara rapi dan sangat cepat saat diload kembali di kemudian hari.

### 3. `ThesaurusTest.ipynb` (Tahap 3: Pembuatan Kamus Thesaurus)
Tahap akhir untuk mencari persamaan makna (sinonim) dari setiap kata yang berhasil kita kumpulkan.
* Membaca data token dari dalam file Pickle.
* Menyaring hanya kata-kata unik yang muncul di dalam keseluruhan berita.
* Membandingkan setiap kata tersebut dengan Kamus Asli NLTK WordNet versi Bahasa Indonesia (`omw-1.4`) untuk mencari sinonim resminya.
* **Output:** Menghasilkan kamus Thesaurus final berformat `full_thesaurus_kamus.csv` yang berisi ribuan kaitan kata untuk keperluan Sistem Temu Kembali.

---

## 🛠 Pustaka (Library) yang Digunakan
Untuk dapat menjalankan keseluruhan *notebook* secara lokal, instal pustaka Python berikut melalui terminal:
```bash
pip install cloudscraper beautifulsoup4 pandas Sastrawi nltk PyMuPDF
```

Fungsi spesifik dari setiap pustaka yang kita panggil:
* **`fitz` (PyMuPDF):** Modul utama untuk membaca dan mengekstrak tulisan dari dalam file PDF.
* **`re` (Regex):** Alat bantu untuk mencari dan memisahkan "pola teks" secara instan (digunakan untuk memilah struktur format URL).
* **`cloudscraper`:** Pengganti `requests` yang canggih untuk menembus tembok pelindung anti-bot bawaan situs besar seperti CNBC.
* **`BeautifulSoup` (bs4):** Membedah bongkahan tulang HTML website lalu mengambil sepesifik teks yang hanya terbungkus dalam bagian judul atau paragraf.
* **`pandas`:** Menyusun seluruh kumpulan data mentah tersebut ke dalam bentuk kerangka tabel (DataFrame) rapi yang siap di ekspor ke Excel/CSV.
* **`string`:** Pustaka standar yang menyediakan seluruh kumpulan daftar tanda baca secara lengkap (seperti koma, titik, dll) agar bisa segera dihapus.
* **`nltk` (Natural Language Toolkit):** Tulang punggung utama pemrograman bahasa alami (*NLP*). Bertugas dalam *tokenisasi* kata serta menyediakan mesin *Thesaurus* bahasa.
* **`Sastrawi`:** Pustaka linguistik khusus Indonesia yang telah diakui karena kepintarannya membuang semua jenis imbuhan bahasa Indonesia yang rumit dan majemuk secara akurat.

---

## 🚀 Cara Penggunaan
1. Buka file **`BS_article_html.ipynb`**. Jalankan (*Run All*) dan bersabarlah menunggu proses *Crawling* dan *Stemming* Sastrawi (proses stemming akan sedikit memakan waktu karena Sastrawi menganalisis setiap kata di dalam kamusnya).
2. Setelah sukses dan file `cnbc_tokenized.csv` muncul, buka **`PickleConverter.ipynb`** dan jalankan semua isinya (akan berlangsung instan).
3. Terakhir, buka **`ThesaurusTest.ipynb`**, jalankan isinya, dan saksikan seluruh kamus *Thesaurus* baru Anda tercipta!
