FROM python:3.11-slim

WORKDIR /app

# Salin requirements.txt dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file proyek
COPY . .

# Ekspos port 7860 (Wajib untuk Hugging Face)
ENV PORT=7860
EXPOSE 7860

# Jalankan aplikasi Flask dengan Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
