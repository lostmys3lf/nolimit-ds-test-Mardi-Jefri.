# Analisis Sentimen kesehatan mental ibu hamil menggunakan IndoBERT + Streamlit

Mini project untuk klasifikasi sentimen teks berbahasa Indonesia (0 = negatif, 1 = positif) dengan model IndoBERT dan antarmuka Streamlit.

## Demo Aplikasi

Aplikasi bisa langsung dicoba di Hugging Face Spaces:

👉 https://jefris3lf-indo-bert-sentiment.hf.space

Project ini juga bisa dijalankan secara lokal dengan `streamlit run app.py`
(setup environment lihat bagian **Setup & Cara Menjalankan**).

## Prasyarat

- Python 3.10 atau lebih (project ini dikembangkan dan diuji dengan Python 3.10.11)
- pip versi terbaru

## Dataset

### Sumber & konteks

- Komentar pengguna dari TikTok terkait topik kesehatan mental ibu hamil di Indonesia.

- Dikumpulkan menggunakan platform web scraping Apify.

- Dataset ini bagian dari tugas akhir saya berjudul:
“PENGEMBANGAN REKOMENDASI FITUR APLIKASI MELALUI ANALISIS SENTIMEN DENGAN ALGORITMA INDOBERT DAN BERTOPIC PADA KESEHATAN MENTAL IBU HAMIL DI INDONESIA”.

### Lisensi & ketersediaan

- Data mentah tidak disertakan di repo ini karena keterbatasan Terms of Service TikTok dan pertimbangan privasi.

- Dataset hanya digunakan untuk keperluan riset akademik internal (skripsi dan technical test ini).

- Untuk replikasi, peneliti lain perlu mengumpulkan data sendiri sesuai kebijakan TikTok dan regulasi yang berlaku.

### Tipe label & anotasi

- Tugas: binary sentiment classification

    - 0 = negatif

    - 1 = positif

- ±4.000 komentar dilabeli manual oleh peneliti.

- Model IndoBERT awal kemudian di-fine-tune dan digunakan untuk melabeli otomatis ±6.000 komentar tambahan (self-training / semi-supervised), merujuk paper: “Self-training Improves Pre-training for Natural Language Understanding”.

## Model yang Digunakan

- **Base model**: [`indobenchmark/indobert-base-p2`](https://huggingface.co/indobenchmark/indobert-base-p2)  
  (BERT base untuk Bahasa Indonesia.)

- **Arsitektur fine-tuning**:
  - `BertForSequenceClassification(num_labels=2)`
  - Tokenizer: `BertTokenizer` dari model yang sama.

- **Training (ringkas)**:
  - Framework: Hugging Face **Transformers** + **PyTorch**
  - Loss: `CrossEntropyLoss` dengan **class weight** (imbalance)
  - 5-fold **StratifiedKFold**
  - Metrik utama: **macro F1**
  - `load_best_model_at_end=True` (ambil checkpoint terbaik di validasi)
Checkpoint dan tokenizer disimpan di:

models/indoBERT_best/
    vocab.txt
    tokenizer_config.json
    special_tokens_map.json
    checkpoint-2390/
        config.json
        model.safetensors
        training_args.bin
        ...
## Kode training model dapat dilihat di:

- `indoBERT_sentiment_training.ipynb`

Notebook ini menunjukkan alur pelatihan IndoBERT pada dataset yang sama.

## Setup & Cara Menjalankan
1. **Buat dan aktifkan virtual environment**
# Masuk ke folder project
cd path/to/project

# Buat virtualenv
python -m venv .venv

# Aktifkan (Command Prompt / CMD, Windows)
.\.venv\Scripts\activate

2. **Install dependensi**
pip install --upgrade pip
pip install -r requirements.txt

3. **Jalankan aplikasi Streamlit**
streamlit run app.py
# jika port 8501 sudah dipakai:
# streamlit run app.py --server.port 8502 atau port yang belum terpakai lainnya
Buka URL yang muncul di terminal (misalnya http://localhost:8501), masukkan teks berbahasa Indonesia, lalu klik “Analisis Sentimen” untuk melihat prediksi positif/negatif beserta tingkat keyakinan model.

> Catatan: file checkpoint model (`models/indoBERT_best/checkpoint-2390/model.safetensors`) 
> tidak disertakan di repo ini karena ukurannya ~475 MB dan melebihi batas GitHub (100 MB).
> Untuk menjalankan inference lokal, unduh model dari https://drive.google.com/drive/folders/1CM8MMJvswgbHGVOdr3BdOcPRLMPduE4a?usp=sharing 
> lalu letakkan di path tersebut.



