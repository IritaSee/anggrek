# Anggrek Dataset — EDA & Classification Pipeline

Repositori ini berisi **data pipeline** untuk dataset gambar anggrek, mulai dari eksplorasi data (EDA) hingga klasifikasi dengan model deep learning. Dataset diorganisasi seperti struktur `flow_from_directory` (setiap sub-folder = satu kelas).

```
anggrek/
├── README.md
├── requirements.txt
├── eda_anggrek.ipynb              ← eksplorasi & ROI detection (unsupervised)
├── coelogyne_classification.ipynb ← training & evaluasi model klasifikasi
└── data/                          ← letakkan dataset di sini
    ├── Asperata/
    │   ├── img001.jpg
    │   └── ...
    ├── Dayana/
    ├── Pandurata/
    ├── Rumphii/
    └── Swaniana/
```

---

## 1. `eda_anggrek.ipynb` — Eksplorasi Data (Unsupervised)

Mengeksplorasi dataset secara **unsupervised** untuk menemukan **Region of Interest (ROI)** — area dalam gambar yang mengandung paling banyak perbedaan visual antar sampel.

| Teknik | Tujuan |
|---|---|
| Distribusi statistik pixel | Baseline karakteristik warna & kecerahan |
| Variance map | Temukan piksel/area paling bervariasi → ROI kandidat utama |
| Gradient magnitude map | Area tepi & tekstur dengan kontras tinggi |
| PCA pada pixel | Reduksi dimensi & komponen visual utama (eigenimage) |
| K-Means patch clustering | Kelompokkan pola visual tanpa label, distribusi spasial |
| t-SNE embedding | Visualisasi kesamaan antar gambar (thumbnail grid) |
| Final ROI score | Skor gabungan: 40% variance + 35% gradient + 25% PC1 |

---

## 2. `coelogyne_classification.ipynb` — Klasifikasi 5 Spesies Coelogyne

Klasifikasi **5 spesies Coelogyne** (Asperata, Dayana, Pandurata, Rumphii, Swaniana) dengan dua arsitektur model.

### Dataset
| Split | Sumber |
|---|---|
| Train / Val | `flow_from_directory` dengan `validation_split`, atau subfolder `train/` & `val/` terpisah |

### Model yang Dibandingkan

| Model | Strategi | Highlight |
|---|---|---|
| **MobileNetV2** | Transfer Learning (ImageNet pretrained) | Phase 1: head-only → Phase 2: fine-tune top-50 layers backbone |
| **CSPDarknet53** | From scratch (YOLO backbone, Pure Keras) | Mish activation, residual + CSP blocks, cosine decay + warmup LR |

### Pipeline Notebook

| Bagian | Isi |
|---|---|
| 0–1 | Install, konfigurasi, import |
| 2 | Data pipeline + augmentasi + distribusi kelas |
| 3 | Build & training MobileNetV2 (2 phase) |
| 4 | Build & training CSPDarknet53 |
| 5 | Visualisasi training history (loss & accuracy) |
| 6 | Evaluasi: confusion matrix & classification report |
| 7 | **Grad-CAM** — visualisasi area yang diperhatikan model |
| 8 | Inference — prediksi gambar baru |
| 9 | Simpan model (`.keras` + TFLite untuk mobile) |
| 10 | Ringkasan hasil perbandingan model |

---

## Cara Pakai

```bash
conda activate py313
pip install -r requirements.txt

# Eksplorasi data terlebih dahulu
jupyter notebook eda_anggrek.ipynb

# Lanjut training & evaluasi model
jupyter notebook coelogyne_classification.ipynb
```
