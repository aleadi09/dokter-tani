# Data

Folder ini **tidak di-upload ke GitHub** (lihat `.gitignore`). Isi sendiri di laptop.

## Susunan yang diharapkan

```
data/
└── raw/
    └── chilli_leaf/          ← isi zip dataset diekstrak di sini
        ├── Bacterial/
        ├── Cercospora/
        ├── Curl Virus/
        ├── Healthy Leaf/
        ├── Nutrient Deficiency/
        └── Powdery Mildew/
```

Nama kelas diambil dari **nama folder tempat foto berada**, jadi subfolder tambahan (misalnya `train/`, `test/`) tidak masalah.

## Langkah

1. Download dataset dari [Mendeley Data](https://data.mendeley.com/datasets/tm3v4zmh7c/1).
2. Ekstrak ke `data/raw/chilli_leaf/`.
3. Untuk Colab: upload zip yang sama ke Google Drive di `MyDrive/dokter-tani/chilli_leaf.zip`.
4. Foto hasil memotret sendiri: simpan di `data/own_photos/<nama_kelas>/`.

## Lisensi

Catat lisensi tiap dataset di sini sebelum dipakai di presentasi.
