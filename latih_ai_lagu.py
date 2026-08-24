from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "Spotify Dataset.csv"
MODEL_PATH = BASE_DIR / "song_similarity_model.joblib"
METADATA_PATH = BASE_DIR / "song_metadata.parquet"

# Statistik yang menggambarkan karakter suara lagu.
FITUR_NUMERIK = [
    "Bpm",
    "Decibel",
    "Energy",
    "Danceability",
    "Liveness",
    "Valence",
    "Duration",
    "Acousticness",
    "Speechiness",
]

# Genre dan mode tidak boleh dianggap sebagai angka biasa, sehingga diproses
# dengan OneHotEncoder sebelum digunakan oleh NearestNeighbors.
FITUR_KATEGORI = ["Genre", "Mode"]
KOLOM_METADATA = ["Song", "Artist", "Genre", "Year", "Popularity"]
KOLOM_DIBACA = KOLOM_METADATA + FITUR_NUMERIK + ["Mode"]


if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

print("1. Membaca dataset Spotify baru...")
df = pd.read_csv(DATA_PATH, usecols=KOLOM_DIBACA)
print(f"   Data awal: {len(df):,} lagu")

print("2. Membersihkan data...")
kolom_wajib = KOLOM_METADATA + FITUR_NUMERIK + FITUR_KATEGORI
df = df.dropna(subset=kolom_wajib).copy()
df = df.drop_duplicates(subset=["Song", "Artist"], keep="first")
df = df.reset_index(drop=True)
print(f"   Data unik dan valid: {len(df):,} lagu")

print("3. Memproses fitur numerik, genre, dan mode...")
preprocessor = ColumnTransformer(
    transformers=[
        ("numerik", StandardScaler(), FITUR_NUMERIK),
        (
            "kategori",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            FITUR_KATEGORI,
        ),
    ],
    verbose_feature_names_out=False,
)

X = df[FITUR_NUMERIK + FITUR_KATEGORI]
X_processed = preprocessor.fit_transform(X).astype("float32")

print("4. Membuat indeks kemiripan dengan Scikit-Learn...")
model = NearestNeighbors(
    n_neighbors=min(11, len(df)),
    algorithm="brute",
    metric="euclidean",
    n_jobs=-1,
)
model.fit(X_processed)

paket_model = {
    "model": model,
    "preprocessor": preprocessor,
    "fitur_numerik": FITUR_NUMERIK,
    "fitur_kategori": FITUR_KATEGORI,
    "jumlah_lagu": len(df),
    "metric": "euclidean",
    "versi": 2,
}

metadata = df[KOLOM_METADATA].rename(
    columns={
        "Song": "name",
        "Artist": "artists",
        "Genre": "genre",
        "Year": "year",
        "Popularity": "popularity",
    }
)

print("5. Menyimpan model dan metadata baru...")
joblib.dump(paket_model, MODEL_PATH, compress=3)
metadata.to_parquet(METADATA_PATH, index=False)

print("6. Menguji satu pencarian lagu terdekat...")
jarak, indeks = model.kneighbors(X_processed[0:1], n_neighbors=min(6, len(df)))
lagu_awal = metadata.iloc[0]
print(f"   Lagu contoh: {lagu_awal['name']} — {lagu_awal['artists']}")

for urutan, (idx, distance) in enumerate(zip(indeks[0][1:], jarak[0][1:]), start=1):
    lagu = metadata.iloc[idx]
    print(
        f"   {urutan}. {lagu['name']} — {lagu['artists']} "
        f"[{lagu['genre']}] (jarak: {distance:.4f})"
    )

print("\nTraining/indexing dataset baru selesai.")
print(f"Model    : {MODEL_PATH}")
print(f"Metadata : {METADATA_PATH}")
