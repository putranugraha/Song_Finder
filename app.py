from ast import literal_eval
from difflib import get_close_matches
from pathlib import Path
from urllib.parse import quote_plus

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "song_similarity_model.joblib"
METADATA_PATH = BASE_DIR / "song_metadata.parquet"

st.set_page_config(
    page_title="AI Song Finder",
    page_icon="🎵",
    layout="centered",
)


def format_artists(value):
    """Ubah teks seperti "['Artist A', 'Artist B']" menjadi teks biasa."""
    try:
        artists = literal_eval(str(value))
        if isinstance(artists, list):
            return ", ".join(str(artist) for artist in artists)
    except (ValueError, SyntaxError):
        pass
    return str(value)


@st.cache_resource(show_spinner="Memuat model dan katalog lagu...")
def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata tidak ditemukan: {METADATA_PATH}")

    paket = joblib.load(MODEL_PATH)
    metadata = pd.read_parquet(METADATA_PATH)

    if len(metadata) != paket["model"].n_samples_fit_:
        raise ValueError("Jumlah baris metadata tidak sama dengan data model.")

    # Kolom ini hanya digunakan untuk pencarian tanpa membedakan huruf besar.
    metadata["_search_name"] = metadata["name"].fillna("").str.casefold()
    metadata["_search_artist"] = metadata["artists"].fillna("").str.casefold()
    return paket, metadata


@st.cache_data(show_spinner=False, max_entries=100)
def search_song(query, limit=30):
    query_normal = query.strip().casefold()
    names = metadata["_search_name"]
    artists = metadata["_search_artist"]

    # Prioritas: judul persis, artis persis, awalan judul/artis, lalu teks yang
    # mengandung query. Satu indeks hanya boleh muncul sekali.
    kelompok_indeks = [
        metadata.index[names.eq(query_normal)].tolist(),
        metadata.index[artists.eq(query_normal)].tolist(),
        metadata.index[names.str.startswith(query_normal)].tolist(),
        metadata.index[artists.str.startswith(query_normal)].tolist(),
        metadata.index[
            names.str.contains(query_normal, regex=False)
            | artists.str.contains(query_normal, regex=False)
        ].tolist(),
    ]

    hasil = []
    sudah_ada = set()
    for kelompok in kelompok_indeks:
        for idx in kelompok:
            if idx not in sudah_ada:
                hasil.append(idx)
                sudah_ada.add(idx)
            if len(hasil) == limit:
                return hasil

    # Jika tidak ditemukan, coba koreksi salah eja ringan pada nama artis atau
    # judul. Contoh: "Justin Beaber" tetap menemukan "Justin Bieber".
    if not hasil:
        pilihan_artis = artists.drop_duplicates().tolist()
        artis_mirip = get_close_matches(
            query_normal, pilihan_artis, n=3, cutoff=0.65
        )
        for artist in artis_mirip:
            hasil.extend(metadata.index[artists.eq(artist)].tolist())

        if not hasil:
            pilihan_judul = names.drop_duplicates().tolist()
            judul_mirip = get_close_matches(
                query_normal, pilihan_judul, n=10, cutoff=0.65
            )
            for judul in judul_mirip:
                hasil.extend(metadata.index[names.eq(judul)].tolist())

    return hasil[:limit]


def song_label(index):
    lagu = metadata.iloc[index]
    artist = format_artists(lagu["artists"])
    year = int(lagu["year"]) if pd.notna(lagu["year"]) and lagu["year"] > 0 else "?"
    return f"{lagu['name']} — {artist} ({year})"


try:
    paket_model, metadata = load_resources()
except (FileNotFoundError, ValueError, KeyError) as error:
    st.error(str(error))
    st.info("Jalankan `py -3.13 latih_ai_lagu.py` untuk membuat ulang model.")
    st.stop()

model = paket_model["model"]

st.title("🎵 AI Song Finder")
st.write(
    "Cari sebuah judul lagu, lalu AI akan merekomendasikan lagu dengan "
    "karakter audio yang paling mirip."
)
st.caption(
    f"Katalog berisi {paket_model['jumlah_lagu']:,} lagu. "
    "Kemiripan dihitung dari genre, mode, energy, danceability, tempo, dan fitur audio lainnya."
)

query = st.text_input(
    "Cari judul lagu atau nama artis",
    placeholder="Contoh: Baby atau Justin Bieber",
    help="Masukkan minimal 2 karakter dari judul lagu atau nama artis.",
)

if not query.strip():
    st.info("Masukkan judul lagu atau nama artis untuk memulai pencarian.")
    st.stop()

if len(query.strip()) < 2:
    st.warning("Masukkan minimal 2 karakter agar hasil pencarian lebih spesifik.")
    st.stop()

hasil_pencarian = search_song(query)

if not hasil_pencarian:
    st.warning(
        "Lagu atau artis tidak ditemukan. Coba gunakan sebagian nama atau periksa ejaannya."
    )
    st.stop()

selected_index = st.selectbox(
    "Pilih lagu dan artis yang benar",
    options=hasil_pencarian,
    format_func=song_label,
)

jumlah_rekomendasi = st.slider(
    "Jumlah rekomendasi",
    min_value=5,
    max_value=15,
    value=5,
)

lagu_terpilih = metadata.iloc[selected_index]
with st.container(border=True):
    st.markdown(f"#### {lagu_terpilih['name']}")
    st.write(f"**Artis:** {format_artists(lagu_terpilih['artists'])}")
    st.write(f"**Genre:** {lagu_terpilih['genre']}")
    st.write(
        f"**Tahun:** {int(lagu_terpilih['year'])} · "
        f"**Popularitas:** {int(lagu_terpilih['popularity'])}/100"
    )

tombol_rekomendasi = st.button(
    "Cari Rekomendasi Seimbang",
    type="primary",
    use_container_width=True,
)

if tombol_rekomendasi:
    # Ambil lebih banyak kandidat berdasarkan audio, lalu urutkan ulang dengan
    # popularitas dan bonus artis yang sama.
    fitur_lagu = model._fit_X[selected_index : selected_index + 1]
    jumlah_kandidat = min(max(50, jumlah_rekomendasi * 5) + 1, len(metadata))
    jarak, indeks = model.kneighbors(
        fitur_lagu,
        n_neighbors=jumlah_kandidat,
    )

    artis_sumber = format_artists(lagu_terpilih["artists"]).casefold()
    kandidat = []

    for idx, distance in zip(indeks[0], jarak[0]):
        if idx == selected_index:
            continue

        lagu = metadata.iloc[idx]
        audio_score = 1 / (1 + float(distance))
        popularity_score = max(0.0, min(1.0, float(lagu["popularity"]) / 100))
        same_artist = (
            format_artists(lagu["artists"]).casefold() == artis_sumber
        )

        balanced_score = (
            audio_score * 0.70
            + popularity_score * 0.20
            + float(same_artist) * 0.10
        )

        kandidat.append(
            {
                "index": idx,
                "audio_score": audio_score,
                "popularity_score": popularity_score,
                "same_artist": same_artist,
                "balanced_score": balanced_score,
            }
        )

    rekomendasi = sorted(
        kandidat,
        key=lambda item: item["balanced_score"],
        reverse=True,
    )[:jumlah_rekomendasi]

    st.divider()
    st.subheader("Rekomendasi Seimbang")
    st.caption(
        "Peringkat dihitung dari 70% kemiripan audio, 20% popularitas, dan "
        "10% bonus jika artisnya sama. Skor ini bukan probabilitas."
    )

    for urutan, hasil in enumerate(rekomendasi, start=1):
        lagu = metadata.iloc[hasil["index"]]
        skor_seimbang = hasil["balanced_score"] * 100

        with st.container(border=True):
            info, skor = st.columns([4, 1])
            with info:
                st.markdown(f"#### {urutan}. {lagu['name']}")
                st.write(f"**Artis:** {format_artists(lagu['artists'])}")
                st.caption(
                    f"Genre: {lagu['genre']} · Tahun: {int(lagu['year'])} · "
                    f"Popularitas: {int(lagu['popularity'])}/100"
                )
                bonus_artis = "Ya (+10)" if hasil["same_artist"] else "Tidak"
                st.caption(
                    f"Audio: {hasil['audio_score'] * 100:.1f}/100 · "
                    f"Bonus artis sama: {bonus_artis}"
                )
                spotify_query = quote_plus(f"{lagu['name']} {lagu['artists']}")
                st.link_button(
                    "Cari di Spotify",
                    f"https://open.spotify.com/search/{spotify_query}",
                )
            with skor:
                st.metric("Skor seimbang", f"{skor_seimbang:.1f}/100")
