import numpy as np
import streamlit as st

from inference import load_model, predict_sentiment


# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen Kesehatan Mental Ibu Hamil di Indonesia🧠",
    page_icon="🧠",
    layout="centered",
)


@st.cache_resource(show_spinner="Sedang memuat model IndoBERT, tunggu sebentar...")
def get_model():
    tokenizer, model, device = load_model()
    return tokenizer, model, device


def format_prob(p: float) -> str:
    """Ubah probabilitas jadi persen rapi, misal 0.873 → '87.3%'."""
    return f"{p * 100:.1f}%"


def main():
    st.markdown("## 🧠 Analisis Sentimen Teks (Kesehatan Mental Ibu Hamil di Indonesia🧠)")

    st.markdown(
        """
        Aplikasi ini membaca **teks berbahasa Indonesia** lalu mencoba
        menebak apakah isi kalimat tersebut bernada **positif** atau **negatif**.

        Cukup tulis kalimat di bawah, lalu tekan tombol **Analisis Sentimen**.
        """
    )

    st.divider()

    # Kolom utama: input di kiri, penjelasan singkat di kanan
    col_input, col_info = st.columns([2, 1])

    with col_input:
        st.markdown("### ✍️ Tulis teks di sini")

        default_text = "mertua saya selalu bikin saya sedih dan suami saya lagi kerja di luar kota"
        text = st.text_area(
            "Teks yang ingin dianalisis",
            value=default_text,
            height=140,
            placeholder="Contoh: saya bahagia karena selama saya hamil suami saya peduli dengan saya",
        )

    with col_info:
        st.markdown("### ℹ️ Petunjuk singkat")
        st.markdown(
            """
            - Tulis kalimat yang **alami**, seperti ulasan, komentar, atau opini.  
            - Model hanya mengenali dua jenis sentimen:
              - 😊 **Positif**  
              - 😞 **Negatif**  
            - Hasil bukan “benar-salah mutlak”, tapi **perkiraan model**.
            """
        )

    # Pengaturan teknis kita sembunyikan di expander
    with st.expander("⚙️ Pengaturan lanjutan (opsional)"):
        max_len = st.slider(
            "Panjang maksimal teks yang diproses model (dalam token)",
            min_value=64,
            max_value=256,
            value=128,
            step=16,
            help="Untuk kebanyakan kasus, 128 sudah cukup. Naikkan hanya jika teks sangat panjang.",
        )
    # default jika expander tidak dibuka
    if "max_len" not in locals():
        max_len = 128

    st.divider()

    # Tombol aksi
    analyze = st.button("🔍 Analisis Sentimen", use_container_width=True)

    if analyze:
        if not text or text.strip() == "":
            st.warning("Teks masih kosong. Silakan isi kalimat terlebih dahulu.")
            return

        tokenizer, model, device = get_model()

        with st.spinner("Model sedang membaca teks kamu..."):
            result = predict_sentiment(
                text=text,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=max_len,
            )

        label = result["label_name"]  # "Negatif" / "Positif"
        probs = result["probs"]       # [p_neg, p_pos]

        p_neg, p_pos = float(probs[0]), float(probs[1])

        # Blok hasil utama: dibuat sangat "manusiawi"
        if label.lower().startswith("positif"):
            st.success(
                f"**Perkiraan model: Sentimen POSITIF 😊**\n\n"
                f"Model cukup yakin bahwa kalimat ini bernada **positif**.\n"
                f"• Perkiraan positif: **{format_prob(p_pos)}**\n"
                f"• Perkiraan negatif: {format_prob(p_neg)}",
                icon="✅",
            )
        else:
            st.error(
                f"**Perkiraan model: Sentimen NEGATIF 😞**\n\n"
                f"Model membaca kalimat ini cenderung bernada **negatif**.\n"
                f"• Perkiraan negatif: **{format_prob(p_neg)}**\n"
                f"• Perkiraan positif: {format_prob(p_pos)}",
                icon="⚠️",
            )

        st.markdown("---")

        # Visual probabilitas versi sederhana
        st.markdown("### 🎯 Tingkat keyakinan model")

        chart_data = {
            "Sentimen": ["Negatif (0)", "Positif (1)"],
            "Probabilitas": [p_neg, p_pos],
        }

        st.bar_chart(
            data={
                "Negatif (0)": [p_neg],
                "Positif (1)": [p_pos],
            }
        )

        # Detail teknis disembunyikan, hanya untuk user yang penasaran
        with st.expander("🔍 Detail teknis (opsional)"):
            st.write("Logits (sebelum softmax):", result["logits"])
            st.write(
                "Probabilitas mentah:",
                {
                    "negatif (0)": p_neg,
                    "positif (1)": p_pos,
                },
            )
            st.caption(
                "Bagian ini lebih bersifat teknis dan biasanya hanya dipakai untuk debugging atau analisis model."
            )


if __name__ == "__main__":
    main()
