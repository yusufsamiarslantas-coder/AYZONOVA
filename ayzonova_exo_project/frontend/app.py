import os, json, io, requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8001")
st.set_page_config(page_title="Ayzonova — Exoplanet Classifier", page_icon="🪐", layout="wide")

# ---- Dil seçici ----
LANG = st.sidebar.selectbox("Language / Dil", ["Türkçe", "English"], index=0)
def T(tr, en): return tr if LANG == "Türkçe" else en

# ---- Basit stil ----
st.markdown("""
<style>
  .stApp {background: linear-gradient(180deg,#0b0f1a 0%, #111726 60%, #0b0f1a 100%); color: #e8ebf0;}
  .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
  .ayz-title {font-weight:700; font-size:1.5rem; margin: 0.2rem 0 0.8rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown(f"<div class='ayz-title'>🪐 Ayzonova — {T('Ötegezegen Sınıflandırıcı','Exoplanet Classifier')}</div>", unsafe_allow_html=True)
st.caption("NASA Space Apps 2025 — A World Away (AI)")

with st.sidebar:
    st.markdown("### API")
    st.write(f"URL: {API_URL}")
    if st.button(T("Sağlık Kontrolü","Health Check")):
        try:
            r = requests.get(f"{API_URL}/health", timeout=20)
            st.json(r.json())
        except Exception as e:
            st.error(str(e))
    st.info(T("CSV başlıkları /features ile bire bir aynı olmalı.",
              "CSV headers must exactly match /features."))

# ---- Meta verileri çek ----
try:
    meta = requests.get(f"{API_URL}/features", timeout=20).json()
    FEATURES = meta.get("features", [])
    CLASSES  = meta.get("classes", [])
    TARGET   = meta.get("target", "label")
except Exception as e:
    FEATURES, CLASSES, TARGET = [], [], "label"
    st.error(T(f"Özellik listesi alınamadı: {e}", f"Failed to fetch features: {e}"))

tab1, tab2, tab3, tab4 = st.tabs([
    T("Tahmin","Predict"),
    T("Işık Eğrisi","Light Curve"),
    T("Dünya ile Kıyas","Compare w/ Earth"),
    T("Modeli Geliştir","Improve Model")
])

# ==================== Tahmin ====================
with tab1:
    st.markdown(f"### {T('CSV Yükle ve Sınıflandır','Upload CSV and Classify')}")
    with st.expander(T("Gerekli sütunlar","Required columns"), expanded=False):
        st.code("\n".join(FEATURES) if FEATURES else "—")

    up = st.file_uploader(T("CSV/JSON yükle","Upload CSV/JSON"), type=["csv","json"])
    if up and st.button(T("Tahmin Et","Predict")):
        try:
            files = {"file": (up.name, up.getvalue())}
            r = requests.post(f"{API_URL}/predict", files=files, timeout=120)
            if not r.ok:
                try: st.error(r.json())
                except: st.error(r.text)
            else:
                res = r.json()["results"]
                df = pd.DataFrame([{
                    "prediction": x["prediction"],
                    **{f"proba_{k}": v for k, v in x["proba"].items()}
                } for x in res])
                st.markdown("#### " + T("Sonuçlar","Results"))
                st.dataframe(df, use_container_width=True)
                if len(res) == 1:
                    labels = list(res[0]["proba"].keys()); vals = list(res[0]["proba"].values())
                    fig = go.Figure(data=[go.Bar(x=labels, y=vals)])
                    fig.update_layout(
                        title=T("Sınıf Olasılıkları","Class Probabilities"),
                        xaxis_title=T("Sınıf","Class"), yaxis_title="P"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(str(e))

# ==================== Işık Eğrisi ====================
with tab2:
    st.markdown(f"### {T('Işık Eğrisi Çiz','Plot Light Curve')}")
    st.info(T("CSV time & flux (veya t/jd/btjd/bjd ve pdcsap_flux/sap_flux) sütunlarını içermelidir.",
              "CSV must contain time & flux (or t/jd/btjd/bjd and pdcsap_flux/sap_flux)."))
    lc = st.file_uploader(T("Işık eğrisi (CSV)","Light curve (CSV)"), type=["csv"], key="lc")
    if lc:
        try:
            df = pd.read_csv(lc)
            tcol = next((c for c in df.columns if c.lower() in ["time","t","jd","btjd","bjd"]), None)
            fcol = next((c for c in df.columns if c.lower() in ["flux","f","pdcsap_flux","sap_flux"]), None)
            if tcol and fcol:
                st.success(T(f"Bulunan sütunlar: {tcol}, {fcol}", f"Detected columns: {tcol}, {fcol}"))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df[tcol], y=df[fcol], mode="lines", name="flux"))
                fig.update_layout(title=T("Işık Eğrisi","Light Curve"), xaxis_title="time", yaxis_title="flux (norm)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(T("time/flux sütunları bulunamadı.","time/flux columns not found."))
        except Exception as e:
            st.error(T(f"Dosya okunamadı: {e}", f"Could not read file: {e}"))

# ==================== Dünya ile Kıyas ====================
with tab3:
    st.markdown(f"### {T('Gezegen Özelliklerini Dünya ile Kıyasla','Compare Planet Props vs Earth')}")
    st.info(T("Örnek sütunlar: pl_orbper (gün), pl_trandep (ppm), sy_dist (pc).",
              "Example columns: pl_orbper (days), pl_trandep (ppm), sy_dist (pc)."))
    props = st.file_uploader(T("Özellik CSV'si","Properties CSV"), type=["csv"], key="prop")
    if props:
        dfp = pd.read_csv(props)
        st.dataframe(dfp.head(), use_container_width=True)
        idx = st.number_input(T("Satır seç","Select row"), min_value=0, max_value=len(dfp)-1, value=0, step=1)
        row = dfp.iloc[int(idx)]

        EARTH = {"pl_orbper": 365.25, "pl_trandep_ppm": 84.0}  # referanslar
        c1, c2, c3 = st.columns(3)

        plp = row.get("pl_orbper", np.nan)
        if pd.notna(plp):
            ratio = float(plp) / EARTH["pl_orbper"]
            c1.metric(T("Yörünge Periyodu (gün)","Orbital Period (days)"),
                      f"{float(plp):.2f}", T(f"Dünya x{ratio:.2f}", f"Earth x{ratio:.2f}"))

        trp = row.get("pl_trandep", np.nan)
        if pd.notna(trp):
            ratio = float(trp) / EARTH["pl_trandep_ppm"]
            c2.metric(T("Transit Derinliği (ppm)","Transit Depth (ppm)"),
                      f"{float(trp):.1f}", T(f"Dünya x{ratio:.2f}", f"Earth x{ratio:.2f}"))

        dist = row.get("sy_dist", np.nan)
        if pd.notna(dist):
            c3.metric(T("Uzaklık (pc)","Distance (pc)"), f"{float(dist):.1f}")

        notes = []
        if pd.notna(plp):
            if float(plp) < 50: notes.append(T("Kısa periyot → yıldızına yakın olabilir.","Short period → possibly close-in orbit."))
            elif float(plp) > 500: notes.append(T("Uzun periyot → geniş yörünge.","Long period → wide orbit."))
        if pd.notna(trp):
            if float(trp) > 1000: notes.append(T("Derin transit → büyük yarıçap olasılığı.","Deep transit → likely larger radius."))
            elif float(trp) < 100: notes.append(T("Sığ transit → küçük yarıçap olasılığı (Dünya ~84 ppm).","Shallow transit → possibly small radius (Earth ~84 ppm)."))
        if not notes:
            notes.append(T("Kıyas için daha çok sütun ekleyebilirsin.","Add more columns for richer comparison."))
        st.markdown("\n".join([f"- {n}" for n in notes]))

# ==================== Modeli Geliştir ====================
with tab4:
    st.markdown("### " + T("Kullanıcı Verisi ile Geliştir","Improve with User Data"))
    st.info(T("Etiketli veri (features + label) yükle. Backend bunu data/user_feedback.csv içine yazar. "
              "Yeterli veri olunca '/retrain' ile basit bir model güncellemesi yapılır.",
              "Upload labeled data (features + label). Backend appends to data/user_feedback.csv. "
              "Once enough rows, call '/retrain' for a simple update."))
    st.caption(T("Beklenen etiket sütunu: ", "Expected label column: ") + TARGET)

    fb = st.file_uploader(T("Geri bildirim CSV/JSON","Feedback CSV/JSON"), type=["csv","json"], key="fb")
    if fb and st.button(T("Gönder","Submit")):
        try:
            files = {"file": (fb.name, fb.getvalue())}
            r = requests.post(f"{API_URL}/feedback", files=files, timeout=120)
            if r.ok: st.success(r.json())
            else:
                try: st.error(r.json())
                except: st.error(r.text)
        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    st.markdown("#### " + T("Basit Yeniden Eğitim","Simple Retrain"))
    if st.button(T("Retrain’i Çalıştır","Run Retrain")):
        try:
            r = requests.post(f"{API_URL}/retrain", timeout=600)
            if r.ok: st.success(r.json())
            else:
                try: st.error(r.json())
                except: st.error(r.text)
        except Exception as e:
            st.error(str(e))
