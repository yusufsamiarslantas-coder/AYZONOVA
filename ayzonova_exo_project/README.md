
# Ayzonova Exoplanet AI — Quick Start

## 0) Ön Koşullar
- Python 3.10+ (Mac: `python3`, Windows: `python`)
- Git (opsiyonel)
- (Opsiyonel) Docker / Docker Compose

## 1) Klasörü Aç
Bu zip'i bir klasöre çıkar: `ayzonova_exo_project/`

```
ayzonova_exo_project/
  backend/
  frontend/
  models/
  sample_data/
  docker-compose.yml
```

## 2) Model Dosyaları
`models/` içinde **model.joblib** ve **model_card.json** olmalı.
Bu paket, sohbetten otomatik kopyaladı. Kendi güncel modellerini buraya koyabilirsin.

## 3) Backend'i Çalıştır
### a) Sanal ortam (önerilir)
Mac/Linux:
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```
Windows (PowerShell):
```powershell
cd backend
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

### b) Sağlık kontrolü
Tarayıcıda aç:
- http://localhost:8001/health
- http://localhost:8001/features

## 4) Frontend'i Çalıştır
Yeni bir terminal aç.
### Mac/Linux
```bash
cd frontend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```
### Windows
```powershell
cd frontend
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Tarayıcı: http://localhost:8501

## 5) Test Et
- Sidebar "Health Check" butonuna bas.
- `sample_data/sample_exoplanet_inputs.csv` dosyasını yükle.
- "Tahmin Et" butonuna bas.

## 6) Docker ile Tek Komutta (Opsiyonel)
Kök klasörde:
```bash
docker compose up --build
```
- Backend: http://localhost:8001
- Frontend: http://localhost:8501

## 7) Sık Hatalar
- `zsh: command not found: python` → Mac'te `python3` kullan.
- 422 / 400 → CSV sütunları `model_card.json` içindeki `features` ile **aynı adlarda** olmalı.
- 500 → Model `predict_proba` sağlamıyorsa backend pseudo-olasılık için `predict`'e düşer; yine de sonuç döner.
- Boş sonuç / NaN → CSV'de metin kalan sütunlar sayısala çevrilir; kalan NaN'lar median ile doldurulur.

## 8) Model Kartı Beklentisi
`models/model_card.json` en az `{"features": [...], "classes": [...]}` anahtarlarını içermeli.

## 9) Proje Sunumu İpuçları
- README'ye veri kaynağını (NASA Exoplanet Archive) ve eğitim özetini ekle.
- Kısa bir ekran kaydı ile demo hazırla.
- Kod deposu bağlantısını proje sayfasına koy.
