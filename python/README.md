# HEXIMAP — Python Port

KH-9 Hexagon uydu görüntülerinden sayısal yükseklik modeli (DEM) üretimi için
Python pipeline'ı ve modern Streamlit arayüzü.

---

## Mimari

```
┌─────────────────────────────────────────┐
│           Arayüz Katmanı                │
│   app.py (Streamlit)  │  QGIS eklentisi │
└──────────────┬──────────────────────────┘
               │  yalnızca run_* çağırır
┌──────────────▼──────────────────────────┐
│         pipeline_runner.py              │
│   Köprü — core'a tek giriş noktası      │
└──┬───────┬───────┬────────┬─────────────┘
   │       │       │        │
 1_stitch 2_extract 3_georef 4_rasterize   shared/
```

`pipeline_runner.py` köprü katmanı, `importlib` ile rakamla başlayan
klasörlerden (`1_stitch/`, `2_extract/`, `3_georef/`, `4_rasterize/`)
modül yükler. Arayüz katmanları core koda hiç dokunmaz.

---

## Kurulum

```bash
git clone https://github.com/Ksatura/heximap.git
cd heximap/python

# Sanal ortam (önerilen)
python -m venv heximap-env
source heximap-env/bin/activate        # Windows: heximap-env\Scripts\activate

pip install -r requirements.txt
```

### Sistem bağımlılıkları

```bash
# Ubuntu / Debian
sudo apt install zenity          # dosya seçici diyaloğu
sudo apt install libgl1          # OpenCV için

# macOS — zenity yerine tkinter ile değiştirilebilir
brew install zenity
```

---

## Çalıştırma

```bash
cd heximap/python
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` açılır.

---

## Pipeline Adımları

### Adım 1 — Stitch
KH-9 her kareyi iki yarı (a/b) olarak tarar. Bu adım:
- Sol ve sağ GeoTIFF yarılarını okur
- ORB özellik eşleştirmesi + RANSAC ile hizalama hesaplar
- Birleştirilmiş görüntüyü `.npz` olarak kaydeder

**Giriş:** İki GeoTIFF (`.tif`) + her biri için 2 köşe noktası  
**Çıkış:** Birleştirilmiş görüntü (`.npz`)

### Adım 2 — Extract
Stereo görüntü çiftinden disparity haritası ve ham nokta bulutu üretir:
- Görüntüleri `Transform` matrisi ile hizalar
- `ext_filter_images`: histogram eşleştirme + CLAHE + Wiener filtresi
- `ext_stereo_rect`: ORB + epipolar rektifikasyon
- `ext_disparity`: SGBM stereo eşleştirme

**Giriş:** İki `.npz` (stitch çıktısı) — otomatik aktarılır  
**Çıkış:** Disparity haritası + rektifiye görüntü çifti

### Adım 3 — Georef
Ham nokta bulutunu coğrafi koordinat sistemine dönüştürür:
- Referans DEM ile ilk dönüşüm tahmini
- Bundle adjustment optimizasyonu
- UTM projeksiyon

**Giriş:** Disparity çıktısı + referans DEM (`.tif`) + UTM dilimi  
**Çıkış:** Jeoreferanslanmış nokta bulutu

### Adım 4 — Rasterize
Nokta bulutundan GeoTIFF DEM üretir:
- Opsiyonel: DEM temizleme, medyan filtresi, denoising
- Opsiyonel: ortofoto dışa aktarma

**Giriş:** Georef çıktısı + çıktı klasörü  
**Çıkış:** DEM GeoTIFF (`.tif`)

---

## Arayüz Özellikleri

- **Koyu tema** — IBM Plex Mono/Sans tipografi
- **Batch stitch** — dinamik kare çifti listesi, her çift bağımsız
- **Canvas köşe seçimi** — `streamlit-drawable-canvas` ile görüntü
  üzerinde 2 tıklama ile köşe koordinatı seçimi
- **Dosya seçici** — Zenity GUI diyaloğu, thread-safe (UI donmaz)
- **Pipeline thread** — Queue tabanlı log/progress aktarımı
- **GeoTIFF önizleme** — `rasterio out_shape` ile RAM dostu küçültme
- **DEM önizleme** — matplotlib `terrain` colormap ile koyu temalı

---

## Bellek Yönetimi

KH-9 görüntüleri ~60 000 × 33 000 piksel boyutundadır (~2 GB/kare).
Pipeline'da bellek tasarrufu için:

| Adım | Yöntem |
|------|--------|
| Önizleme | `rasterio out_shape` ile doğrudan küçük okuma |
| Extract pencere kırpma | Kullanıcı köşe seçimi + `MAX_PX=6000` sınırı |
| Filtre sonrası | `del` ile kaynak array hemen serbest bırakılır |

---

## Dosya Yapısı

```
python/
├── app.py                  # Streamlit arayüzü
├── pipeline_runner.py      # Köprü katmanı
├── requirements.txt
├── logo/
│   └── logo.png            # (opsiyonel) header logosu
├── 1_stitch/
│   └── sti_stitch.py
├── 2_extract/
│   ├── ext_read_sort.py
│   ├── ext_filter_images.py
│   ├── ext_stereo_rect.py
│   ├── ext_windows.py
│   └── ext_disparity.py
├── 3_georef/
│   └── ...
├── 4_rasterize/
│   └── ...
└── shared/
    ├── geo_optimize.py
    └── ...
```

---

## Bilinen Sınırlamalar

- Zenity yalnızca Linux/X11 ortamında çalışır. macOS/Windows için
  `tkinter` tabanlı alternatif gerekir.
- `MAX_PX=6000` sınırı ~16 GB RAM sistemler için optimize edilmiştir.
  Daha az RAM varsa değeri düşürün (`pipeline_runner.py`, `run_extract`).
- Georef adımı referans DEM gerektirir (ör. SRTM, Copernicus DEM).

---

## Katkı

[github.com/Ksatura/heximap](https://github.com/Ksatura/heximap)
