# HEXIMAP — Python Port <img src='/python/logo/logo.png' align="right" height="200" />

Bu klasör, orijinal MATLAB tabanlı HEXIMAP uygulamasının Python portunu ve
**Streamlit tabanlı masaüstü arayüzünü** içermektedir.

## Amaç

Orijinal HEXIMAP; MATLAB 2020b, mexopencv ve Visual Studio 2015 gerektiriyor.
Bu port aşağıdaki hedeflerle geliştirilmektedir:

- **Platform:** Linux Mint / Ubuntu (Debian tabanlı) · Windows · macOS
- **Dil:** Python 3.10+
- **MATLAB yerine:** NumPy, SciPy, OpenCV (cv2), scikit-image
- **mexopencv yerine:** `opencv-python` (doğrudan `cv2` çağrıları)
- **Visual Studio yerine:** Derleme adımı yok — `pip install` yeterli
- **Arayüz:** Streamlit masaüstü uygulaması (localhost, tarayıcıda açılır)

## Kurulum

```bash
# 1. Sistem bağımlılıkları (Linux)
sudo apt install -y python3-pip python3-venv gdal-bin libgdal-dev zenity

# 2. Sanal ortam oluştur
python3 -m venv ~/heximap-env
source ~/heximap-env/bin/activate

# 3. Python paketleri
pip install -r requirements.txt
```

### requirements.txt'e eklenmesi gereken yeni bağımlılıklar

```
streamlit>=1.32
streamlit-drawable-canvas-fix
Pillow>=10.0
rasterio>=1.3
matplotlib>=3.7
```

> **Not:** `streamlit-drawable-canvas` (orijinal) Streamlit 1.32+ ile uyumsuz.
> Yerine `streamlit-drawable-canvas-fix` kullanılmaktadır.

## Arayüzü Çalıştırma

```bash
source ~/heximap-env/bin/activate
cd ~/heximap/python
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` otomatik açılır.

## Klasör Yapısı

```
python/
├── app.py                           → Streamlit masaüstü arayüzü
├── pipeline_runner.py               → Arayüz ↔ core köprü katmanı
├── shared/                          → Paylaşılan yardımcı modüller
│   ├── utils.py
│   ├── shared_utils.py
│   ├── grid2grid.py
│   ├── estimate_transform_ransac.py
│   └── geo_optimize.py              → Georef optimizasyonu (shared/ altında)
├── 1_stitch/                        → Hexagon görüntü yarılarını birleştirme
│   └── sti_stitch.py
├── 2_extract/                       → Stereo DEM çıkarma
│   ├── ext_filter_images.py
│   ├── ext_disparity.py
│   ├── ext_stereo_rect.py
│   ├── ext_init_bundle.py
│   ├── ext_read_sort.py
│   └── ext_windows.py
├── 3_georef/                        → (geo_optimize.py → shared/ altına taşındı)
├── 4_rasterize/                     → GeoTIFF dışa aktarma
│   └── ras_export.py
└── tests/                           → Entegrasyon testleri
```

## Mimari: İki Katmanlı Arayüz

```
┌─────────────────────────────────────────────────────┐
│           CORE PIPELINE  (değişmez)                 │
│  shared/ · 1_stitch/ · 2_extract/ · 4_rasterize/   │
└────────────────────┬────────────────────────────────┘
                     │
         pipeline_runner.py  ← tek köprü noktası
         run_stitch() · run_extract()
         run_georef() · run_rasterize()
         run_pipeline()  →  RunResult
                     │
          ┌──────────┴──────────┐
          │                     │
      app.py               heximap_plugin/   (planlandı)
   Streamlit UI            QGIS Eklentisi
```

**Kural:** `pipeline_runner.py` hiçbir Streamlit veya QGIS import'u içermez.
Arayüz katmanı hiçbir zaman `shared/` veya `extract/` modüllerini doğrudan çağırmaz.

## pipeline_runner.py

Core pipeline ile arayüz katmanı arasındaki köprü. Her adım için:

- `run_stitch(params, log_cb, progress_cb) → RunResult`
- `run_extract(params, log_cb, progress_cb) → RunResult`
- `run_georef(params, log_cb, progress_cb) → RunResult`
- `run_rasterize(params, log_cb, progress_cb) → RunResult`
- `run_pipeline(...)  → dict[str, RunResult]`

`RunResult` alanları: `success: bool · output: dict · error: str · tb: str`

### Teknik notlar

- Rakamla başlayan klasörler (`1_stitch`, `2_extract` vb.) Python `import`
  sistemiyle doğrudan yüklenemez; `importlib.util` ile dosya yolundan yüklenir.
- `corners` parametresi: kullanıcı arayüzünden `[[x_üst,y_üst],[x_alt,y_alt]]`
  listesi gelir, `pipeline_runner` bunu `numpy.ndarray shape (2,2)`'ye çevirir.
- `sti_stitch` için `path` = görüntü dosyalarının bulunduğu klasör
  (çıktı klasörü değil); `file_l/file_r` = yalnızca dosya adı.

## app.py — Streamlit Arayüzü

### Özellikler

| Özellik | Detay |
|---|---|
| Batch stitch | "Çift Ekle" ile istediğiniz kadar (a+b) kare çifti |
| GeoTIFF önizleme | `rasterio out_shape` ile RAM dostu küçültme |
| Köşe seçimi | `streamlit-drawable-canvas-fix` nokta modu — 2 tıklama |
| Koordinat dönüşümü | Önizleme pikseli → orijinal GeoTIFF pikseli |
| Dosya seçici | `zenity` thread'de çalışır, Streamlit donmaz |
| Pipeline thread | `Queue` tabanlı log/progress, 0.3s polling |
| DEM önizleme | `matplotlib terrain` colormap, koyu tema |
| Adım kontrolü | Her adım sidebar'dan açılıp kapatılabilir |

### Kullanım akışı

1. Sidebar'dan proje klasörünü ve aktif adımları seçin
2. **Adım 1:** Her kare için sol (a) + sağ (b) GeoTIFF yollarını girin
3. "Önizleme Yükle" → önizleme üzerinde 2 köşe tıklayın (üst-sol, alt-sağ)
4. **Adım 2:** Birleştirilmiş `.npz` dosya yollarını girin
5. **Adım 3:** Referans DEM yolunu girin, UTM dilimini seçin
6. **Adım 4:** Çıktı klasörünü belirleyin
7. "▶ Pipeline'ı Çalıştır" butonuna basın

### Dosya seçici (Zenity)

Linux'ta `zenity` kullanılır. Kurulu değilse:

```bash
sudo apt install zenity
```

Zenity ayrı bir thread'de başlatılır — `📂` butonu `⏳` olarak değişir,
seçim tamamlanınca yol otomatik olarak ilgili alana yansır.

## Modül Karşılıkları

| Python Dosyası | MATLAB Karşılığı |
|---|---|
| `shared/utils.py` | `makeRotMat.m · triangulate.m · rotro2eu.m · getFiles.m` |
| `shared/shared_utils.py` | `blockProcess.m · polygons2grid.m · makeSpatialRefVecs.m · neighbors.m · shiftDem.m · readGeotiffRegion.m · ll2ps.m · ps2ll.m · checkInput.m · checkShpPath.m · colormapDEM.m · getFiles.m` |
| `shared/grid2grid.py` | `grid2grid.m · blockProcess.m · makeSpatialRefVecs.m` |
| `shared/estimate_transform_ransac.py` | `estimateTransformRansac.m · absor.m` |
| `shared/geo_optimize.py` | `geoOptimize.m · geoOptiTrans.m · geoInitTrans.m · geoSamplePoints.m · alignDem.m · optimizeDem.m · ll2utm.m · normals.m · points2grid.m · transformUsingSolverVar.m` |
| `1_stitch/sti_stitch.py` | `stiStitch.m · stiGetCorners.m · stiResize.m · stiSaveInfo.m` |
| `2_extract/ext_filter_images.py` | `extFilterImages.m` |
| `2_extract/ext_disparity.py` | `extDisparity.m · extDisparityLoop.m` |
| `2_extract/ext_stereo_rect.py` | `extStereoRect.m` |
| `2_extract/ext_init_bundle.py` | `extInitTrans.m · extBundleAdjust.m · extBundleAdjustLoop.m` |
| `2_extract/ext_read_sort.py` | `extReadImage.m · extSortImages.m` |
| `2_extract/ext_windows.py` | `extGetROI.m · extChooseWindows.m · extControlPoints.m` |
| `4_rasterize/ras_export.py` | `rasDem.m · rasClean.m · rasSmooth.m · rasOrtho.m · meshDenoise.m · curvatureCorrection.m` |

## Entegrasyon Testlerini Çalıştırma

```bash
source ~/heximap-env/bin/activate

python3 integration_test_stitch.py
python3 integration_test_extract.py
python3 integration_test_georef.py
python3 integration_test_rasterize.py
python3 integration_test_shared.py
python3 integration_test_ransac.py
python3 integration_test_grid2grid.py
python3 integration_test_utils.py
```

## Test Durumu

| Modül | Test Dosyası | Sonuç |
|---|---|---|
| `1_stitch/sti_stitch.py` | `integration_test_stitch.py` | ✅ Tamamlandı |
| `2_extract/ext_*.py` | `integration_test_extract.py` | ✅ Tamamlandı |
| `3_georef/geo_optimize.py` | `integration_test_georef.py` | ✅ 30/30 |
| `4_rasterize/ras_export.py` | `integration_test_rasterize.py` | ✅ 29/29 |
| `shared/shared_utils.py` | `integration_test_shared.py` | ✅ 46/46 |
| `shared/estimate_transform_ransac.py` | `integration_test_ransac.py` | ✅ 34/34 |
| `shared/grid2grid.py` | `integration_test_grid2grid.py` | ✅ 37/37 |
| `shared/utils.py` | `integration_test_utils.py` | ✅ 34/34 |

**Toplam: 210+ test, tümü geçiyor.**

## Düzeltilen Hatalar

### Port sürecinde tespit edilen MATLAB kaynaklı hatalar

| Modül | Hata | Düzeltme |
|---|---|---|
| `3_georef/geo_optimize.py` | `align_dem`: `n_moving` indeksleme hatası; `pcd_ref` normalsiz oluşturuluyordu | `n_moving = n_moving[idx]` eklendi; `pcd_ref.normals` atandı |
| `4_rasterize/ras_export.py` | `ras_ortho`: `sT['trans']` anahtarı eksikti | `win_obj` yapısına `'trans': np.eye(4)` eklendi |
| `shared/shared_utils.py` | `ps2ll`: kuzey yarımküre boylam sarmalama hatası | `np.mod(lon + π, 2π) − π` her iki yarımküreye uygulandı |

### pipeline_runner.py geliştirme sürecinde tespit edilenler

| Sorun | Düzeltme |
|---|---|
| `corners` liste olarak gönderiliyordu, `sti_stitch` NumPy array bekliyordu | `np.array(corners, dtype=float)` ile dönüşüm eklendi |
| `corners` 4 nokta geliyordu, fonksiyon `shape (2,2)` bekliyor | 2 nokta formatına (`[[x_üst,y_üst],[x_alt,y_alt]]`) geçildi |
| `path` parametresi çıktı klasörü olarak geçiliyordu | `file_l_path.parent` ile kaynak klasör kullanıldı |
| `geo_optimize.py` `3_georef/` yerine `shared/` altında | Import yolu `shared.geo_optimize` olarak güncellendi |

## OpenCV Sürümü

Orijinal HEXIMAP yalnızca OpenCV 3.4.1 + mexopencv ile çalışıyor.
Bu port **OpenCV 4.8+** kullanıyor — tüm API çağrıları güncellendi.

## Planlanan: QGIS Eklentisi

`pipeline_runner.py` QGIS eklentisi için de hazır. Planlanan mimari:

- `QDialog` / `QDockWidget` ile form arayüzü
- `QgsTask` ile arka plan işlem (UI donmaz)
- Çıktı GeoTIFF'ler doğrudan QGIS katmanına eklenir
- Plugin Manager zip kurulum (Win/Mac/Linux)

## Referanslar

- Maurer, J., & Rupper, S. (2015). Automation of ice surface velocity measurements
  in raw Hexagon KH-9 imagery. *ISPRS Journal of Photogrammetry and Remote Sensing*,
  108, 113–127.
- Maurer et al. (2019). Serac ice avalanches from the Iliamna Volcano (Alaska)
  mapped and quantified from time series of very high-resolution satellite imagery.
  *Science Advances*, 5(6), eaav7266.
