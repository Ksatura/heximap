# HEXIMAP — Python Port

Bu klasör, orijinal MATLAB tabanlı HEXIMAP uygulamasının Python portunu içermektedir.

## Amaç

Orijinal HEXIMAP; MATLAB 2020b, mexopencv ve Visual Studio 2015 gerektiriyor.
Bu port aşağıdaki hedeflerle geliştirilmektedir:

- **Platform:** Linux Mint / Ubuntu (Debian tabanlı)
- **Dil:** Python 3.10+
- **MATLAB yerine:** NumPy, SciPy, OpenCV (cv2), scikit-image
- **mexopencv yerine:** `opencv-python` (doğrudan `cv2` çağrıları)
- **Visual Studio yerine:** Derleme adımı yok — `pip install` yeterli

## Kurulum

```bash
# 1. Sistem bağımlılıkları
sudo apt install -y python3-pip python3-venv gdal-bin libgdal-dev

# 2. Sanal ortam oluştur
python3 -m venv ~/heximap-env
source ~/heximap-env/bin/activate

# 3. Python paketleri
pip install -r requirements.txt
```

## Klasör Yapısı

```
python/
├── shared/          → Paylaşılan yardımcı modüller
├── 1_stitch/        → Hexagon görüntü yarılarını birleştirme
├── 2_extract/       → Stereo DEM çıkarma
├── 3_georef/        → Jeoreferanslama optimizasyonu
├── 4_rasterize/     → GeoTIFF dışa aktarma
└── tests/           → Birim ve entegrasyon testleri
```

## Modül Karşılıkları

| Python Dosyası | MATLAB Karşılığı |
|---|---|
| `shared/estimate_transform_ransac.py` | `estimateTransformRansac.m` + `absor.m` |
| `shared/grid2grid.py` | `grid2grid.m` + `blockProcess.m` |
| `shared/utils.py` | `makeRotMat.m` + `triangulate.m` + `rotro2eu.m` + `getFiles.m` |
| `shared/geo_optimize.py` | `geoOptiTrans.m` + `alignDem.m` + `optimizeDem.m` + diğerleri |
| `1_stitch/sti_stitch.py` | `stiStitch.m` |
| `2_extract/ext_filter_images.py` | `extFilterImages.m` |
| `2_extract/ext_disparity.py` | `extDisparity.m` |
| `2_extract/ext_stereo_rect.py` | `extStereoRect.m` |
| `2_extract/ext_init_bundle.py` | `extInitTrans.m` + `extBundleAdjust.m` |
| `2_extract/ext_windows.py` | `extGetROI.m` + `extControlPoints.m` |
| `3_georef/geo_optimize.py` | `geoOptiTrans.m` + `geoInitTrans.m` + `geoOptimize.m` |
| `4_rasterize/ras_export.py` | `rasDem.m` + `rasClean.m` + `rasSmooth.m` + `rasOrtho.m` |

## OpenCV Sürümü

Orijinal HEXIMAP yalnızca OpenCV 3.4.1 + mexopencv ile çalışıyor.
Bu port **OpenCV 4.8+** kullanıyor — tüm API çağrıları güncellendi.

## Durum

| Aşama | Durum |
|---|---|
| Birim testler | ✅ Tamamlandı |
| Stitch entegrasyon testi | 🔄 Devam ediyor |
| Extract entegrasyon testi | ⏳ Bekliyor |
| Georef entegrasyon testi | ⏳ Bekliyor |
| Rasterize entegrasyon testi | ⏳ Bekliyor |
| PyQt5 GUI | ⏳ Planlandı |

## Referanslar

- Maurer, J., & Rupper, S. (2015). ISPRS Journal of Photogrammetry and Remote Sensing, 108, 113–127.
- Maurer et al. (2019). Science Advances, 5(6), eaav7266.
