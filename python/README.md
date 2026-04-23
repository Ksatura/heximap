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
main/
├── shared/                          → Paylaşılan yardımcı modüller
│   ├── utils.py                     → Rotasyon, üçgenleme, dosya listeleme
│   ├── shared_utils.py              → Koordinat dönüşümleri, ızgara araçları
│   ├── grid2grid.py                 → Izgara yeniden örnekleme
│   └── estimate_transform_ransac.py → RANSAC + Horn dönüşüm tahmini
├── 1_stitch/                        → Hexagon görüntü yarılarını birleştirme
│   └── sti_stitch.py
├── 2_extract/                       → Stereo DEM çıkarma
│   ├── ext_filter_images.py
│   ├── ext_disparity.py
│   ├── ext_stereo_rect.py
│   ├── ext_init_bundle.py
│   ├── ext_read_sort.py
│   └── ext_windows.py
├── 3_georef/                        → Jeoreferanslama optimizasyonu
│   └── geo_optimize.py
└── 4_rasterize/                     → GeoTIFF dışa aktarma
    └── ras_export.py
```

## Modül Karşılıkları

| Python Dosyası | MATLAB Karşılığı | Entegrasyon Testi |
|---|---|---|
| `shared/utils.py` | `makeRotMat.m` · `triangulate.m` · `rotro2eu.m` · `getFiles.m` | `integration_test_utils.py` |
| `shared/shared_utils.py` | `blockProcess.m` · `polygons2grid.m` · `makeSpatialRefVecs.m` · `neighbors.m` · `shiftDem.m` · `readGeotiffRegion.m` · `ll2ps.m` · `ps2ll.m` · `checkInput.m` · `checkShpPath.m` · `colormapDEM.m` · `getFiles.m` | `integration_test_shared.py` |
| `shared/grid2grid.py` | `grid2grid.m` · `blockProcess.m` · `makeSpatialRefVecs.m` | `integration_test_grid2grid.py` |
| `shared/estimate_transform_ransac.py` | `estimateTransformRansac.m` · `absor.m` | `integration_test_ransac.py` |
| `1_stitch/sti_stitch.py` | `stiStitch.m` · `stiGetCorners.m` · `stiResize.m` · `stiSaveInfo.m` | `integration_test_stitch.py` |
| `2_extract/ext_filter_images.py` | `extFilterImages.m` | `integration_test_extract.py` |
| `2_extract/ext_disparity.py` | `extDisparity.m` · `extDisparityLoop.m` | `integration_test_extract.py` |
| `2_extract/ext_stereo_rect.py` | `extStereoRect.m` | `integration_test_extract.py` |
| `2_extract/ext_init_bundle.py` | `extInitTrans.m` · `extBundleAdjust.m` · `extBundleAdjustLoop.m` | `integration_test_extract.py` |
| `2_extract/ext_read_sort.py` | `extReadImage.m` · `extSortImages.m` | `integration_test_extract.py` |
| `2_extract/ext_windows.py` | `extGetROI.m` · `extChooseWindows.m` · `extControlPoints.m` | `integration_test_extract.py` |
| `3_georef/geo_optimize.py` | `geoOptimize.m` · `geoOptiTrans.m` · `geoInitTrans.m` · `geoSamplePoints.m` · `alignDem.m` · `optimizeDem.m` · `ll2utm.m` · `normals.m` · `points2grid.m` · `transformUsingSolverVar.m` | `integration_test_georef.py` |
| `4_rasterize/ras_export.py` | `rasDem.m` · `rasClean.m` · `rasSmooth.m` · `rasOrtho.m` · `meshDenoise.m` · `curvatureCorrection.m` | `integration_test_rasterize.py` |

## Entegrasyon Testlerini Çalıştırma

Tüm testler repo kökünden çalıştırılır:

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

## Önemli Düzeltmeler

Port sürecinde tespit edilen ve düzeltilen MATLAB kaynaklı hatalar:

| Modül | Hata | Düzeltme |
|---|---|---|
| `3_georef/geo_optimize.py` | `align_dem`: `geo_sample_points` sonrası `n_moving` indeksleme hatası; `pcd_ref` normalsiz oluşturuluyordu (`PointToPlane` ICP çöküyordu) | `n_moving = n_moving[idx]` eklendi; `pcd_ref.normals` atandı |
| `4_rasterize/ras_export.py` | `ras_ortho`: `sT['trans']` anahtarı eksikti | `win_obj` yapısına `'trans': np.eye(4)` eklendi |
| `shared/shared_utils.py` | `ps2ll`: kuzey yarımküre boylam sarmalama (`180° → -180°` çakışması) | `np.mod(lon + π, 2π) − π` her iki yarımküreye de uygulandı |

## OpenCV Sürümü

Orijinal HEXIMAP yalnızca OpenCV 3.4.1 + mexopencv ile çalışıyor.
Bu port **OpenCV 4.8+** kullanıyor — tüm API çağrıları güncellendi.

## Referanslar

- Maurer, J., & Rupper, S. (2015). Automation of ice surface velocity measurements in raw Hexagon KH-9 imagery. *ISPRS Journal of Photogrammetry and Remote Sensing*, 108, 113–127.
- Maurer et al. (2019). Serac ice avalanches from the Iliamna Volcano (Alaska) mapped and quantified from time series of very high-resolution satellite imagery. *Science Advances*, 5(6), eaav7266.
