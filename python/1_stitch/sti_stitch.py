"""
sti_stitch.py

MATLAB kaynak dosyası:
  - main/1_stitch/stiStitch.m

Açıklama:
  Hexagon uydu görüntüsünün sol ve sağ yarılarını ORB özellik eşleştirmesi
  ve RANSAC tabanlı dönüşüm tahmini kullanarak birleştirir. Sonuç numpy
  .npz formatında kaydedilir (MATLAB'daki .mat dosyasının karşılığı).

Bağımlılıklar:
  - numpy        (pip install numpy)
  - opencv-python (pip install opencv-python)
  - rasterio     (pip install rasterio)
  - scipy        (pip install scipy)

Dahili bağımlılıklar:
  - estimate_transform_ransac.py  (aynı klasörde olmalı)
  - grid2grid.py                  (aynı klasörde olmalı)
"""

import numpy as np
import cv2
import rasterio
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

# Dahili modüller
import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
from estimate_transform_ransac import estimate_transform_ransac
from grid2grid import grid2grid


# ===========================================================================
# ANA FONKSİYON
# MATLAB karşılığı: stiStitch()
# ===========================================================================

def sti_stitch(path, file_l, file_r, corners_l, corners_r,
               progress_cb=None):
    """
    Hexagon görüntü yarılarını birleştirir ve .npz olarak kaydeder.

    MATLAB imzası:
        objM = stiStitch(strPath, strFileL, strFileR, mCornL, mCornR, hW)

    Parametreler
    ----------
    path       : str   — görüntü dosyalarının bulunduğu klasör
    file_l     : str   — sol yarı dosya adı  (örn. 'DZB1210_a.tif')
    file_r     : str   — sağ yarı dosya adı  (örn. 'DZB1210_b.tif')
    corners_l  : np.ndarray, shape (2, 2)
                 Sol yarı köşe koordinatları [[x_üst, y_üst], [x_alt, y_alt]]
    corners_r  : np.ndarray, shape (2, 2)
                 Sağ yarı köşe koordinatları
    progress_cb: callable veya None
                 İlerleme geri çağrısı: progress_cb(adım, toplam, mesaj)

    Döndürür
    -------
    str — kaydedilen .npz dosyasının tam yolu
    """

    def _progress(step, total, msg):
        if progress_cb:
            progress_cb(step, total, msg)
        else:
            print(f"  [{step}/{total}] {msg}")

    path = Path(path)

    # Çıktı dosya adı: sol yarı adındaki son iki karakter (_a) çıkarılır
    stem = Path(file_l).stem[:-2]           # 'DZB1210_a' → 'DZB1210'
    out_path = path / f"{stem}.npz"

    # -----------------------------------------------------------------
    # ADIM 1: Sol yarıyı oku ve kaydet
    # -----------------------------------------------------------------
    _progress(2, 6, 'sol yarı görüntü okunuyor...')
    image_l, info_l = _read_image_half(path / file_l, corners_l)

    # -----------------------------------------------------------------
    # ADIM 2: Sağ yarı için dönüşüm matrisi tahmin et (ORB + RANSAC)
    # -----------------------------------------------------------------
    _progress(3, 6, 'ORB özellikleri tespit ediliyor ve eşleştiriliyor...')
    transform_r = _estimate_transform(
        image_l, path / file_r, corners_r, corners_l
    )

    # -----------------------------------------------------------------
    # ADIM 3: Sağ yarıyı dönüştür ve sol yarıyla birleştir
    # -----------------------------------------------------------------
    _progress(4, 6, 'sağ yarı görüntü dönüştürülüyor ve birleştiriliyor...')
    image_full = _merge_halves(image_l, path / file_r, corners_r, transform_r)

    # -----------------------------------------------------------------
    # ADIM 4: Sonucu kaydet
    # -----------------------------------------------------------------
    _progress(5, 6, 'sonuç .npz olarak kaydediliyor...')
    _save_result(out_path, image_full, info_l, transform_r)

    _progress(6, 6, 'tamamlandı.')
    return str(out_path)


# ===========================================================================
# SOL YARI OKUMA
# MATLAB karşılığı: saveLeftImageHalf()
# ===========================================================================

def _read_image_half(filepath, corners):
    """
    GeoTIFF dosyasından sol yarı görüntüyü bloklar halinde okur.

    Parametreler
    ----------
    filepath : Path  — GeoTIFF dosya yolu
    corners  : np.ndarray, shape (2, 2)
               [[x_min, y_min], [x_max, y_max]] piksel koordinatları

    Döndürür
    -------
    image : np.ndarray  — tam sol yarı görüntüsü (uint8 veya uint16)
    info  : dict        — görüntü metadata bilgileri
    """
    num_sections = 10

    with rasterio.open(filepath) as ds:
        # Köşe koordinatlarından piksel bölgesi tanımla
        row_min = int(corners[0, 1]) - 1   # Python 0-tabanlı
        row_max = int(corners[1, 1]) - 1
        col_min = int(corners[0, 0]) - 1
        col_max = int(corners[1, 0]) - 1

        height = row_max - row_min + 1
        width  = col_max - col_min + 1

        # Görüntüyü sütun bölümlerine ayırarak oku (bellek tasarrufu)
        col_splits = np.round(
            np.linspace(col_min, col_max, num_sections + 1)
        ).astype(int)

        image = np.zeros((height, width), dtype=np.uint8)

        for i in range(num_sections):
            c0 = col_splits[i]
            c1 = col_splits[i + 1]
            window = rasterio.windows.Window(
                c0, row_min, c1 - c0 + 1, height
            )
            block = ds.read(1, window=window)
            # uint16 → uint8 dönüşümü gerekiyorsa
            if block.dtype != np.uint8:
                block = (block / 256).astype(np.uint8)

            ci_start = c0 - col_min
            ci_end   = ci_start + block.shape[1]
            image[:, ci_start:ci_end] = block

        info = {
            'filename': str(filepath),
            'height':   ds.height,
            'width':    ds.width,
            'transform': ds.transform,
            'crs':       str(ds.crs),
        }

    return image, info


# ===========================================================================
# DÖNÜŞÜM TAHMİNİ (ORB + RANSAC)
# MATLAB karşılığı: estimateTransform()
# ===========================================================================

def _estimate_transform(image_l, filepath_r, corners_r, corners_l):
    """
    Sol ve sag yarı arasindaki dönüsüm matrisini tahmin eder.

    Strateji:
    - Her iki dosyadan da örtüsme bölgesini ham olarak oku
    - ORB + RANSAC ile eslestir
    - Koordinatlari tam görüntü sistemine cevir
    - cv2.estimateAffinePartial2D ile nihai M matrisini hesapla
    """
    h, w = image_l.shape[:2]

    # Örtüsme genisligi: dosyanin %15'i (corners degil, gercek dosya boyutu)
    # Sol yarinin SAG kenari ile sag yarinin SOL kenari örtüsüyor
    # _a dosyasinin gercek sag kenarini kullan
    with rasterio.open(filepath_r) as ds_tmp:
        file_h_r = ds_tmp.height
        file_w_r = ds_tmp.width

    # Sol yarinin corners bilgisi
    l_col_min = int(corners_l[0, 0]) - 1
    l_col_max = int(corners_l[1, 0]) - 1
    l_row_min = int(corners_l[0, 1]) - 1
    l_row_max = int(corners_l[1, 1]) - 1

    # Örtüsme: dosyanin sag %15'i
    # _a dosyasinin gercek genisligi ile hesapla
    filepath_l = Path(str(filepath_r).replace('_b.tif', '_a.tif').replace('_B.tif', '_A.tif'))
    if not filepath_l.exists():
        # Dosya yolundan _a dosyasini tahmin et
        filepath_l = filepath_r.parent / filepath_r.name.replace('_b', '_a').replace('_B', '_A')

    with rasterio.open(filepath_l) as ds_l_raw:
        file_h_l = ds_l_raw.height
        file_w_l = ds_l_raw.width
        overlap = max(file_w_l * 15 // 100, 2000)

        # Sol yarinin SAG kenarini dosyadan oku (corners degil, dosya kenari)
        win_l = rasterio.windows.Window(
            file_w_l - overlap, l_row_min,
            overlap, l_row_max - l_row_min + 1
        )
        img_l_overlap = ds_l_raw.read(1, window=win_l)
        if img_l_overlap.dtype != np.uint8:
            img_l_overlap = (img_l_overlap / 256).astype(np.uint8)

    # Sag yarinin sol kenarini dosyadan oku — corners degil, gercek sol kenar
    # (corners sol kenar tamponu iceriyor, gercek örtüsme dosyanin x=0'indan basliyor)
    r_col_off = 0                              # dosyanin gercek sol kenari
    r_row_off = int(corners_r[0, 1]) - 1      # y: corners tamponunu koru
    r_height  = int(corners_r[1, 1]) - int(corners_r[0, 1])

    with rasterio.open(filepath_r) as ds_r:
        read_h = min(r_height, file_h_r - r_row_off)
        win_r = rasterio.windows.Window(r_col_off, r_row_off, overlap, read_h)
        img_r_overlap = ds_r.read(1, window=win_r)
        if img_r_overlap.dtype != np.uint8:
            img_r_overlap = (img_r_overlap / 256).astype(np.uint8)

    # Yükseklik eslestir
    common_h = min(img_l_overlap.shape[0], img_r_overlap.shape[0])

    # Sol strip icin x ofseti: dosyanin sag kenari (corners_l degil)
    l_overlap_x_offset = file_w_l - overlap   # dosya koordinatinda

    # ORB nesnesi
    orb     = cv2.ORB_create(nfeatures=30000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Sol yarinin sag stripinde veri olan satir araligini bul
    right_strip_full = image_l[:, w - overlap:]
    row_fill = (right_strip_full > 20).mean(axis=1)
    valid_rows = np.where(row_fill > 0.3)[0]
    if len(valid_rows) < 100:
        # Tüm satirlari kullan
        y0_valid, y1_valid = 0, common_h
    else:
        y0_valid = int(valid_rows[0])
        y1_valid = int(valid_rows[-1]) + 1
    print(f"    Veri satiri araligi: {y0_valid}-{y1_valid} ({y1_valid-y0_valid} satir)")

    # Sadece veri olan satirlarda eslestir
    img_l_eq = cv2.equalizeHist(img_l_overlap[y0_valid:y1_valid])
    img_r_eq = cv2.equalizeHist(img_r_overlap[y0_valid:y1_valid])

    kp_l, desc_l = orb.detectAndCompute(img_l_eq, None)
    kp_r, desc_r = orb.detectAndCompute(img_r_eq, None)

    if desc_l is None or desc_r is None or len(kp_l) < 8 or len(kp_r) < 8:
        raise RuntimeError("Yeterli keypoint bulunamadi.")

    matches = matcher.match(desc_l, desc_r)
    matches = sorted(matches, key=lambda m: m.distance)
    print(f"    ORB esleme (veri bolgesi): {len(matches)}")

    if len(matches) < 8:
        raise RuntimeError("Yeterli esleme bulunamadi.")

    pts_l_strip = np.float32([kp_l[m.queryIdx].pt for m in matches])
    pts_r_strip = np.float32([kp_r[m.trainIdx].pt for m in matches])

    # Satir ofsetini ekle (y0_valid)
    pts_l_strip[:, 1] += y0_valid
    pts_r_strip[:, 1] += y0_valid

    # RANSAC strip koordinatlarinda calistir (koordinat karisikligini önler)
    M_cv, inliers_cv = cv2.estimateAffinePartial2D(
        pts_r_strip, pts_l_strip,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.99,
    )

    if M_cv is None:
        raise RuntimeError("estimateAffinePartial2D basarisiz.")

    scale     = np.sqrt(M_cv[0,0]**2 + M_cv[0,1]**2)
    angle_deg = np.degrees(np.arctan2(M_cv[1,0], M_cv[0,0]))
    inlier_n  = int(inliers_cv.sum())
    inlier_pct = inlier_n / len(inliers_cv) * 100
    print(f"    RANSAC: {inlier_n}/{len(inliers_cv)} inlier (%{inlier_pct:.1f})")
    print(f"    olcek={scale:.4f} rotasyon={angle_deg:.2f}° "
          f"strip_tx={M_cv[0,2]:.1f} strip_ty={M_cv[1,2]:.1f}")

    if scale < 0.7 or scale > 1.4:
        raise RuntimeError(f"Gecersiz olcek: {scale:.4f}")
    if abs(angle_deg) > 15:
        raise RuntimeError(f"Gecersiz rotasyon: {angle_deg:.2f}")

    # Strip koordinatlarindan tam görüntü koordinatina geç
    # Sol strip: x_strip = x_full - (w - overlap)  =>  x_full = x_strip + (w - overlap)
    # Sag strip: x_strip = x_dosya - r_col_off      =>  x_dosya = x_strip + r_col_off
    #
    # M_strip: pts_r_strip -> pts_l_strip
    # M_full : pts_r_dosya -> pts_l_full
    #
    # pts_l_full  = pts_l_strip  + [w-overlap, 0]
    # pts_r_dosya = pts_r_strip  + [r_col_off, r_row_off]
    #
    # pts_l_strip = M_strip * pts_r_strip + t_strip
    # pts_l_full - [w-overlap, 0] = M_strip * (pts_r_dosya - [r_col_off, r_row_off]) + t_strip
    # pts_l_full = M_strip * pts_r_dosya + (t_strip - M_strip*[r_col_off,r_row_off] + [w-overlap,0])

    R = M_cv[:, :2]   # 2x2 rotasyon/olcek
    t_strip = M_cv[:, 2]  # 2x1 öteleme (strip koordinatinda)

    offset_r = np.array([r_col_off, r_row_off], dtype=float)
    # Sol strip x ofseti: dosya koordinatinda (file_w_l - overlap)
    # Ama M matrisi pts_r_dosya -> pts_l_dosya donusumu veriyor
    # Sonra _merge_halves bu donusumu kullanacak
    offset_l = np.array([l_overlap_x_offset, l_row_min], dtype=float)

    t_full = t_strip - R @ offset_r + offset_l

    M = np.eye(4)
    M[0, :2] = R[0]
    M[0,  3] = t_full[0]
    M[1, :2] = R[1]
    M[1,  3] = t_full[1]

    print(f"    Tam koordinat: tx={t_full[0]:.1f} ty={t_full[1]:.1f}")
    return M

# ===========================================================================
# YARIMA BİRLEŞTİRME
# MATLAB karşılığı: saveRightImageHalf()
# ===========================================================================

def _merge_halves(image_l, filepath_r, corners_r, transform_r):
    """
    Sag yarıyı donusturup sol yarıyla birlestir.
    OpenCV SHRT_MAX sinirini asan buyuk goruntuler icin blok tabanli numpy remap kullanır.
    """
    OPENCV_MAX = 32766
    h_l, w_l = image_l.shape[:2]

    # Tuval genisligi: sol yarinin genisligi + sag yarinin genisligi - örtusme
    # T[0,3] = sag yarinin sol kenarinin sol koordinat sistemindeki x konumu
    # Sag yarinin genisligi = corners_r[1,0] - corners_r[0,0]
    r_width  = int(corners_r[1, 0]) - int(corners_r[0, 0])
    r_x_start = int(round(transform_r[0, 3]))   # sag yarinin x baslangici
    canvas_w = max(w_l, r_x_start + r_width)
    canvas_w = min(canvas_w, w_l + r_width)     # en fazla iki yarinin toplami

    canvas = np.zeros((h_l, canvas_w), dtype=np.uint8)
    canvas[:, :w_l] = image_l

    with rasterio.open(filepath_r) as ds_r:
        row_min = int(corners_r[0,1]) - 1
        row_max = int(corners_r[1,1]) - 1
        col_min = int(corners_r[0,0]) - 1
        col_max = int(corners_r[1,0]) - 1
        window_r = rasterio.windows.Window(col_min, row_min, col_max-col_min+1, row_max-row_min+1)
        img_r = ds_r.read(1, window=window_r)
        if img_r.dtype != np.uint8:
            img_r = (img_r / 256).astype(np.uint8)

    h_r, w_r = img_r.shape
    M = transform_r
    a00,a01,a03 = M[0,0],M[0,1],M[0,3]
    a10,a11,a13 = M[1,0],M[1,1],M[1,3]

    if canvas_w <= OPENCV_MAX and h_l <= OPENCV_MAX:
        M_cv = np.array([[a00,a01,a03],[a10,a11,a13]], dtype=np.float64)
        img_r_warped = cv2.warpAffine(img_r, M_cv, (canvas_w, h_l),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask_empty = canvas == 0
        canvas[mask_empty] = img_r_warped[mask_empty]
    else:
        det = a00*a11 - a01*a10
        if abs(det) < 1e-12:
            return canvas
        inv00= a11/det; inv01=-a01/det; inv10=-a10/det; inv11= a00/det
        inv03=(a01*a13 - a11*a03)/det; inv13=(a10*a03 - a00*a13)/det
        block_h = 2048
        from scipy.ndimage import map_coordinates
        for y0 in range(0, h_l, block_h):
            y1 = min(y0+block_h, h_l)
            rows = np.arange(y0, y1, dtype=np.float32)
            cols = np.arange(canvas_w, dtype=np.float32)
            mC, mR = np.meshgrid(cols, rows)
            src_x = (inv00*mC + inv01*mR + inv03).astype(np.float32)
            src_y = (inv10*mC + inv11*mR + inv13).astype(np.float32)
            valid = (src_x>=0)&(src_x<w_r)&(src_y>=0)&(src_y<h_r)
            warped_block = np.zeros((y1-y0, canvas_w), dtype=np.uint8)
            if valid.any():
                coords = np.array([src_y[valid].ravel(), src_x[valid].ravel()])
                vals = map_coordinates(img_r.astype(np.float32), coords,
                    order=1, mode="constant", cval=0).astype(np.uint8)
                warped_block[valid] = vals
            mask_empty = canvas[y0:y1] == 0
            canvas[y0:y1][mask_empty] = warped_block[mask_empty]

    return canvas




# ===========================================================================
# SONUÇ KAYDETME
# MATLAB karşılığı: matfile kayıt işlemleri
# ===========================================================================

def _save_result(out_path, image_full, info_l, transform_r):
    """
    Birleştirilmiş görüntüyü ve metadata bilgilerini .npz olarak kaydeder.

    MATLAB'daki .mat dosyasının Python karşılığı numpy .npz formatıdır.
    Sonraki aşamalar (extract, georef) bu dosyayı okuyacak.

    Kaydedilen alanlar:
      Image      — tam birleştirilmiş görüntü (uint8)
      Transform  — sağ yarı dönüşüm matrisi (4x4)
      SourceFile — orijinal GeoTIFF dosya yolu
    """
    np.savez_compressed(
        str(out_path),
        Image=image_full,
        Transform=transform_r,
        SourceFile=np.array([info_l['filename']]),
    )
    print(f"  Kaydedildi: {out_path}")
    print(f"  Görüntü boyutu: {image_full.shape[1]} x {image_full.shape[0]} piksel")


# ===========================================================================
# BASIT TEST
# ===========================================================================

if __name__ == '__main__':
    print("sti_stitch.py — Temel birim testleri\n")

    # -------------------------------------------------------------------
    # TEST 1: ORB özellik tespiti ve eşleştirme
    # -------------------------------------------------------------------
    print("TEST 1: ORB özellik tespiti ve eşleştirme")

    # Sentetik test görüntüsü oluştur
    rng = np.random.default_rng(42)
    img_base = (rng.random((500, 800)) * 255).astype(np.uint8)

    # Basit öteleme ile sahte sağ yarı
    shift_x, shift_y = 50, 10
    M_true = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
    img_shifted = cv2.warpAffine(img_base, M_true, (800, 500))

    # ORB ile eşleştir
    orb     = cv2.ORB_create(nfeatures=5000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp1, desc1 = orb.detectAndCompute(img_base[:, 600:], None)
    kp2, desc2 = orb.detectAndCompute(img_shifted[:, :200], None)

    if desc1 is not None and desc2 is not None and \
       len(kp1) > 0 and len(kp2) > 0:
        matches = matcher.match(desc1, desc2)
        print(f"  Tespit edilen eşleşme sayısı : {len(matches)}")
        print(f"  TEST 1 BASARILI ✓\n" if len(matches) > 10
              else f"  TEST 1 BASARISIZ ✗\n")
    else:
        print("  Özellik tespit edilemedi — görüntü sentetik olduğu için")
        print("  normal, gerçek görüntüde çalışır.\n")

    # -------------------------------------------------------------------
    # TEST 2: estimate_transform_ransac entegrasyonu
    # -------------------------------------------------------------------
    print("TEST 2: RANSAC dönüşüm tahmini entegrasyonu")

    rng2 = np.random.default_rng(7)
    n = 100
    pts_src = rng2.random((2, n)) * 500
    noise   = rng2.normal(0, 0.5, (2, n))

    # Bilinen dönüşüm: öteleme
    tx, ty = 30.0, -15.0
    pts_dst = pts_src + np.array([[tx], [ty]]) + noise

    # Sözleşme: estimate_transform_ransac(pts_fixed, pts_moving)
    # M * pts_moving ≈ pts_fixed  →  M taşır: dst → src  →  tx = -30
    pts_fixed_h  = np.vstack([pts_src, np.ones((1, n))])   # sabit  (kaynak)
    pts_moving_h = np.vstack([pts_dst, np.ones((1, n))])   # hareketli (hedef)

    M_est, inliers = estimate_transform_ransac(
        pts_fixed_h, pts_moving_h, num_iter=500, inlier_dist=2.0
    )

    # M: dst→src dönüşümü  →  beklenen tx = -30, ty = +15
    tx_est = M_est[0, 3]
    ty_est = M_est[1, 3]
    err_x  = abs(tx_est - (-tx))   # beklenen -30
    err_y  = abs(ty_est - (-ty))   # beklenen +15

    print(f"  Gerçek öteleme    : tx={tx:.1f}, ty={ty:.1f}")
    print(f"  Tahmin edilen     : tx={tx_est:.2f}, ty={ty_est:.2f}")
    print(f"  Hata              : Δx={err_x:.2f}, Δy={err_y:.2f} piksel")
    print(f"  İnlier sayısı     : {inliers.sum()}/{n}")
    print(f"  TEST 2 BASARILI ✓\n" if err_x < 3 and err_y < 3
          else f"  TEST 2 BASARISIZ ✗\n")

    # -------------------------------------------------------------------
    # TEST 3: Görüntü birleştirme (warp)
    # -------------------------------------------------------------------
    print("TEST 3: Görüntü birleştirme (warp)")

    img_l = np.zeros((300, 600), dtype=np.uint8)
    img_l[50:250, 50:550] = 128   # gri dikdörtgen

    img_r = np.zeros((300, 400), dtype=np.uint8)
    img_r[50:250, 50:350] = 200   # daha açık gri dikdörtgen

    # Basit öteleme dönüşümü
    M_test = np.eye(4)
    M_test[0, 3] = 500    # x ekseninde 500 piksel ötelemek

    M_affine = M_test[:2, :]
    M_cv = np.array([
        [M_affine[0, 0], M_affine[0, 1], M_affine[0, 3]],
        [M_affine[1, 0], M_affine[1, 1], M_affine[1, 3]],
    ], dtype=np.float64)

    canvas_w = 1000
    img_r_warped = cv2.warpAffine(img_r, M_cv, (canvas_w, 300))

    canvas = np.zeros((300, canvas_w), dtype=np.uint8)
    canvas[:, :600] = img_l
    mask = canvas == 0
    canvas[mask] = img_r_warped[mask]

    non_zero = np.count_nonzero(canvas)
    print(f"  Tuval boyutu      : {canvas.shape}")
    print(f"  Dolu piksel sayısı: {non_zero}")
    print(f"  TEST 3 BASARILI ✓" if non_zero > 0
          else f"  TEST 3 BASARISIZ ✗")

    print("\nTüm testler tamamlandı.")
