"""
ext_stereo_rect.py

MATLAB kaynak dosyası:
  - main/2_extract/extStereoRect.m

Açıklama:
  Stereo görüntü çiftini epipolar doğrultma (rectification) için
  homojenlik dönüşümleri hesaplar ve uygular. ORB özellik eşleştirmesi,
  temel matris tahmini ve cv2.stereoRectifyUncalibrated kullanır.

Bağımlılıklar:
  - numpy         (pip install numpy)
  - opencv-python (pip install opencv-python)
  - scipy         (pip install scipy)
  - scikit-image  (pip install scikit-image)

Dahili bağımlılıklar:
  - ext_filter_images.py (aynı klasörde)
"""

import numpy as np
import cv2
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ext_filter_images import ext_filter_images


# ===========================================================================
# ANA FONKSİYON
# MATLAB karşılığı: extStereoRect()
# ===========================================================================

def ext_stereo_rect(obj_l, obj_r, save_path=None, progress_cb=None):
    """
    Stereo görüntü çiftini epipolar doğrultur.

    MATLAB imzası:
        extStereoRect(objL, objR, strSavePath, hW, cWin)

    Parametreler
    ----------
    obj_l, obj_r : dict
        Stereo görüntü nesneleri. Beklenen anahtarlar:
            'Image'     : np.ndarray — ham görüntü
            'Window'    : np.ndarray, shape (2,2) — piksel penceresi
        Fonksiyon çıktısı olarak şunlar eklenir:
            'RectImage'    : np.ndarray — doğrultulmuş görüntü
            'RectWindow'   : np.ndarray — doğrultulmuş pencere sınırları
            'Homography'   : np.ndarray, shape (3,3)
            'PointMatches' : np.ndarray — eşleşen nokta çiftleri
            'Accuracy'     : dict — dikey hata istatistikleri

    save_path    : str veya None  — ara çıktı figürlerinin kaydedileceği yol
    progress_cb  : callable veya None

    Döndürür
    -------
    obj_l, obj_r : dict  — güncellenerek döndürülür
    """

    def _progress(msg):
        if progress_cb:
            progress_cb(msg)

    _progress('epipolar görüntüler hesaplanıyor...')

    # -----------------------------------------------------------------
    # Görüntüleri filtrele
    # -----------------------------------------------------------------
    opt = {'histmatch': True, 'adapthisteq': True, 'wiener2': True}
    img_l_f, img_r_f = ext_filter_images(
        obj_l['Image'], obj_r['Image'], empty_val=0, opt=opt
    )
    # float [0,1] → uint8
    obj_l['ImageFilt'] = (img_l_f * 255).astype(np.uint8)
    obj_r['ImageFilt'] = (img_r_f * 255).astype(np.uint8)

    # -----------------------------------------------------------------
    # Parametreler
    # -----------------------------------------------------------------
    max_features = 10000
    num_keep     = 100
    max_error_1  = 6.0    # ilk tur
    max_error_2  = 3.0    # ikinci tur
    blk_sz       = round(np.mean(np.array(obj_l['Image'].shape[:2]) / 4))

    # -----------------------------------------------------------------
    # TUR 1: Blok tabanlı ilk eşleştirme → H1, H2 tahmin et
    # -----------------------------------------------------------------
    pts_l, pts_r = _block_match(obj_l, obj_r, blk_sz, max_features, num_keep)
    H1, H2, pts_l, pts_r = _compute_homographies(
        obj_l['Image'].shape[:2], pts_l, pts_r, max_error_1
    )

    # -----------------------------------------------------------------
    # TUR 2: H1/H2 kullanarak geliştirilmiş eşleştirme
    # -----------------------------------------------------------------
    pts_l2, pts_r2 = _block_match_refined(
        obj_l, obj_r, H1, H2, blk_sz, max_features, num_keep
    )
    if pts_l2 is not None and len(pts_l2[0]) > 8:
        pts_l_all = np.hstack([pts_l, pts_l2])
        pts_r_all = np.hstack([pts_r, pts_r2])
    else:
        pts_l_all, pts_r_all = pts_l, pts_r

    H1, H2, pts_l_all, pts_r_all = _compute_homographies(
        obj_l['Image'].shape[:2], pts_l_all, pts_r_all, max_error_2
    )

    # -----------------------------------------------------------------
    # Görüntüleri dönüştür (remap)
    # -----------------------------------------------------------------
    obj_l, obj_r = _transform_images(obj_l, obj_r, H1, H2)

    # -----------------------------------------------------------------
    # Nokta eşleşmelerini ve doğruluğu kaydet
    # -----------------------------------------------------------------
    v_error = _vertical_error(H1, H2, pts_l_all, pts_r_all)

    obj_l['PointMatches'] = pts_l_all
    obj_r['PointMatches'] = pts_r_all
    obj_l['Homography']   = H1
    obj_r['Homography']   = H2
    obj_l['RectWindow']   = obj_l.get('RectWindow',
                                       np.array([[0, 0],
                                                 [obj_l['RectImage'].shape[1]-1,
                                                  obj_l['RectImage'].shape[0]-1]]))
    obj_r['RectWindow']   = obj_l['RectWindow'].copy()

    accuracy = obj_l.get('Accuracy', {})
    accuracy['PointMatches'] = v_error
    obj_l['Accuracy'] = accuracy

    return obj_l, obj_r


# ===========================================================================
# BLOK TABANLI EŞLEŞTİRME
# MATLAB karşılığı: block1() iç fonksiyonu
# ===========================================================================

def _block_match(obj_l, obj_r, blk_sz, max_features, num_keep):
    """
    Görüntüyü bloklara bölerek ORB eşleştirmesi yapar.
    MATLAB: blockProcess + block1()
    """
    img_l = obj_l['ImageFilt']
    img_r = obj_r['ImageFilt']
    h, w  = img_l.shape[:2]

    orb     = cv2.ORB_create(nfeatures=max_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    pts_l_all = []
    pts_r_all = []

    # Blok sınırları
    x_splits = _block_indices(w, blk_sz)
    y_splits = _block_indices(h, blk_sz)

    buffer = 200

    for iy in range(len(y_splits) - 1):
        for ix in range(len(x_splits) - 1):

            # Sol blok
            x0_l = x_splits[ix];    x1_l = x_splits[ix + 1]
            y0_l = y_splits[iy];    y1_l = y_splits[iy + 1]
            blk_l = img_l[y0_l:y1_l, x0_l:x1_l]

            # Sağ blok (buffer ekle)
            x0_r = max(0, x0_l - buffer);  x1_r = min(w, x1_l + buffer)
            y0_r = max(0, y0_l - buffer);  y1_r = min(h, y1_l + buffer)
            blk_r = img_r[y0_r:y1_r, x0_r:x1_r]

            if blk_l.size == 0 or blk_r.size == 0:
                continue

            kp_l, desc_l = orb.detectAndCompute(blk_l, None)
            kp_r, desc_r = orb.detectAndCompute(blk_r, None)

            if desc_l is None or desc_r is None:
                continue
            if len(kp_l) < 4 or len(kp_r) < 4:
                continue

            matches = matcher.match(desc_l, desc_r)
            if len(matches) == 0:
                continue

            # En iyi num_keep eşleşmeyi al
            matches = sorted(matches, key=lambda m: m.distance)[:num_keep]

            pl = np.array([kp_l[m.queryIdx].pt for m in matches])
            pr = np.array([kp_r[m.trainIdx].pt for m in matches])

            # Tam görüntü koordinatlarına çevir
            pl[:, 0] += x0_l;  pl[:, 1] += y0_l
            pr[:, 0] += x0_r;  pr[:, 1] += y0_r

            pts_l_all.append(pl)
            pts_r_all.append(pr)

    if not pts_l_all:
        return np.zeros((2, 0)), np.zeros((2, 0))

    pts_l = np.vstack(pts_l_all).T   # (2, N)
    pts_r = np.vstack(pts_r_all).T
    return pts_l, pts_r


# ===========================================================================
# İYİLEŞTİRİLMİŞ BLOK EŞLEŞTİRME
# MATLAB karşılığı: block2() iç fonksiyonu
# ===========================================================================

def _block_match_refined(obj_l, obj_r, H1, H2, blk_sz, max_features, num_keep):
    """
    H1/H2 homografilerini kullanarak daha hassas eşleştirme yapar.
    MATLAB: blockProcess + block2()
    """
    img_l = obj_l['ImageFilt']
    img_r = obj_r['ImageFilt']
    h, w  = img_l.shape[:2]

    orb     = cv2.ORB_create(nfeatures=max_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    pts_l_all = []
    pts_r_all = []

    x_splits = _block_indices(w, blk_sz)
    y_splits = _block_indices(h, blk_sz)
    search   = 200

    H2_inv_H1 = np.linalg.solve(H2, H1)

    for iy in range(len(y_splits) - 1):
        for ix in range(len(x_splits) - 1):

            # 25 rastgele nokta seç
            rng = np.random.default_rng()
            pts_lb = np.array([
                rng.integers(x_splits[ix], x_splits[ix+1], 25),
                rng.integers(y_splits[iy], y_splits[iy+1], 25),
            ], dtype=float)

            # Sol noktaları sağ koordinatlara dönüştür
            pts_lb_h = np.vstack([pts_lb, np.ones((1, 25))])
            pts_rb_h = H2_inv_H1 @ pts_lb_h
            pts_rb   = (pts_rb_h[:2] / pts_rb_h[2:3]).round().astype(int)

            for j in range(25):
                xl, yl = int(pts_lb[0, j]), int(pts_lb[1, j])
                xr, yr = int(pts_rb[0, j]), int(pts_rb[1, j])

                # Alt blok sınırları
                x0_l = max(0, xl - search);  x1_l = min(w, xl + search)
                y0_l = max(0, yl - search);  y1_l = min(h, yl + search)
                x0_r = max(0, xr - search);  x1_r = min(w, xr + search)
                y0_r = max(0, yr - search);  y1_r = min(h, yr + search)

                if (x1_l - x0_l < 4 or y1_l - y0_l < 4 or
                        x1_r - x0_r < 4 or y1_r - y0_r < 4):
                    continue

                blk_l = img_l[y0_l:y1_l, x0_l:x1_l]
                blk_r = img_r[y0_r:y1_r, x0_r:x1_r]

                kp_l, desc_l = orb.detectAndCompute(blk_l, None)
                kp_r, desc_r = orb.detectAndCompute(blk_r, None)

                if desc_l is None or desc_r is None:
                    continue
                if len(kp_l) < 2 or len(kp_r) < 2:
                    continue

                matches = matcher.match(desc_l, desc_r)
                if not matches:
                    continue

                matches = sorted(matches, key=lambda m: m.distance)[:10]

                pl = np.array([kp_l[m.queryIdx].pt for m in matches])
                pr = np.array([kp_r[m.trainIdx].pt for m in matches])

                pl[:, 0] += x0_l;  pl[:, 1] += y0_l
                pr[:, 0] += x0_r;  pr[:, 1] += y0_r

                pts_l_all.append(pl)
                pts_r_all.append(pr)

    if not pts_l_all:
        return None, None

    return np.vstack(pts_l_all).T, np.vstack(pts_r_all).T


# ===========================================================================
# HOMOGRAFİ HESAPLAMA
# MATLAB karşılığı: homographies() iç fonksiyonu
# ===========================================================================

def _compute_homographies(img_shape, pts_l, pts_r, max_error,
                          num_attempts=100):
    """
    RANSAC ile temel matris ve stereoRectifyUncalibrated hesaplar.
    En iyi homojen dönüşüm çiftini seçer.

    MATLAB karşılığı: homographies() iç fonksiyonu
    """
    h, w = img_shape
    n    = pts_l.shape[1]

    if n < 8:
        # Yeterli nokta yoksa birim matris döndür
        return np.eye(3), np.eye(3), pts_l, pts_r

    best_count = -1
    best_H1 = np.eye(3)
    best_H2 = np.eye(3)
    best_pts_l = pts_l
    best_pts_r = pts_r

    rng = np.random.default_rng()

    for _ in range(num_attempts):
        # %95 rastgele alt küme
        idx = rng.choice(n, size=max(8, round(n * 0.95)), replace=False)
        sub_l = pts_l[:, idx].T.astype(np.float32)
        sub_r = pts_r[:, idx].T.astype(np.float32)

        # RANSAC ile aykırı değerleri temizle
        F_r, mask = cv2.findFundamentalMat(
            sub_l, sub_r,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=max_error,
            confidence=0.99,
        )
        if F_r is None or mask is None:
            continue

        inliers_l = sub_l[mask.ravel() == 1]
        inliers_r = sub_r[mask.ravel() == 1]

        if len(inliers_l) < 8:
            continue

        # 8-nokta algoritması ile temel matris hesapla
        F, _ = cv2.findFundamentalMat(
            inliers_l, inliers_r, method=cv2.FM_8POINT
        )
        if F is None:
            continue

        # Doğrultma homojenlik matrisleri
        ret, H1_c, H2_c = cv2.stereoRectifyUncalibrated(
            inliers_l, inliers_r, F, (w, h),
            threshold=max_error
        )
        if not ret:
            continue

        # Dikey hata kontrolü
        v_err = _vertical_error(H1_c, H2_c,
                                inliers_l.T, inliers_r.T)
        count = int(np.sum(np.abs(v_err) < max_error))

        if count > best_count:
            best_count  = count
            best_H1     = H1_c
            best_H2     = H2_c
            best_pts_l  = inliers_l.T
            best_pts_r  = inliers_r.T

    return best_H1, best_H2, best_pts_l, best_pts_r


# ===========================================================================
# GÖRÜNTÜ DÖNÜŞTÜRME (REMAP)
# MATLAB karşılığı: transform() iç fonksiyonu
# ===========================================================================

def _transform_images(obj_l, obj_r, H1, H2):
    """
    H1 ve H2 homojenlik matrisleri ile görüntüleri cv2.remap ile dönüştürür.
    MATLAB karşılığı: transform() + cv.remap()
    """
    h, w = obj_l['Image'].shape[:2]

    # Her iki görüntünün köşelerini dönüştür
    corners = np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]], dtype=float).T

    proj1 = H1 @ corners
    proj1 = (proj1[:2] / proj1[2:3]).T

    proj2 = H2 @ corners
    proj2 = (proj2[:2] / proj2[2:3]).T

    all_proj = np.vstack([proj1, proj2])
    win_min  = np.floor(all_proj.min(axis=0)).astype(int)
    win_max  = np.ceil(all_proj.max(axis=0)).astype(int)
    out_size = (win_max - win_min + 1)   # (w, h)

    # Boyut sınırı kontrolü
    if np.any(out_size > np.array([w, h]) * 1.5):
        warnings.warn(
            f"Doğrultulmuş görüntü çok büyük ({out_size}), "
            "kötü stereo doğrultma sonucu olabilir."
        )
        out_size = np.array([w, h])
        win_min  = np.array([0, 0])

    out_w, out_h = int(out_size[0]), int(out_size[1])

    # Koordinat ızgarası → ters dönüşüm → remap
    for obj, H in [(obj_l, H1), (obj_r, H2)]:
        mX, mY = np.meshgrid(
            np.arange(win_min[0], win_min[0] + out_w),
            np.arange(win_min[1], win_min[1] + out_h),
        )
        pts_h  = np.vstack([mX.ravel(), mY.ravel(),
                             np.ones(out_w * out_h)])
        pts_src = np.linalg.solve(H, pts_h)
        pts_src = pts_src[:2] / pts_src[2:3]

        map_x = pts_src[0].reshape(out_h, out_w).astype(np.float32)
        map_y = pts_src[1].reshape(out_h, out_w).astype(np.float32)

        obj['RectImage'] = cv2.remap(
            obj['Image'], map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    # Doğrultulmuş pencere
    rect_win = np.array([
        [win_min[0], win_min[1]],
        [win_min[0] + out_w - 1, win_min[1] + out_h - 1],
    ])
    obj_l['RectWindow'] = rect_win
    obj_r['RectWindow'] = rect_win.copy()

    return obj_l, obj_r


# ===========================================================================
# DİKEY HATA
# MATLAB karşılığı: error() iç fonksiyonu
# ===========================================================================

def _vertical_error(H1, H2, pts_l, pts_r):
    """
    Homojenlik dönüşümü sonrası dikey (y) koordinat farkını hesaplar.
    MATLAB karşılığı: error() iç fonksiyonu
    """
    n = pts_l.shape[1]
    pts_l_h = np.vstack([pts_l, np.ones((1, n))])
    pts_r_h = np.vstack([pts_r, np.ones((1, n))])

    p1 = H1 @ pts_l_h;  p1 = p1 / p1[2:3]
    p2 = H2 @ pts_r_h;  p2 = p2 / p2[2:3]

    return p1[1] - p2[1]   # dikey fark


# ===========================================================================
# YARDIMCI
# ===========================================================================

def _block_indices(total, blk_sz):
    n_blk = max(1, int(np.ceil(total / blk_sz)))
    return np.round(np.linspace(0, total, n_blk + 1)).astype(int)


# ===========================================================================
# BASIT TEST
# ===========================================================================

if __name__ == '__main__':
    print("ext_stereo_rect.py — Temel testler\n")

    import numpy as np

    rng = np.random.default_rng(42)

    # -------------------------------------------------------------------
    # TEST 1: cv2.findFundamentalMat + stereoRectifyUncalibrated
    # -------------------------------------------------------------------
    print("TEST 1: findFundamentalMat + stereoRectifyUncalibrated")

    # Sentetik nokta çiftleri (küçük dikey fark)
    n = 50
    pts1 = rng.random((n, 2)).astype(np.float32) * 400
    pts2 = pts1 + rng.normal(0, 0.5, (n, 2)).astype(np.float32)
    pts2[:, 0] += 10   # yatay disparite

    F, mask = cv2.findFundamentalMat(
        pts1, pts2, method=cv2.FM_RANSAC,
        ransacReprojThreshold=3.0, confidence=0.99
    )
    inlier_count = int(mask.sum()) if mask is not None else 0
    print(f"  Temel matris inlier sayısı : {inlier_count}/{n}")

    if F is not None and inlier_count >= 8:
        ret, H1, H2 = cv2.stereoRectifyUncalibrated(
            pts1[mask.ravel()==1],
            pts2[mask.ravel()==1],
            F, (400, 400)
        )
        print(f"  stereoRectifyUncalibrated  : {'başarılı' if ret else 'başarısız'}")
        print(f"  TEST 1 BASARILI ✓\n" if ret else f"  TEST 1 BASARISIZ ✗\n")
    else:
        print(f"  TEST 1 BASARISIZ ✗ — yeterli inlier yok\n")

    # -------------------------------------------------------------------
    # TEST 2: cv2.remap
    # -------------------------------------------------------------------
    print("TEST 2: cv2.remap")
    img = (rng.random((200, 300)) * 255).astype(np.uint8)
    H = np.array([[1.0, 0.0, 10.0],
                  [0.0, 1.0,  5.0],
                  [0.0, 0.0,  1.0]])

    mX, mY = np.meshgrid(np.arange(300), np.arange(200))
    pts_h  = np.vstack([mX.ravel(), mY.ravel(), np.ones(300*200)])
    pts_src = np.linalg.solve(H, pts_h)
    pts_src = pts_src[:2] / pts_src[2:3]

    map_x = pts_src[0].reshape(200, 300).astype(np.float32)
    map_y = pts_src[1].reshape(200, 300).astype(np.float32)

    remapped = cv2.remap(img, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
    assert remapped.shape == img.shape
    print(f"  Remap çıktı boyutu : {remapped.shape}")
    print(f"  TEST 2 BASARILI ✓\n")

    # -------------------------------------------------------------------
    # TEST 3: Tam ext_stereo_rect çağrısı (sentetik görüntü)
    # -------------------------------------------------------------------
    print("TEST 3: Tam ext_stereo_rect (sentetik görüntü)")

    # Gerçekçi doku için checker pattern
    checker = np.zeros((300, 400), dtype=np.uint8)
    for r in range(0, 300, 30):
        for c in range(0, 400, 30):
            if (r // 30 + c // 30) % 2 == 0:
                checker[r:r+30, c:c+30] = 200

    M_shift = np.float32([[1, 0, -15], [0, 1, -2]])
    shifted = cv2.warpAffine(checker, M_shift, (400, 300))

    obj_l = {
        'Image':  checker,
        'Window': np.array([[1, 1], [400, 300]]),
    }
    obj_r = {
        'Image':  shifted,
        'Window': np.array([[1, 1], [400, 300]]),
    }

    try:
        out_l, out_r = ext_stereo_rect(obj_l, obj_r)
        assert 'RectImage' in out_l, "'RectImage' eklenmeli"
        assert 'Homography' in out_l, "'Homography' eklenmeli"
        print(f"  RectImage boyutu  : {out_l['RectImage'].shape}")
        print(f"  Homography H1     :\n{np.round(out_l['Homography'], 4)}")
        print(f"  TEST 3 BASARILI ✓")
    except Exception as e:
        print(f"  TEST 3 BASARISIZ ✗ — {e}")
