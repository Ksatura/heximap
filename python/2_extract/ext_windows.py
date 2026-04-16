"""
ext_windows.py

MATLAB kaynak dosyaları:
  - main/2_extract/extGetROI.m
  - main/2_extract/extControlPoints.m  (veri dönüşüm mantığı)
  - main/2_extract/extChooseWindows.m  (pencere hesaplama mantığı)

Açıklama:
  Hexagon görüntü işleme pencerelerinin yönetimi. GUI bağımlı kısımlar
  (kullanıcı diyalogları, interaktif ROI seçimi) PyQt5 arayüzüne
  bırakılmıştır — bu modül yalnızca saf hesaplama mantığını içerir.

Bağımlılıklar:
  - numpy   (pip install numpy)
  - pyproj  (pip install pyproj)
"""

import numpy as np
from pathlib import Path


# ===========================================================================
# CONTROL POINTS — Köşe GCP Dönüşümü
# MATLAB karşılığı: extControlPoints.m > inputCornerGCPs() hesap kısmı
# ===========================================================================

def compute_spatial_transform(corner_gcps_wld, image_shape):
    """
    4 köşe GCP'den görüntü → dünya koordinatı affin dönüşümünü hesaplar.

    MATLAB karşılığı: inputCornerGCPs() içindeki cp2tform + affine2d

    Parametreler
    ----------
    corner_gcps_wld : np.ndarray, shape (4, 2)
        Dünya koordinatları [lon, lat] sırasında, MATLAB sırasında:
        [NE, SE, SW, NW] (EarthExplorer metaverisi sırası)
        Kullanıcı girişi sırası: [NW_lat, NW_lon, NE_lat, NE_lon,
                                   SE_lat, SE_lon, SW_lat, SW_lon]

    image_shape : tuple (height, width)
        Tam Hexagon görüntüsünün piksel boyutu.

    Döndürür
    -------
    T : np.ndarray, shape (3, 3)
        Homojen affin dönüşüm matrisi (görüntü → dünya).
    """
    h, w = image_shape

    # Hexagon tarama yonu: sol_ust->NE, sag_ust->SE, sag_alt->SW, sol_alt->NW
    # (permutasyon analizi ile dogrulandı)
    pts_img = np.array([
        [1,  -1],    # sol_ust -> NE
        [w,  -1],    # sag_ust -> SE
        [w,  -h],    # sag_alt -> SW
        [1,  -h],    # sol_alt -> NW
    ], dtype=float)

    # Dunya koseleri: parse_corner_gcps ciktisi [NE, SE, SW, NW] sirasinda
    pts_wld = np.array(corner_gcps_wld, dtype=float)

    # Affin donusum: [x y 1] @ params = [lon lat]
    A_src = np.hstack([pts_img, np.ones((4, 1))])
    T_params, _, _, _ = np.linalg.lstsq(A_src, pts_wld, rcond=None)

    # 3x3 homojen T matrisi: T @ [x,y,1] = [lon,lat,1]
    T = np.zeros((3, 3)); T[2, 2] = 1.0
    T[0, 0] = T_params[0, 0]; T[0, 1] = T_params[1, 0]; T[0, 2] = T_params[2, 0]
    T[1, 0] = T_params[0, 1]; T[1, 1] = T_params[1, 1]; T[1, 2] = T_params[2, 1]

    return T

    return T


def parse_corner_gcps(user_input):
    """
    Kullanıcı girişini köşe GCP matrisine çevirir.

    Parametreler
    ----------
    user_input : list of float, uzunluk 8
        [NW_lat, NW_lon, NE_lat, NE_lon, SE_lat, SE_lon, SW_lat, SW_lon]

    Döndürür
    -------
    np.ndarray, shape (4, 2) — [lon, lat] sırasında [NE, SE, SW, NW]
    """
    # MATLAB sırası: mPtsWld = [mPtsWld(2,:); mPtsWld(4,:); mPtsWld(3,:); mPtsWld(1,:)]
    # Yani: NE, SE, SW, NW (lon, lat) sırasında
    nw_lat, nw_lon = user_input[0], user_input[1]
    ne_lat, ne_lon = user_input[2], user_input[3]
    se_lat, se_lon = user_input[4], user_input[5]
    sw_lat, sw_lon = user_input[6], user_input[7]

    return np.array([
        [ne_lon, ne_lat],
        [se_lon, se_lat],
        [sw_lon, sw_lat],
        [nw_lon, nw_lat],
    ])


# ===========================================================================
# CHOOSE WINDOWS — Pencere Hesaplama Mantığı
# MATLAB karşılığı: extChooseWindows.m (GUI kısmı hariç)
# ===========================================================================

def compute_windows(H1, H2, img_size_l, img_size_r,
                    roi_list, win_size_pix=4400, buffer_pix=300):
    """
    Kullanıcının seçtiği ROI'lardan işleme pencerelerini hesaplar.

    MATLAB karşılığı: extChooseWindows.m içindeki pencere döngüsü

    Parametreler
    ----------
    H1, H2          : np.ndarray, shape (3,3)
        Sol ve sağ görüntü homojenlik matrisleri.
    img_size_l      : tuple (height, width)  — sol görüntü boyutu
    img_size_r      : tuple (height, width)  — sağ görüntü boyutu
    roi_list        : list of dict
        Her ROI için {'x_min','x_max','y_min','y_max'} sözlüğü
        (tam ölçek piksel koordinatları).
    win_size_pix    : int — hedef pencere boyutu (piksel)
    buffer_pix      : int — pencere kenar tamponu

    Döndürür
    -------
    windows : list of dict
        Her pencere için:
            'left'   : np.ndarray (2,2) — [[x_min,y_min],[x_max,y_max]]
            'right'  : np.ndarray (2,2)
            'region' : int — ROI numarası (1-tabanlı)
    """
    h_l, w_l = img_size_l
    h_r, w_r = img_size_r
    H2_inv_H1 = np.linalg.solve(H2, H1)

    windows = []

    for i_roi, roi in enumerate(roi_list):
        x_min = int(roi['x_min'])
        x_max = int(roi['x_max'])
        y_min = int(roi['y_min'])
        y_max = int(roi['y_max'])

        roi_w = x_max - x_min + 1
        roi_h = y_max - y_min + 1

        # ROI'yi alt pencerelere böl
        n_x = max(1, round(roi_w / win_size_pix))
        n_y = max(1, round(roi_h / win_size_pix))
        vX = np.round(np.linspace(x_min, x_max, n_x + 1)).astype(int)
        vY = np.round(np.linspace(y_min, y_max, n_y + 1)).astype(int)

        for ix in range(len(vX) - 1):
            for iy in range(len(vY) - 1):

                # Sol pencere (buffer ekle)
                win_l = np.array([
                    [vX[ix]   - buffer_pix, vY[iy]   - buffer_pix],
                    [vX[ix+1] + buffer_pix, vY[iy+1] + buffer_pix],
                ], dtype=float)

                # Sınırları kontrol et
                win_l[0, 0] = max(1,   win_l[0, 0])
                win_l[0, 1] = max(1,   win_l[0, 1])
                win_l[1, 0] = min(w_l, win_l[1, 0])
                win_l[1, 1] = min(h_l, win_l[1, 1])

                # Sol pencereyi sağ görüntüye dönüştür
                corners_l = np.array([
                    [win_l[0,0], win_l[0,1], 1],
                    [win_l[1,0], win_l[1,1], 1],
                ]).T
                corners_r = H2_inv_H1 @ corners_l
                corners_r = corners_r[:2] / corners_r[2:3]
                win_r = corners_r.T

                # Sınırları kontrol et
                win_r[0, 0] = max(1,   win_r[0, 0])
                win_r[0, 1] = max(1,   win_r[0, 1])
                win_r[1, 0] = min(w_r, win_r[1, 0])
                win_r[1, 1] = min(h_r, win_r[1, 1])

                # İki pencereyi aynı boyuta getir
                size_l = win_l[1] - win_l[0]
                size_r = win_r[1] - win_r[0]
                win_sz = np.minimum(size_l, size_r) - 1

                cen_l = win_l.mean(axis=0)
                cen_r = win_r.mean(axis=0)

                win_l = np.round(np.array([
                    cen_l - win_sz / 2,
                    cen_l + win_sz / 2,
                ])).astype(int)

                win_r = np.round(np.array([
                    cen_r - win_sz / 2,
                    cen_r + win_sz / 2,
                ])).astype(int)

                windows.append({
                    'left':   win_l,
                    'right':  win_r,
                    'region': i_roi + 1,   # 1-tabanlı (MATLAB uyumlu)
                })

    return windows


# ===========================================================================
# GET ROI — Bölgeye Ait Pencereleri Filtrele
# MATLAB karşılığı: extGetROI.m
# ===========================================================================

def ext_get_roi(files_l, files_r, roi_ids, target_roi):
    """
    Belirtilen ROI'ya ait ve başarıyla işlenmiş pencereleri döndürür.

    MATLAB imzası:
        [cL, cR] = extGetROI(cFileL, cFileR, vROI, iROI)

    Parametreler
    ----------
    files_l    : list of dict  — sol pencere nesneleri
    files_r    : list of dict  — sağ pencere nesneleri
    roi_ids    : list of int   — her pencerenin ROI numarası
    target_roi : int           — hedef ROI numarası

    Döndürür
    -------
    out_l, out_r : list of dict  — filtrelenmiş pencere nesneleri
    """
    if len(files_l) != len(files_r) or len(files_l) != len(roi_ids):
        raise ValueError("files_l, files_r ve roi_ids aynı uzunlukta olmalı.")

    # Hedef ROI'ya ait pencereleri seç
    out_l, out_r = [], []
    for fl, fr, rid in zip(files_l, files_r, roi_ids):
        if rid != target_roi:
            continue

        # Başarılı işlem kontrolü: 'ImagePoints' anahtarı mevcut ve dolu mu?
        has_pts_l = ('ImagePoints' in fl and
                     fl['ImagePoints'] is not None and
                     fl['ImagePoints'].size > 0)
        has_pts_r = ('ImagePoints' in fr and
                     fr['ImagePoints'] is not None and
                     fr['ImagePoints'].size > 0)

        if has_pts_l and has_pts_r:
            out_l.append(fl)
            out_r.append(fr)

    if not out_l:
        raise RuntimeError(
            f"ROI {target_roi} için başarıyla hesaplanmış disparite haritası yok."
        )

    return out_l, out_r


# ===========================================================================
# BASIT TEST
# ===========================================================================

if __name__ == '__main__':
    print("ext_windows.py — Temel testler\n")

    # -------------------------------------------------------------------
    # TEST 1: compute_spatial_transform
    # -------------------------------------------------------------------
    print("TEST 1: compute_spatial_transform")

    # Sentetik köşe GCPs — yaklaşık Türkiye merkezi
    # [NW_lat, NW_lon, NE_lat, NE_lon, SE_lat, SE_lon, SW_lat, SW_lon]
    user_input = [
        40.0, 30.0,   # NW
        40.0, 32.0,   # NE
        38.0, 32.0,   # SE
        38.0, 30.0,   # SW
    ]

    gcps = parse_corner_gcps(user_input)
    img_shape = (10000, 20000)   # tipik Hexagon görüntü boyutu

    T = compute_spatial_transform(gcps, img_shape)

    # Sol-üst köşe (piksel [1,1]) yaklaşık NW GCP'ye dönüşmeli
    pt_img = np.array([1, -1, 1.0])
    pt_wld = T @ pt_img
    print(f"  Sol-üst köşe piksel [1,-1] → dünya {pt_wld[:2].round(4)}")
    print(f"  Beklenen yaklaşık          → [32.0, 40.0] (NE)")
    err = np.abs(pt_wld[:2] - np.array([32.0, 40.0]))
    print(f"  Hata                       : {err.round(4)}")
    print(f"  TEST 1 BASARILI ✓\n" if err.max() < 1.0
          else f"  TEST 1 BASARISIZ ✗\n")

    # -------------------------------------------------------------------
    # TEST 2: compute_windows
    # -------------------------------------------------------------------
    print("TEST 2: compute_windows")

    H1 = np.eye(3)
    H2 = np.eye(3)
    img_size = (50000, 20000)

    roi_list = [
        {'x_min': 1000, 'x_max': 6000, 'y_min': 1000, 'y_max': 6000},
    ]

    windows = compute_windows(
        H1, H2, img_size, img_size,
        roi_list, win_size_pix=4400, buffer_pix=300
    )

    print(f"  ROI boyutu    : 5000x5000 piksel")
    print(f"  Pencere sayısı: {len(windows)}")
    for i, w in enumerate(windows):
        print(f"  Pencere {i+1}: sol={w['left'].tolist()}, "
              f"sağ={w['right'].tolist()}, bölge={w['region']}")
    print(f"  TEST 2 BASARILI ✓\n" if len(windows) >= 1
          else f"  TEST 2 BASARISIZ ✗\n")

    # -------------------------------------------------------------------
    # TEST 3: ext_get_roi
    # -------------------------------------------------------------------
    print("TEST 3: ext_get_roi")

    # 4 pencere — 2 tanesi ROI=1, 2 tanesi ROI=2
    # ROI=1'in ilki başarılı, ikincisi başarısız (ImagePoints yok)
    files_l = [
        {'ImagePoints': np.ones((3, 10)), 'id': 'L1'},  # ROI=1 ✓
        {'id': 'L2'},                                    # ROI=1 ✗
        {'ImagePoints': np.ones((3, 5)),  'id': 'L3'},  # ROI=2 ✓
        {'ImagePoints': np.ones((3, 8)),  'id': 'L4'},  # ROI=2 ✓
    ]
    files_r = [
        {'ImagePoints': np.ones((3, 10)), 'id': 'R1'},
        {'ImagePoints': np.ones((3, 6)),  'id': 'R2'},
        {'ImagePoints': np.ones((3, 5)),  'id': 'R3'},
        {'ImagePoints': np.ones((3, 8)),  'id': 'R4'},
    ]
    roi_ids = [1, 1, 2, 2]

    out_l1, out_r1 = ext_get_roi(files_l, files_r, roi_ids, target_roi=1)
    out_l2, out_r2 = ext_get_roi(files_l, files_r, roi_ids, target_roi=2)

    print(f"  ROI=1 başarılı pencere: {len(out_l1)} (beklenen 1)")
    print(f"  ROI=2 başarılı pencere: {len(out_l2)} (beklenen 2)")
    ok = (len(out_l1) == 1 and len(out_l2) == 2)
    print(f"  TEST 3 BASARILI ✓" if ok else f"  TEST 3 BASARISIZ ✗")
