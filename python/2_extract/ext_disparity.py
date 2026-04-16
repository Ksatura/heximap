"""
ext_disparity.py

MATLAB kaynak dosyası:
  - main/2_extract/extDisparity.m

Açıklama:
  Semi-Global Block Matching (SGBM) algoritması ile stereo görüntü
  çiftinden disparite haritası hesaplar. Çok ölçekli (coarse-to-fine)
  yaklaşım kullanır.

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
from scipy.ndimage import binary_dilation, label
from skimage.morphology import remove_small_objects

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ext_filter_images import ext_filter_images


# ===========================================================================
# ANA FONKSİYON
# MATLAB karşılığı: extDisparity()
# ===========================================================================

def ext_disparity(obj_l, obj_r, resolution='1/2', block_size=7,
                  progress_cb=None):
    """
    Stereo görüntü çiftinden çok ölçekli SGBM ile disparite haritası üretir.

    MATLAB imzası:
        extDisparity(objL, objR, strRes, iBlkSz, hW, cWin)

    Parametreler
    ----------
    obj_l, obj_r : dict
        Stereo görüntü nesneleri. Beklenen anahtarlar:
            'RectImage'   : np.ndarray — doğrultulmuş görüntü
            'Window'      : np.ndarray, shape (2,2) — pencere koordinatları
            'RectWindow'  : np.ndarray, shape (2,2) — doğrultulmuş pencere
            'Homography'  : np.ndarray, shape (3,3)
        Fonksiyon çıktısı olarak şunlar eklenir:
            'Disparity'      : np.ndarray — disparite haritası
            'DisparityScale' : tuple      — ölçek faktörleri
            'ImagePoints'    : np.ndarray — görüntü noktaları

    resolution : str
        'Full' | '1/2' | '1/4' | '1/8'
        MATLAB'daki strRes parametresine karşılık gelir.

    block_size : int
        SGBM blok boyutu (tek sayı, 3–11 arası önerilir).

    progress_cb : callable veya None
        İlerleme geri çağrısı: progress_cb(mesaj)

    Döndürür
    -------
    obj_l : dict  — 'Disparity', 'DisparityScale', 'ImagePoints' eklenerek
    obj_r : dict  — 'ImagePoints' eklenerek
    """

    def _progress(msg):
        if progress_cb:
            progress_cb(msg)

    _progress('görüntüler filtreleniyor...')

    # Filtreleme seçenekleri
    opt = {'histmatch': True, 'adapthisteq': True, 'wiener2': False}
    filt_l, filt_r = ext_filter_images(
        obj_l['RectImage'], obj_r['RectImage'], empty_val=0, opt=opt
    )
    # [0,1] float → uint8 (SGBM uint8 ister)
    filt_l_u8 = (filt_l * 255).astype(np.uint8)
    filt_r_u8 = (filt_r * 255).astype(np.uint8)

    # -----------------------------------------------------------------
    # Ölçek dizisi (coarse → fine)
    # MATLAB: vScale = [16 8 4 2 1] veya [16 8 4 2] vb.
    # -----------------------------------------------------------------
    scale_map = {
        'Full': [16, 8, 4, 2, 1],
        '1/2':  [16, 8, 4, 2],
        '1/4':  [16, 8, 4],
        '1/8':  [16, 8],
    }
    if resolution not in scale_map:
        raise ValueError(f"Geçersiz çözünürlük: '{resolution}'. "
                         f"'Full', '1/2', '1/4', '1/8' olmalı.")
    scales = scale_map[resolution]

    # -----------------------------------------------------------------
    # SGBM parametreleri (MATLAB ile aynı)
    # -----------------------------------------------------------------
    vD = [-8 * 30, 8 * 30]
    sgbm_params = {
        'MinDisparity':      vD[0],
        'NumDisparities':    vD[1] - vD[0],
        'BlockSize':         block_size,
        'P1':                0,                    # MATLAB'da P1=sStereo.P2=0
        'P2':                64 * block_size ** 2, # MATLAB'da P2=sStereo.P1
        'Disp12MaxDiff':     1,
        'PreFilterCap':      0,
        'UniquenessRatio':   0,
        'SpeckleWindowSize': 300,
        'SpeckleRange':      1,
        'Mode':              cv2.STEREO_SGBM_MODE_SGBM,
    }
    outlier_thresh = 50.0

    _progress('disparite haritası hesaplanıyor...')

    disparity_map = None
    lo_res_map    = None
    final_scale   = None

    for i, scale in enumerate(scales):

        disp, scale_factors = _sgbm(
            filt_l_u8, filt_r_u8, sgbm_params, scale
        )

        # Ölçeği tam boyuta uygula
        disp = disp * scale_factors[1]

        if i > 0 and lo_res_map is not None:
            # Aykırı değerleri düşük çözünürlüklü harita ile temizle
            lo_resized = cv2.resize(
                lo_res_map, (disp.shape[1], disp.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
            outlier_mask = np.abs(disp - lo_resized) > outlier_thresh
            disp[outlier_mask] = np.nan

            if i < len(scales) - 1:
                # Boşlukları düşük çözünürlüklü haritayla doldur
                fill_mask = ~np.isnan(lo_resized) & np.isnan(disp)
                disp[fill_mask] = lo_resized[fill_mask]

        if i < len(scales) - 1:
            lo_res_map = disp.copy()
        else:
            # Son ölçek — geçerli pikselleri kaydet
            valid_mask  = ~np.isnan(disp.ravel())
            final_scale = scale_factors

        disparity_map = disp

    # -----------------------------------------------------------------
    # Sonuçları obj_l ve obj_r'ye kaydet
    # -----------------------------------------------------------------
    obj_l['Disparity']      = disparity_map
    obj_l['DisparityScale'] = final_scale

    # Görüntü noktaları hesapla (MATLAB: ImagePoints)
    hex_win_l  = obj_l['Window']
    hex_win_r  = obj_r['Window']
    proj_win   = obj_l['RectWindow']
    H1         = obj_l['Homography']
    H2         = obj_r['Homography']

    sz_full = np.array(obj_l['RectImage'].shape[:2])
    sz_disp = np.array(disparity_map.shape[:2])

    obj_l['ImagePoints'] = _compute_image_points(
        disparity_map, proj_win, H1, hex_win_l, sz_full, sz_disp, disp=None
    )
    obj_r['ImagePoints'] = _compute_image_points(
        disparity_map, proj_win, H2, hex_win_r, sz_full, sz_disp,
        disp=disparity_map
    )

    return obj_l, obj_r


# ===========================================================================
# SGBM YARDIMCI FONKSİYONU
# MATLAB karşılığı: sgbm() iç fonksiyonu
# ===========================================================================

def _sgbm(img_l, img_r, params, scale):
    """
    Belirtilen ölçekte SGBM disparite haritası hesaplar.

    Parametreler
    ----------
    img_l, img_r : np.ndarray  — uint8 görüntüler
    params       : dict        — SGBM parametreleri
    scale        : int/float   — küçültme faktörü (1 = tam boyut)

    Döndürür
    -------
    disp         : np.ndarray  — float32 disparite haritası (NaN = geçersiz)
    scale_factors: tuple       — (dikey_ölçek, yatay_ölçek)
    """
    h, w = img_l.shape[:2]
    new_h = max(1, round(h / scale))
    new_w = max(1, round(w / scale))

    scale_y = h / new_h
    scale_x = w / new_w

    # Disparity arama aralığını ölçekle (16'nın katı olmalı)
    vd_min = params['MinDisparity']
    vd_max = vd_min + params['NumDisparities']
    vd_min_s = int(round(vd_min / scale / 8) * 8)
    vd_max_s = int(round(vd_max / scale / 8) * 8)

    if abs(vd_min_s) < 8 or abs(vd_max_s) < 8:
        vd_min_s, vd_max_s = -8, 8
    num_disp = vd_max_s - vd_min_s
    if num_disp <= 0 or num_disp % 16 != 0:
        num_disp = max(16, (abs(num_disp) // 16 + 1) * 16)

    # Görüntüleri küçült
    small_l = cv2.resize(img_l, (new_w, new_h), interpolation=cv2.INTER_AREA)
    small_r = cv2.resize(img_r, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # SGBM nesnesi oluştur
    stereo = cv2.StereoSGBM_create(
        minDisparity=vd_min_s,
        numDisparities=num_disp,
        blockSize=params['BlockSize'],
        P1=params['P1'],
        P2=params['P2'],
        disp12MaxDiff=params['Disp12MaxDiff'],
        preFilterCap=params['PreFilterCap'],
        uniquenessRatio=params['UniquenessRatio'],
        speckleWindowSize=params['SpeckleWindowSize'],
        speckleRange=params['SpeckleRange'],
        mode=params['Mode'],
    )

    # Disparite haritasını hesapla (OpenCV 16x sabit nokta döndürür)
    raw = stereo.compute(small_l, small_r).astype(np.float32) / 16.0

    # Geçersiz dispariteleri NaN yap
    disp = raw.astype(np.float64)
    disp[(disp < vd_min_s) | (disp > vd_max_s)] = np.nan

    # Kenarlardaki boş bölgeleri maskele (MATLAB: imclearborder + imdilate)
    mask_l = cv2.resize(img_l, (new_w, new_h)) == 0
    mask_r = cv2.resize(img_r, (new_w, new_h)) == 0
    border_mask = mask_l | mask_r
    # Kenarlara temas eden bölgeleri koru (imclearborder benzeri)
    labeled, _ = label(border_mask)
    border_labels = set()
    for edge_arr in [labeled[0, :], labeled[-1, :],
                     labeled[:, 0], labeled[:, -1]]:
        border_labels.update(edge_arr[edge_arr > 0].tolist())
    interior_mask = np.isin(labeled, list(border_labels))
    # Genişlet (imdilate)
    dil_size = max(1, int(np.ceil(100 / scale)))
    struct = np.ones((dil_size, dil_size), dtype=bool)
    dilated = binary_dilation(interior_mask, structure=struct)
    disp[dilated] = np.nan

    return disp, (scale_y, scale_x)


# ===========================================================================
# GÖRÜNTÜ NOKTALARI HESAPLAMA
# MATLAB karşılığı: extDisparity.m > ImagePoints hesabı
# ===========================================================================

def _compute_image_points(disp_map, proj_win, H, hex_win,
                          sz_full, sz_disp, disp=None):
    """
    Disparite haritasındaki geçerli piksellerden görüntü noktaları üretir.

    Parametreler
    ----------
    disp_map : np.ndarray     — disparite haritası (NaN = geçersiz)
    proj_win : np.ndarray     — doğrultulmuş pencere koordinatları (2,2)
    H        : np.ndarray     — homojenlik dönüşüm matrisi (3,3)
    hex_win  : np.ndarray     — Hexagon görüntü pencere koordinatları (2,2)
    sz_full  : array-like     — tam ölçek görüntü boyutu [H, W]
    sz_disp  : array-like     — disparite haritası boyutu [H, W]
    disp     : np.ndarray veya None  — varsa disparity offseti (sağ görüntü için)

    Döndürür
    -------
    np.ndarray, shape (3, N)  — homojen görüntü noktaları
    """
    valid = ~np.isnan(disp_map.ravel())

    # Koordinat ızgarası
    x_range = np.arange(proj_win[0, 0], proj_win[1, 0] + 1)
    y_range = np.arange(proj_win[0, 1], proj_win[1, 1] + 1)
    mX, mY  = np.meshgrid(x_range, y_range)

    # Disparite ölçeğine küçült
    scale_h = sz_full[0] / sz_disp[0]
    scale_w = sz_full[1] / sz_disp[1]
    mX_s = cv2.resize(mX.astype(np.float32),
                      (sz_disp[1], sz_disp[0]),
                      interpolation=cv2.INTER_LINEAR)
    mY_s = cv2.resize(mY.astype(np.float32),
                      (sz_disp[1], sz_disp[0]),
                      interpolation=cv2.INTER_LINEAR)

    if disp is not None:
        # Sağ görüntü: x koordinatını disparity ile kaydır
        mX_s = mX_s - disp_map

    # Geçerli pikselleri seç
    x_valid = mX_s.ravel()[valid]
    y_valid = mY_s.ravel()[valid]
    ones    = np.ones(valid.sum())
    pts_h   = np.vstack([x_valid, y_valid, ones])   # (3, N)

    # Ters homojenlik dönüşümü uygula
    pts_t   = np.linalg.solve(H, pts_h)
    pts_t   = pts_t / pts_t[2:3, :]

    # Hexagon pencere ofsetini ekle
    pts_t[0] += hex_win[0, 0] - 1
    pts_t[1] += hex_win[0, 1] - 1
    pts_t[2]  = 1.0

    return pts_t


# ===========================================================================
# BASIT TEST
# ===========================================================================

if __name__ == '__main__':
    print("ext_disparity.py — Temel testler\n")

    rng = np.random.default_rng(0)

    # Sentetik stereo görüntü çifti (sağa kaydırılmış)
    base = (rng.random((200, 400)) * 200 + 30).astype(np.uint8)
    shift = 8
    M_shift = np.float32([[1, 0, -shift], [0, 1, 0]])
    shifted = cv2.warpAffine(base, M_shift, (400, 200))

    H_eye = np.eye(3)
    proj_win = np.array([[0, 0], [399, 199]])
    hex_win  = np.array([[1, 1], [400, 200]])

    obj_l = {
        'RectImage':   base,
        'Window':      hex_win,
        'RectWindow':  proj_win,
        'Homography':  H_eye,
    }
    obj_r = {
        'RectImage':   shifted,
        'Window':      hex_win,
        'RectWindow':  proj_win,
        'Homography':  H_eye,
    }

    # -------------------------------------------------------------------
    # TEST 1: ext_filter_images entegrasyonu
    # -------------------------------------------------------------------
    print("TEST 1: Görüntü filtreleme")
    from ext_filter_images import ext_filter_images
    opt = {'histmatch': True, 'adapthisteq': True, 'wiener2': False}
    f1, f2 = ext_filter_images(base.copy(), shifted.copy(), opt=opt)
    assert f1.shape == base.shape
    assert f1.max() <= 1.0
    print(f"  Filtre çıktısı boyutu : {f1.shape}, dtype: {f1.dtype}")
    print(f"  TEST 1 BASARILI ✓\n")

    # -------------------------------------------------------------------
    # TEST 2: StereoSGBM_create
    # -------------------------------------------------------------------
    print("TEST 2: SGBM disparite haritası")
    stereo = cv2.StereoSGBM_create(
        minDisparity=-16,
        numDisparities=32,
        blockSize=7,
        P1=0,
        P2=64 * 49,
        disp12MaxDiff=1,
        preFilterCap=0,
        uniquenessRatio=0,
        speckleWindowSize=300,
        speckleRange=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM,
    )
    f1_u8 = (f1 * 255).astype(np.uint8)
    f2_u8 = (f2 * 255).astype(np.uint8)
    raw   = stereo.compute(f1_u8, f2_u8).astype(np.float32) / 16.0
    valid_pct = np.sum((raw > -16) & (raw < 16)) / raw.size * 100
    print(f"  Disparite haritası boyutu : {raw.shape}")
    print(f"  Geçerli piksel oranı      : %{valid_pct:.1f}")
    print(f"  TEST 2 BASARILI ✓\n" if valid_pct > 10
          else f"  TEST 2 BASARISIZ ✗\n")

    # -------------------------------------------------------------------
    # TEST 3: Tam ext_disparity çağrısı
    # -------------------------------------------------------------------
    print("TEST 3: Tam ext_disparity çağrısı (1/8 çözünürlük)")
    try:
        out_l, out_r = ext_disparity(obj_l, obj_r,
                                     resolution='1/8', block_size=7)
        disp = out_l['Disparity']
        nan_pct = np.isnan(disp).mean() * 100
        print(f"  Disparite boyutu  : {disp.shape}")
        print(f"  NaN oranı         : %{nan_pct:.1f}")
        print(f"  Ölçek faktörleri  : {out_l['DisparityScale']}")
        print(f"  TEST 3 BASARILI ✓")
    except Exception as e:
        print(f"  TEST 3 BASARISIZ ✗ — {e}")
