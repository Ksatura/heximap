"""
ext_init_bundle.py

MATLAB kaynak dosyaları:
  - main/2_extract/extInitTrans.m
  - main/2_extract/extBundleAdjust.m

Açıklama:
  Hexagon stereo kamera çiftinin başlangıç poz matrislerini ORB eşleştirme
  ve temel matris ayrıştırması ile tahmin eder. Ardından yeniden projeksiyon
  hatasını doğrusal olmayan en küçük kareler (bundle adjustment) ile minimize
  eder.

Bağımlılıklar:
  - numpy  (pip install numpy)
  - opencv-python (pip install opencv-python)
  - scipy  (pip install scipy)

Dahili bağımlılıklar:
  - utils.py             (make_rot_mat, rotro2eu, triangulate)
  - ext_filter_images.py (ext_filter_images)
"""

import numpy as np
import cv2
from scipy.optimize import least_squares

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
sys.path.insert(0, str(Path(__file__).parent))

from utils import make_rot_mat, rotro2eu, triangulate
from ext_filter_images import ext_filter_images


# ===========================================================================
# EXT INIT TRANS — Başlangıç Poz Tahmini
# MATLAB karşılığı: extInitTrans.m
# ===========================================================================

def ext_init_trans(mat_l, mat_r, progress_cb=None):
    """
    İki Hexagon görüntüsü arasındaki başlangıç kamera poz matrislerini
    ORB eşleştirme ve temel matris ayrıştırması ile tahmin eder.

    MATLAB imzası:
        extInitTrans(cM, strHexPath, hW)

    Parametreler
    ----------
    mat_l, mat_r : dict
        Sol ve sağ görüntü nesneleri. Beklenen anahtarlar:
            'Image10'              : np.ndarray — 1/10 ölçek görüntü
            'SpatialTrans'         : np.ndarray (3,3) — affin dönüşüm
            'FocalLengthPixels'    : float
            'PrincipalPointPixels' : np.ndarray (2,)
        Çıktı olarak şunlar eklenir:
            'IntrinsicMatrix'  : np.ndarray (3,3)
            'LeftPoseMatrix'   : np.ndarray (3,4)
            'RightPoseMatrix'  : np.ndarray (3,4)
            'LeftHomography'   : np.ndarray (3,3)
            'RightHomography'  : np.ndarray (3,3)

    progress_cb : callable veya None

    Döndürür
    -------
    mat_l, mat_r : dict — güncellenerek döndürülür
    """
    def _progress(msg):
        if progress_cb:
            progress_cb(msg)

    _progress('başlangıç dönüşüm matrisleri hesaplanıyor...')

    # -----------------------------------------------------------------
    # Örtüşen bölgeleri belirle
    # -----------------------------------------------------------------
    h_l, w_l = mat_l['Image10'].shape[:2]
    h_r, w_r = mat_r['Image10'].shape[:2]

    T_l = mat_l['SpatialTrans']
    T_r = mat_r['SpatialTrans']

    win_l, win_r = _find_overlap(T_l, T_r, h_l, w_l, h_r, w_r)

    # -----------------------------------------------------------------
    # Nokta karşılıklarını hesapla (ORB)
    # -----------------------------------------------------------------
    pts1, pts2 = _get_point_correspondences(
        mat_l['Image10'], mat_r['Image10'], win_l, win_r
    )

    # -----------------------------------------------------------------
    # Kamera matrisleri ve poz tahmini (50 deneme)
    # -----------------------------------------------------------------
    num_tries = 50
    pose2_candidates = []

    for _ in range(num_tries):
        try:
            pts1s, pts2s, K1, K2, P1, P2 = _compute_matrices(
                mat_l, mat_r, pts1, pts2
            )
            pose2_candidates.append((P2, pts1s, pts2s, K1, K2, P1))
        except Exception:
            continue

    if not pose2_candidates:
        raise RuntimeError("Kamera poz matrisleri hesaplanamadı.")

    # Z koordinatı sıfıra en yakın çözümü seç
    best_P2, best_pts1, best_pts2, K1, K2, P1 = min(
        pose2_candidates,
        key=lambda x: abs(x[0][2, 3]) if x[0] is not None else float('inf')
    )

    # -----------------------------------------------------------------
    # Başlangıç homojenlik matrislerini hesapla
    # -----------------------------------------------------------------
    H1, H2 = _compute_homographies(K1, K2, best_pts1, best_pts2)

    # -----------------------------------------------------------------
    # Sonuçları kaydet
    # -----------------------------------------------------------------
    mat_l['IntrinsicMatrix']  = K1
    mat_l['LeftPoseMatrix']   = P1
    mat_l['RightPoseMatrix']  = best_P2
    mat_l['LeftHomography']   = H1
    mat_l['RightHomography']  = H2

    return mat_l, mat_r


# ===========================================================================
# EXT BUNDLE ADJUST — Bundle Adjustment
# MATLAB karşılığı: extBundleAdjust.m
# ===========================================================================

def ext_bundle_adjust(windows_l, windows_r, hex_path=None,
                      progress_cb=None):
    """
    Yeniden projeksiyon hatasını doğrusal olmayan en küçük kareler ile
    minimize eder (bundle adjustment).

    MATLAB imzası:
        extBundleAdjust(strHexPath, cL, cR, hW, cReg)

    Parametreler
    ----------
    windows_l, windows_r : list of dict
        İşlenmiş pencere nesneleri. Her dict şu anahtarları içermeli:
            'PointMatches'    : np.ndarray (2, N)
            'Window'          : np.ndarray (2, 2)
            'IntrinsicMatrix' : np.ndarray (3, 3)
            'PoseMatrix'      : np.ndarray (3, 4)
            'Accuracy'        : dict
    hex_path   : str veya None — önceki sonuçların aranacağı klasör
    progress_cb: callable veya None

    Döndürür
    -------
    windows_l, windows_r : list of dict — güncellenmiş
    """
    def _progress(msg):
        if progress_cb:
            progress_cb(msg)

    _progress('bundle adjustment yapılıyor...')

    # Başlangıç matrislerini seç
    K1 = windows_l[0]['IntrinsicMatrix']
    K2 = windows_r[0]['IntrinsicMatrix']
    P1 = windows_l[0].get('LeftPoseMatrix',
                           np.hstack([np.eye(3), np.zeros((3,1))]))
    P2 = windows_l[0].get('RightPoseMatrix',
                           windows_r[0].get('PoseMatrix', np.eye(3,4)))

    # Nokta eşleşmelerini tüm pencerelerden topla
    pts1, pts2 = _collect_points(windows_l, windows_r, n_per_window=2000)

    # Bundle adjustment — en fazla 3 tur
    f_val = np.inf
    count = 0
    while f_val > 100 and count < 3:
        K1, K2, P1, P2, error, f_val = _bundle_adjust_core(
            pts1, pts2, K1, K2, P1, P2
        )
        count += 1

    # Sonuçları tüm pencerelere kaydet
    for obj_l, obj_r in zip(windows_l, windows_r):
        obj_l['IntrinsicMatrix'] = K1
        obj_r['IntrinsicMatrix'] = K2
        obj_l['PoseMatrix']      = P1
        obj_r['PoseMatrix']      = P2

        acc = obj_l.get('Accuracy', {})
        acc['Reprojection']  = error
        acc['BundleAdjust']  = f_val
        obj_l['Accuracy'] = acc

    return windows_l, windows_r


# ===========================================================================
# İÇ YARDIMCI FONKSİYONLAR — extInitTrans
# ===========================================================================

def _find_overlap(T_l, T_r, h_l, w_l, h_r, w_r):
    """İki görüntünün örtüşen bölgelerini dünya koordinatında hesaplar."""

    def img_corners_world(T, h, w):
        # Görüntü köşeleri (y negatif, MATLAB uyumlu)
        corners = np.array([
            [1,   -1,   1],
            [w,   -h,   1],
            [w,   -1,   1],
            [1,   -h,   1],
        ], dtype=float).T
        wld = T @ corners
        return wld[:2].T

    corners_l = img_corners_world(T_l, h_l, w_l)
    corners_r = img_corners_world(T_r, h_r, w_r)

    # Örtüşme bölgesi: her iki görüntünün x aralığının kesişimi
    x_min = max(corners_l[:, 0].min(), corners_r[:, 0].min())
    x_max = min(corners_l[:, 0].max(), corners_r[:, 0].max())
    y_min = max(corners_l[:, 1].min(), corners_r[:, 1].min())
    y_max = min(corners_l[:, 1].max(), corners_r[:, 1].max())

    # Dünya koordinatlarını piksel koordinatlarına çevir (T^-1 ile)
    def world_to_img(T, pts_wld):
        pts_h = np.vstack([pts_wld.T, np.ones(pts_wld.shape[0])])
        pts_img = np.linalg.solve(T, pts_h)
        return pts_img[:2].T

    overlap_wld = np.array([[x_min, y_min], [x_max, y_max]])
    win_l = np.abs(world_to_img(T_l, overlap_wld)).astype(int)
    win_r = np.abs(world_to_img(T_r, overlap_wld)).astype(int)

    # Sırala ve sınırla
    for win, (h, w) in [(win_l, (h_l, w_l)), (win_r, (h_r, w_r))]:
        win[0, 0] = max(1, min(win[:, 0].min(), w))
        win[1, 0] = max(1, min(win[:, 0].max(), w))
        win[0, 1] = max(1, min(win[:, 1].min(), h))
        win[1, 1] = max(1, min(win[:, 1].max(), h))

    # Sifir boyutlu pencere kontrolu
    for win in [win_l, win_r]:
        if win[1, 0] <= win[0, 0]:
            win[1, 0] = win[0, 0] + 1
        if win[1, 1] <= win[0, 1]:
            win[1, 1] = win[0, 1] + 1

    return win_l, win_r


def _get_point_correspondences(img10_l, img10_r, win_l, win_r):
    """ORB ile iki görüntü arasında nokta karşılıkları hesaplar."""

    # Örtüşen bölgeleri kes (1/10 ölçekte)
    win_l10 = np.round(win_l / 10).astype(int)
    win_r10 = np.round(win_r / 10).astype(int)

    # y/x siralari ters gelebilir — her zaman [min:max] kullan
    ly0, ly1 = sorted([win_l10[0,1], win_l10[1,1]])
    lx0, lx1 = sorted([win_l10[0,0], win_l10[1,0]])
    ry0, ry1 = sorted([win_r10[0,1], win_r10[1,1]])
    rx0, rx1 = sorted([win_r10[0,0], win_r10[1,0]])
    # Sinir asimi kontrolu
    h10_l, w10_l = img10_l.shape[:2]
    h10_r, w10_r = img10_r.shape[:2]
    ly0, ly1 = max(0,ly0), min(h10_l, ly1)
    lx0, lx1 = max(0,lx0), min(w10_l, lx1)
    ry0, ry1 = max(0,ry0), min(h10_r, ry1)
    rx0, rx1 = max(0,rx0), min(w10_r, rx1)
    crop_l = img10_l[ly0:ly1, lx0:lx1]
    crop_r = img10_r[ry0:ry1, rx0:rx1]

    # Filtrele
    opt = {'histmatch': True, 'adapthisteq': True, 'wiener2': True}
    f1, f2 = ext_filter_images(crop_l, crop_r, opt=opt)
    f1_u8 = (f1 * 255).astype(np.uint8)
    f2_u8 = (f2 * 255).astype(np.uint8)

    # ORB eşleştirme
    orb     = cv2.ORB_create(nfeatures=30000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp1, desc1 = orb.detectAndCompute(f1_u8, None)
    kp2, desc2 = orb.detectAndCompute(f2_u8, None)

    if desc1 is None or desc2 is None or len(kp1) < 8 or len(kp2) < 8:
        raise RuntimeError("Yeterli ORB özelliği tespit edilemedi.")

    # 5x5 alt pencere döngüsü (MATLAB'daki gibi)
    h1, w1 = f1_u8.shape[:2]
    kp1_arr = np.array([k.pt for k in kp1])
    kp2_arr = np.array([k.pt for k in kp2])

    x_idx = np.round(np.linspace(0, w1, 6)).astype(int)
    y_idx = np.round(np.linspace(0, h1, 6)).astype(int)

    pts1_all, pts2_all = [], []

    for iy in range(5):
        for ix in range(5):
            mask1 = ((kp1_arr[:, 0] > x_idx[ix]) &
                     (kp1_arr[:, 0] < x_idx[ix+1]) &
                     (kp1_arr[:, 1] > y_idx[iy]) &
                     (kp1_arr[:, 1] < y_idx[iy+1]))
            mask2 = ((kp2_arr[:, 0] > x_idx[ix]) &
                     (kp2_arr[:, 0] < x_idx[ix+1]) &
                     (kp2_arr[:, 1] > y_idx[iy]) &
                     (kp2_arr[:, 1] < y_idx[iy+1]))

            if mask1.sum() < 4 or mask2.sum() < 4:
                continue

            idx1 = np.where(mask1)[0]
            idx2 = np.where(mask2)[0]

            sub_desc1 = desc1[idx1]
            sub_desc2 = desc2[idx2]

            try:
                matches = matcher.match(sub_desc1, sub_desc2)
                matches = sorted(matches, key=lambda m: m.distance)[:100]
            except Exception:
                continue

            for m in matches:
                pts1_all.append(kp1_arr[idx1[m.queryIdx]])
                pts2_all.append(kp2_arr[idx2[m.trainIdx]])

    if len(pts1_all) < 8:
        raise RuntimeError("Yeterli nokta karşılığı bulunamadı.")

    # Tam görüntü koordinatlarına çevir (x10)
    pts1 = (np.array(pts1_all) + win_l10[0]) * 10
    pts2 = (np.array(pts2_all) + win_r10[0]) * 10

    return pts1, pts2


def _compute_matrices(mat_l, mat_r, pts1, pts2):
    """Temel matris, esansiyel matris ve kamera poz matrislerini hesaplar."""
    # Rastgele %80 alt küme
    n = len(pts1)
    idx = np.random.choice(n, size=min(round(4/5*n), n), replace=False)
    p1 = pts1[idx].astype(np.float32)
    p2 = pts2[idx].astype(np.float32)

    # RANSAC ile aykırı değerleri temizle
    F, mask = cv2.findFundamentalMat(
        p1, p2, method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0, confidence=0.99
    )
    if F is None or mask is None:
        raise RuntimeError("Temel matris hesaplanamadı.")

    p1 = p1[mask.ravel() == 1]
    p2 = p2[mask.ravel() == 1]

    # Tam ölçeğe çevir
    p1 = p1 * 10
    p2 = p2 * 10

    # 8-nokta ile temel matris
    F8, _ = cv2.findFundamentalMat(p1, p2, method=cv2.FM_8POINT)
    if F8 is None:
        raise RuntimeError("8-nokta temel matris hesaplanamadı.")

    # Kamera iç parametre matrisleri
    f  = mat_l['FocalLengthPixels']
    pp1 = mat_l['PrincipalPointPixels']
    pp2 = mat_r['PrincipalPointPixels']

    K1 = np.array([[f, 0, pp1[0]], [0, f, pp1[1]], [0, 0, 1]], dtype=float)
    K2 = np.array([[f, 0, pp2[0]], [0, f, pp2[1]], [0, 0, 1]], dtype=float)

    # Poz matrislerini SVD ile hesapla
    P1, P2 = _decompose_essential(K1, K2, F8, p1, p2)

    return p1, p2, K1, K2, P1, P2


def _decompose_essential(K1, K2, F, pts1, pts2):
    """Esansiyel matrisi SVD ile ayrıştırarak kamera pozlarını bulur."""
    E = K2.T @ F @ K1

    # SVD
    U, _, Vt = np.linalg.svd(E)
    E_fixed = U @ np.diag([1, 1, 0]) @ Vt
    U, _, Vt = np.linalg.svd(E_fixed)

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)

    # 4 olası çözüm
    candidates = [
        np.hstack([U @ W @ Vt,  U[:, 2:3]]),
        np.hstack([U @ W @ Vt, -U[:, 2:3]]),
        np.hstack([U @ W.T @ Vt,  U[:, 2:3]]),
        np.hstack([U @ W.T @ Vt, -U[:, 2:3]]),
    ]

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])

    pts1_h = np.vstack([pts1.T, np.ones(len(pts1))])
    pts2_h = np.vstack([pts2.T, np.ones(len(pts2))])

    best_P2 = None
    best_score = -1

    for P2_c in candidates:
        if round(np.linalg.det(P2_c[:3, :3])) != 1:
            continue
        try:
            pts3d, _ = triangulate(pts1_h, pts2_h, P1, P2_c, K1, K2,
                                   compute_error=False)
            d1 = (P1 @ pts3d)[2]
            d2 = (P2_c @ pts3d)[2]
            score = ((d1 > 0).mean() + (d2 > 0).mean()) / 2
            if score > best_score:
                best_score = score
                best_P2 = P2_c
        except Exception:
            continue

    if best_P2 is None:
        best_P2 = candidates[0]

    return P1, best_P2


def _compute_homographies(K1, K2, pts1, pts2):
    """Kamera matrislerinden başlangıç homojenlik matrislerini hesaplar."""
    # Noktaları merkeze taşı
    c1 = pts1.mean(axis=0)
    c2 = pts2.mean(axis=0)
    p1c = pts1 - c1
    p2c = pts2 - c2

    p1h = np.vstack([p1c.T, np.ones(len(pts1))])
    p2h = np.vstack([p2c.T, np.ones(len(pts2))])

    # Basit başlangıç: birim homojenlik + öteleme düzeltmesi
    dx = np.mean(p1c[:, 0]) - np.mean(p2c[:, 0])
    H1 = K1 @ (np.eye(3) / K1)
    H2 = K2 @ (np.eye(3) / K2)
    H1[0, 2] = 0
    H2[0, 2] = dx

    # Çeviri matrislerini ekle
    T1 = np.eye(3); T1[:2, 2] = -c1
    T2 = np.eye(3); T2[:2, 2] = -c2
    H1 = H1 @ T1
    H2 = H2 @ T2

    return H1, H2


# ===========================================================================
# İÇ YARDIMCI FONKSİYONLAR — extBundleAdjust
# ===========================================================================

def _collect_points(windows_l, windows_r, n_per_window=2000):
    """Tüm pencerelerden nokta eşleşmelerini toplar."""
    n_win = len(windows_l)
    n_each = round(n_per_window / n_win)
    pts1_all, pts2_all = [], []

    for obj_l, obj_r in zip(windows_l, windows_r):
        pm_l = obj_l['PointMatches']  # (2, N)
        pm_r = obj_r['PointMatches']
        win_l = obj_l['Window']
        win_r = obj_r['Window']

        # Pencere ofsetini ekle
        p1 = pm_l.copy().astype(float)
        p2 = pm_r.copy().astype(float)
        p1[0] += win_l[0, 0] - 1
        p1[1] += win_l[0, 1] - 1
        p2[0] += win_r[0, 0] - 1
        p2[1] += win_r[0, 1] - 1

        n = p1.shape[1]
        idx = np.random.choice(n, size=min(n_each, n), replace=False)
        pts1_all.append(p1[:, idx])
        pts2_all.append(p2[:, idx])

    pts1 = np.hstack(pts1_all)
    pts2 = np.hstack(pts2_all)
    pts1_h = np.vstack([pts1, np.ones((1, pts1.shape[1]))])
    pts2_h = np.vstack([pts2, np.ones((1, pts2.shape[1]))])

    return pts1_h, pts2_h


def _bundle_adjust_core(pts1, pts2, K1, K2, P1, P2):
    """
    scipy least_squares ile yeniden projeksiyon hatasını minimize eder.
    MATLAB: lsqnonlin + computeMatrices + reprojectionError
    """
    # Parametre vektörü: [fx, fy, ppx1, ppy1, ppx2, ppy2, rx, ry, rz, tx, ty, tz]
    angles = rotro2eu('zyx', P2[:3, :3])  # radyan → derece
    angles_deg = np.degrees(angles)

    v0 = np.array([
        K1[0,0], K1[1,1],
        K1[0,2], K1[1,2],
        K2[0,2], K2[1,2],
        angles_deg[0], angles_deg[1], angles_deg[2],
        P2[0,3], P2[1,3], P2[2,3],
    ])

    # Sınırlar (MATLAB'daki gibi)
    dt = np.array([5,5, 5,5, 5,5, 5,5,5,
                   abs(P1[0,3]-P2[0,3])/2,
                   abs(P1[1,3]-P2[1,3])/2,
                   abs(P1[2,3]-P2[2,3])/2])
    dt = np.maximum(dt, 1e-6)
    lb = v0 - dt
    ub = v0 + dt

    # [0,1] aralığına ölçekle
    def scale(v): return (v - lb) / (ub - lb + 1e-12)
    def descale(v): return lb + v * (ub - lb + 1e-12)

    v0_scaled = scale(v0)

    def residuals(v_sc):
        v = descale(v_sc)
        K1o, K2o, P2o = _params_to_matrices(v, P1)
        _, err = triangulate(pts1, pts2, P1, P2o, K1o, K2o,
                             compute_error=True)
        # %98 quantile üstündeki aykırı değerleri sıfırla
        q98 = np.quantile(err, 0.98) if len(err) > 0 else 1.0
        err[err >= q98] = 0.0
        return err

    result = least_squares(
        residuals, v0_scaled,
        bounds=(np.zeros_like(v0_scaled), np.ones_like(v0_scaled)),
        max_nfev=100, ftol=1e-6, xtol=1e-6,
        method='trf',
    )

    v_opt = descale(result.x)
    K1_opt, K2_opt, P2_opt = _params_to_matrices(v_opt, P1)

    _, error = triangulate(pts1, pts2, P1, P2_opt, K1_opt, K2_opt,
                           compute_error=True)

    return K1_opt, K2_opt, P1, P2_opt, error, result.cost


def _params_to_matrices(v, P1):
    """Parametre vektöründen K1, K2, P2 matrislerini oluşturur."""
    K1 = np.array([[v[0], 0, v[2]], [0, v[1], v[3]], [0, 0, 1]])
    K2 = np.array([[v[0], 0, v[4]], [0, v[1], v[5]], [0, 0, 1]])
    R2 = make_rot_mat(v[6], v[7], v[8])
    P2 = np.hstack([R2, [[v[9]], [v[10]], [v[11]]]])
    return K1, K2, P2


# ===========================================================================
# BASIT TEST
# ===========================================================================

if __name__ == '__main__':
    print("ext_init_bundle.py — Temel testler\n")

    rng = np.random.default_rng(42)

    # -------------------------------------------------------------------
    # TEST 1: _decompose_essential — kamera poz ayrıştırma
    # -------------------------------------------------------------------
    print("TEST 1: Esansiyel matris ayrıştırma")

    # Gerçek poz: küçük öteleme
    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=float)
    R_true = make_rot_mat(2.0, -1.0, 0.5)
    t_true = np.array([[50.0], [0.0], [0.0]])
    P1_true = np.hstack([np.eye(3), np.zeros((3,1))])
    P2_true = np.hstack([R_true, t_true])

    # Sentetik 3D noktalar ve projeksiyonlar
    pts3d = rng.random((3, 30)) * 200 + np.array([[0],[0],[500]])
    pts3d_h = np.vstack([pts3d, np.ones((1,30))])

    proj1 = K @ P1_true @ pts3d_h
    proj1 /= proj1[2:3]
    proj2 = K @ P2_true @ pts3d_h
    proj2 /= proj2[2:3]

    # Temel matris hesapla
    F, mask = cv2.findFundamentalMat(
        proj1[:2].T.astype(np.float32),
        proj2[:2].T.astype(np.float32),
        cv2.FM_8POINT
    )

    P1_est, P2_est = _decompose_essential(K, K, F,
                                          proj1[:2].T, proj2[:2].T)

    # P2'nin sıfırdan farklı olduğunu kontrol et
    t_est = P2_est[:, 3]
    t_norm = np.linalg.norm(t_est)
    print(f"  Tahmin edilen P2 ötelemesi normu : {t_norm:.4f}")
    print(f"  Gerçek P2 ötelemesi normu        : {np.linalg.norm(t_true):.4f}")
    print(f"  TEST 1 BASARILI ✓\n" if t_norm > 0.01
          else f"  TEST 1 BASARISIZ ✗\n")

    # -------------------------------------------------------------------
    # TEST 2: _bundle_adjust_core
    # -------------------------------------------------------------------
    print("TEST 2: Bundle adjustment optimizasyonu")

    # Sentetik stereo kurulum
    K1 = np.array([[1000,0,320],[0,1000,240],[0,0,1]], dtype=float)
    K2 = K1.copy()
    P1 = np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = np.hstack([make_rot_mat(1.0, -0.5, 0.2),
                    np.array([[60.],[0.],[0.]])])

    pts3d = rng.random((3, 50)) * 100 + np.array([[0],[0],[300]])
    pts3d_h = np.vstack([pts3d, np.ones((1,50))])

    proj1 = K1 @ P1 @ pts3d_h; proj1 /= proj1[2:3]
    proj2 = K2 @ P2 @ pts3d_h; proj2 /= proj2[2:3]

    # Gürültü ekle
    proj1[:2] += rng.normal(0, 0.5, proj1[:2].shape)
    proj2[:2] += rng.normal(0, 0.5, proj2[:2].shape)

    # Bozulmuş başlangıç matrisi ile test et
    K1_noisy = K1.copy(); K1_noisy[0,0] += 20
    P2_noisy  = P2.copy(); P2_noisy[0,3] += 5

    K1_opt, K2_opt, _, P2_opt, error, cost = _bundle_adjust_core(
        proj1, proj2, K1_noisy, K2.copy(), P1, P2_noisy
    )

    print(f"  Başlangıç fx          : {K1_noisy[0,0]:.1f}")
    print(f"  Optimize edilmiş fx   : {K1_opt[0,0]:.1f}")
    print(f"  Gerçek fx             : {K1[0,0]:.1f}")
    print(f"  Final maliyet         : {cost:.4f}")
    print(f"  Ortalama reprojection : {error.mean():.4f} piksel")
    print(f"  TEST 2 BASARILI ✓\n" if cost < 1e6
          else f"  TEST 2 BASARISIZ ✗\n")

    # -------------------------------------------------------------------
    # TEST 3: _params_to_matrices gidiş-dönüş
    # -------------------------------------------------------------------
    print("TEST 3: Parametre vektörü gidiş-dönüş")

    K_test = np.array([[950,0,310],[0,948,238],[0,0,1]], dtype=float)
    R_test = make_rot_mat(3.0, -2.0, 1.0)
    P2_test = np.hstack([R_test, [[40.],[10.],[-5.]]])

    angles = np.degrees(rotro2eu('zyx', R_test))
    v_test = np.array([
        K_test[0,0], K_test[1,1],
        K_test[0,2], K_test[1,2],
        K_test[0,2], K_test[1,2],
        angles[0], angles[1], angles[2],
        40., 10., -5.,
    ])

    _, _, P2_back = _params_to_matrices(v_test, P1)
    err_R = np.max(np.abs(P2_back[:,:3] - R_test))
    err_t = np.max(np.abs(P2_back[:,3] - P2_test[:,3]))
    print(f"  Rotasyon hatası : {err_R:.2e}")
    print(f"  Öteleme hatası  : {err_t:.2e}")
    print(f"  TEST 3 BASARILI ✓" if err_R < 1e-6 and err_t < 1e-6
          else f"  TEST 3 BASARISIZ ✗")
