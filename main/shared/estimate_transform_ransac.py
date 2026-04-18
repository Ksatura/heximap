"""
estimate_transform_ransac.py

MATLAB kaynak dosyaları:
  - main/shared/estimateTransformRansac.m
  - main/shared/fileExchange/absor/absor.m

Açıklama:
  Horn'un mutlak oryantasyon yöntemi (absor) ve basit RANSAC implementasyonu.
  2D/3D nokta çiftleri arasında ölçekli rijit dönüşüm (Similarity Transform)
  tahmin eder.

Bağımlılıklar:
  - numpy (pip install numpy)
"""

import numpy as np


# ===========================================================================
# ABSOR — Horn'un Mutlak Oryantasyon Yöntemi
# MATLAB karşılığı: shared/fileExchange/absor/absor.m
# ===========================================================================

def absor(pts_moving, pts_fixed, do_scale=True):
    """
    Horn'un mutlak oryantasyon yöntemi ile iki nokta kümesi arasındaki
    ölçekli rijit dönüşümü (R, t, s) tahmin eder.

    Parametreler
    ----------
    pts_moving : np.ndarray, shape (3, N)
        Dönüştürülecek nokta kümesi (MATLAB'daki mPts2 konumuna karşılık gelir)
    pts_fixed : np.ndarray, shape (3, N)
        Hedef nokta kümesi (MATLAB'daki mPts1 konumuna karşılık gelir)
    do_scale : bool
        True ise ölçek faktörü de hesaplanır (MATLAB'da 'doScale', 1)

    Döndürür
    -------
    M : np.ndarray, shape (4, 4)
        Homojen dönüşüm matrisi [s*R | t; 0 0 0 1]
    """
    assert pts_moving.shape == pts_fixed.shape, \
        "Nokta kümeleri aynı boyutta olmalıdır."
    assert pts_moving.shape[0] == 3, \
        "Nokta kümeleri 3xN boyutunda olmalıdır."

    # Sonlu (finite) noktaları seç
    mask = (np.all(np.isfinite(pts_moving), axis=0) &
            np.all(np.isfinite(pts_fixed),  axis=0))
    A = pts_moving[:, mask]   # hareketli noktalar
    B = pts_fixed[:, mask]    # sabit noktalar

    n = A.shape[1]
    assert n >= 3, "En az 3 geçerli nokta çifti gereklidir."

    # Ağırlık merkezi hesapla
    centroid_A = A.mean(axis=1, keepdims=True)
    centroid_B = B.mean(axis=1, keepdims=True)

    # Merkeze taşı
    A_c = A - centroid_A
    B_c = B - centroid_B

    # Kovaryans matrisi
    H = A_c @ B_c.T

    # SVD ayrıştırması
    U, S, Vt = np.linalg.svd(H)

    # Rotasyon matrisi (yansıma kontrolü dahil)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    # Ölçek faktörü (Horn 1987 — Eq. 40)
    if do_scale:
        var_A = np.sum(A_c ** 2) / n  # hareketli nokta varyansı
        var_B = np.sum(B_c ** 2) / n  # sabit nokta varyansı
        s = np.sqrt(var_B / var_A) if var_A > 0 else 1.0
    else:
        s = 1.0

    # Öteleme vektörü
    t = centroid_B - s * R @ centroid_A

    # 4x4 homojen dönüşüm matrisi
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3:] = t

    return M


# ===========================================================================
# RANSAC YARDIMCI FONKSİYONLARI
# MATLAB karşılığı: shared/estimateTransformRansac.m (iç fonksiyonlar)
# ===========================================================================

def _apply_transform(M, pts):
    """
    Homojen dönüşüm matrisini noktalara uygular.

    Parametreler
    ----------
    M   : np.ndarray, shape (4, 4)
    pts : np.ndarray, shape (3, N)  — homojen olmayan 3D noktalar

    Döndürür
    -------
    np.ndarray, shape (3, N)
    """
    pts_h = np.vstack([pts, np.ones((1, pts.shape[1]))])  # (4, N)
    pts_t = M @ pts_h                                      # (4, N)
    return pts_t[:3]                                       # (3, N)


def _transform_distance(M, pts1, pts2):
    """
    Dönüştürülmüş pts2 ile pts1 arasındaki Öklid mesafelerini hesaplar.
    MATLAB: transDist()

    Parametreler
    ----------
    M    : np.ndarray, shape (4, 4)
    pts1 : np.ndarray, shape (3, N)
    pts2 : np.ndarray, shape (3, N)

    Döndürür
    -------
    np.ndarray, shape (N,)
    """
    pts2_t = _apply_transform(M, pts2)
    return np.sqrt(np.sum((pts2_t - pts1) ** 2, axis=0))


def _fit_transform(pts1, pts2):
    """
    Bir alt küme üzerinde absor ile dönüşüm tahmin eder.
    MATLAB: trans()
    """
    return absor(pts2, pts1, do_scale=True)


# ===========================================================================
# ANA FONKSİYON
# MATLAB karşılığı: estimateTransformRansac()
# ===========================================================================

def estimate_transform_ransac(pts1, pts2, num_iter=2000, inlier_dist=1.0):
    """
    RANSAC ile iki nokta kümesi arasındaki 2D/3D dönüşümü tahmin eder.

    MATLAB imzası:
        [mT, lKeep] = estimateTransformRansac(mPts1, mPts2, sRansac)
        sRansac.numIter    → num_iter
        sRansac.inlierDist → inlier_dist

    Parametreler
    ----------
    pts1       : np.ndarray, shape (3, N) — homojen 2D veya 3D noktalar
    pts2       : np.ndarray, shape (3, N) — homojen 2D veya 3D noktalar
    num_iter   : int    — RANSAC iterasyon sayısı (MATLAB'da sRansac.numIter)
    inlier_dist: float  — inlier eşiği piksel cinsinden (sRansac.inlierDist)

    Döndürür
    -------
    M      : np.ndarray, shape (4, 4) — en iyi dönüşüm matrisi
    is_inlier : np.ndarray, shape (N,), dtype bool — inlier maskesi
    """
    # Girdi kontrolü
    if pts1 is None or pts2 is None:
        raise ValueError("Nokta dizileri boş olamaz.")
    if pts1.shape != pts2.shape:
        raise ValueError("Nokta dizileri aynı boyutta olmalıdır.")
    if pts1.shape[0] != 3:
        raise ValueError("Nokta dizileri 3xN boyutunda olmalıdır.")

    n = pts1.shape[1]
    subset_size = 3          # MATLAB'da iSetSize = 3
    best_count = -1
    best_M = np.eye(4)

    rng = np.random.default_rng()

    for _ in range(num_iter):
        # Rastgele alt küme seç
        idx = rng.choice(n, size=min(subset_size, n), replace=False)

        try:
            M_candidate = _fit_transform(pts1[:, idx], pts2[:, idx])
        except Exception:
            continue

        # Consensus kümesini say
        dists = _transform_distance(M_candidate, pts1, pts2)
        count = int(np.sum(dists < inlier_dist))

        if count > best_count:
            best_count = count
            best_M = M_candidate

    # Tüm inlier'lar ile nihai dönüşümü yeniden hesapla
    is_inlier = _transform_distance(best_M, pts1, pts2) < inlier_dist

    if is_inlier.sum() >= subset_size:
        best_M = _fit_transform(pts1[:, is_inlier], pts2[:, is_inlier])

    return best_M, is_inlier


# ===========================================================================
# BASIT TEST
# ===========================================================================

if __name__ == "__main__":
    print("estimate_transform_ransac.py — Temel test")

    # Rastgele 3D nokta kümesi oluştur
    rng = np.random.default_rng(42)
    pts_orig = rng.random((3, 50)) * 100

    # Gerçek bir dönüşüm tanımla (küçük rotasyon + öteleme + ölçek)
    angle = np.radians(15)
    R_true = np.array([
        [ np.cos(angle), -np.sin(angle), 0],
        [ np.sin(angle),  np.cos(angle), 0],
        [             0,              0, 1]
    ])
    s_true = 1.05
    t_true = np.array([[5], [3], [0]])

    # Dönüştürülmüş noktalar (biraz gürültü ekle)
    pts_trans = s_true * R_true @ pts_orig + t_true
    pts_trans += rng.normal(0, 0.1, pts_trans.shape)

    # Homojene çevir (3xN)
    pts1_h = pts_orig
    pts2_h = pts_trans

    # RANSAC ile tahmin et
    M_est, inliers = estimate_transform_ransac(
        pts1_h, pts2_h, num_iter=500, inlier_dist=2.0
    )

    print(f"  Toplam nokta    : {pts_orig.shape[1]}")
    print(f"  Inlier sayisi   : {inliers.sum()}")
    print(f"  Tahmin edilen M :\n{np.round(M_est, 3)}")
    print("\n  Test BASARILI ✓" if inliers.sum() > 40 else "\n  Test BASARISIZ ✗")
