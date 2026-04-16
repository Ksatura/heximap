"""
utils.py

MATLAB kaynak dosyaları:
  - main/shared/makeRotMat.m
  - main/shared/triangulate.m
  - main/shared/fileExchange/voicebox/rotro2eu.m
  - main/shared/getFiles.m

Açıklama:
  Paylaşılan yardımcı fonksiyonlar. Rotasyon matrisleri, 3D üçgenleme,
  Euler açı dönüşümü ve dosya listeleme işlemleri.

Bağımlılıklar:
  - numpy  (pip install numpy)
  - scipy  (pip install scipy)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path


# ===========================================================================
# MAKE ROT MAT
# MATLAB karşılığı: shared/makeRotMat.m
# ===========================================================================

def make_rot_mat(rx_deg, ry_deg, rz_deg):
    """
    3 Euler açısından rotasyon matrisi üretir (ZYX sırası).

    MATLAB imzası:
        mR = makeRotMat(dRx, dRy, dRz, 'deg')

    Parametreler
    ----------
    rx_deg, ry_deg, rz_deg : float  — derece cinsinden Euler açıları

    Döndürür
    -------
    R : np.ndarray, shape (3, 3)
    """
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)

    # X ekseni etrafında rotasyon
    Rx = np.array([
        [1,       0,        0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx),  np.cos(rx)],
    ])

    # Y ekseni etrafında rotasyon
    Ry = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [0,           1,           0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])

    # Z ekseni etrafında rotasyon
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1],
    ])

    # MATLAB sırası: Rx * Ry * Rz
    return Rx @ Ry @ Rz


# ===========================================================================
# ROTRO2EU — Rotasyon Matrisi → Euler Açıları
# MATLAB karşılığı: shared/fileExchange/voicebox/rotro2eu.m
# ===========================================================================

def rotro2eu(convention, R):
    """
    3x3 rotasyon matrisini Euler açılarına dönüştürür.

    MATLAB imzası:
        e = rotro2eu('zyx', mR)

    Parametreler
    ----------
    convention : str  — eksen sırası, örn. 'zyx', 'zxz'
    R          : np.ndarray, shape (3, 3)

    Döndürür
    -------
    e : np.ndarray, shape (3,)  — radyan cinsinden Euler açıları
    """
    # make_rot_mat(rx,ry,rz) = Rx*Ry*Rz = XYZ extrinsic konvansiyonu
    # scipy XYZ ile birebir eslesir, (rx,ry,rz) sırasında radyan döndürür
    rot = Rotation.from_matrix(R)
    angles = rot.as_euler("XYZ", degrees=False)
    return angles


# ===========================================================================
# TRIANGULATE — DLT ile 3D Üçgenleme
# MATLAB karşılığı: shared/triangulate.m
# ===========================================================================

def triangulate(pts1, pts2, P1, P2, K1, K2, compute_error=False):
    """
    Doğrusal yöntemle (DLT / Hartley-Zisserman) 3D nokta koordinatlarını
    hesaplar.

    MATLAB imzası:
        [mPts3D, vError] = triangulate(mPts1, mPts2, mP1, mP2, mK1, mK2, lE)

    Parametreler
    ----------
    pts1, pts2      : np.ndarray, shape (3, N)  — homojen görüntü noktaları
    P1, P2          : np.ndarray, shape (3, 4)  — kamera poz matrisleri
    K1, K2          : np.ndarray, shape (3, 3)  — kamera iç parametre matrisleri
    compute_error   : bool — True ise yeniden projeksiyon hatası hesaplanır

    Döndürür
    -------
    pts3d  : np.ndarray, shape (4, N)  — homojen 3D noktalar
    error  : np.ndarray, shape (4*N,) veya [] — yeniden projeksiyon hatası
    """
    # Orijinal noktaları sakla (hata hesabı için)
    if compute_error:
        pts1_orig = pts1.copy()
        pts2_orig = pts2.copy()

    # Görüntü noktalarını kamera koordinatlarına dönüştür
    pts1_n = np.linalg.solve(K1, pts1)   # K1^-1 * pts1
    pts2_n = np.linalg.solve(K2, pts2)   # K2^-1 * pts2

    n = pts1.shape[1]
    pts3d = np.zeros((4, n))

    for i in range(n):
        # DLT matrisi A (4x4)
        A = np.array([
            pts1_n[0, i] * P1[2, :] - P1[0, :],
            pts1_n[1, i] * P1[2, :] - P1[1, :],
            pts2_n[0, i] * P2[2, :] - P2[0, :],
            pts2_n[1, i] * P2[2, :] - P2[1, :],
        ])

        # Satırları normalize et
        norms = np.linalg.norm(A, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        A_norm = A / norms

        # En küçük tekil değere karşılık gelen tekil vektör
        _, _, Vt = np.linalg.svd(A_norm)
        pt = Vt[-1]                    # son satır = en küçük sv

        # Homojen normalleştirme
        if pt[3] != 0:
            pt = pt / pt[3]
        pts3d[:, i] = pt

    if compute_error:
        # 3D noktaları kamera 1 düzlemine yeniden projekte et
        pts1_r = K1 @ P1[:3, :] @ pts3d
        pts1_r = pts1_r / pts1_r[2:3, :]

        # 3D noktaları kamera 2 düzlemine yeniden projekte et
        pts2_r = K2 @ P2[:3, :] @ pts3d
        pts2_r = pts2_r / pts2_r[2:3, :]

        # Öklid mesafeleri
        e1 = np.sqrt((pts1_orig - pts1_r) ** 2)
        e2 = np.sqrt((pts2_orig - pts2_r) ** 2)
        error = np.concatenate([
            e1[0], e1[1], e2[0], e2[1]
        ])
        error = np.nan_to_num(error, nan=0.0)
    else:
        error = np.array([])

    return pts3d, error


# ===========================================================================
# GET FILES — Özyinelemeli Dosya Listeleme
# MATLAB karşılığı: shared/getFiles.m
# ===========================================================================

def get_files(directory, extension):
    """
    Belirtilen klasör ve alt klasörlerindeki dosyaları özyinelemeli olarak
    listeler.

    MATLAB imzası:
        cFiles = getFiles(strDir, strExt)

    Parametreler
    ----------
    directory : str  — aranacak kök klasör
    extension : str  — dosya uzantısı veya son eki, örn. 'Left.mat', '.npz'

    Döndürür
    -------
    list of str  — tam dosya yolları, sıralı
    """
    results = []
    root = Path(directory)

    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name.endswith(extension):
            results.append(str(path))

    return results


# ===========================================================================
# BASIT TEST
# ===========================================================================

if __name__ == '__main__':
    print("utils.py — Temel testler\n")

    # -------------------------------------------------------------------
    # TEST 1: make_rot_mat
    # -------------------------------------------------------------------
    print("TEST 1: make_rot_mat")
    R = make_rot_mat(0, 0, 0)
    assert np.allclose(R, np.eye(3)), "Sıfır rotasyon birim matris olmalı"

    R90 = make_rot_mat(90, 0, 0)
    expected = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    assert np.allclose(R90, expected, atol=1e-10), \
        f"90° X rotasyonu hatalı:\n{R90}"
    print(f"  Sıfır rotasyon   : birim matris ✓")
    print(f"  90° X rotasyonu  : doğru ✓")
    print(f"  TEST 1 BASARILI ✓\n")

    # -------------------------------------------------------------------
    # TEST 2: rotro2eu (gidiş-dönüş kontrolü)
    # -------------------------------------------------------------------
    print("TEST 2: rotro2eu")
    angles_in = np.array([15.0, -5.0, 30.0])   # derece
    R_test = make_rot_mat(*angles_in)
    angles_out = rotro2eu('zyx', R_test)         # radyan

    # Geri dönüşüm: rotro2eu → make_rot_mat
    R_back = make_rot_mat(
        np.degrees(angles_out[0]),
        np.degrees(angles_out[1]),
        np.degrees(angles_out[2]),
    )
    assert np.allclose(R_test, R_back, atol=1e-8), \
        "Gidiş-dönüş rotasyon matrisi uyuşmuyor"
    print(f"  Giriş (derece)   : {angles_in}")
    print(f"  Çıkış (radyan)   : {np.round(angles_out, 4)}")
    print(f"  Gidiş-dönüş hata : {np.max(np.abs(R_test - R_back)):.2e}")
    print(f"  TEST 2 BASARILI ✓\n")

    # -------------------------------------------------------------------
    # TEST 3: triangulate
    # -------------------------------------------------------------------
    print("TEST 3: triangulate")

    # Basit kamera kurulumu: iki kamera yan yana
    K = np.array([[1000, 0, 320],
                  [0, 1000, 240],
                  [0,    0,   1]], dtype=float)

    P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]], dtype=float)
    P2 = np.array([[1,0,0,-100],[0,1,0,0],[0,0,1,0]], dtype=float)

    # Gerçek 3D noktalar
    pts_true = np.array([
        [10, -5, 20],
        [ 0,  0, 50],
        [30, 10, 80],
    ], dtype=float).T   # (3, N)
    pts_true_h = np.vstack([pts_true, np.ones((1, pts_true.shape[1]))])

    # Görüntü projeksiyonları
    proj1 = K @ P1 @ pts_true_h
    proj1 /= proj1[2:3]
    proj2 = K @ P2 @ pts_true_h
    proj2 /= proj2[2:3]

    # Üçgenleme
    pts3d, error = triangulate(proj1, proj2, P1, P2, K, K,
                               compute_error=True)

    # Homojen → 3D karşılaştırma
    pts3d_xyz = pts3d[:3] / pts3d[3:4]
    max_err = np.max(np.abs(pts3d_xyz - pts_true))

    print(f"  Gerçek 3D noktalar     :\n{pts_true.T}")
    print(f"  Üçgenlenen 3D noktalar :\n{pts3d_xyz.T.round(4)}")
    print(f"  Maksimum koordinat hatası : {max_err:.6f}")
    print(f"  Yeniden projeksiyon hatası (ort): {error.mean():.6f}")
    print(f"  TEST 3 BASARILI ✓\n" if max_err < 0.01
          else f"  TEST 3 BASARISIZ ✗\n")

    # -------------------------------------------------------------------
    # TEST 4: get_files
    # -------------------------------------------------------------------
    print("TEST 4: get_files")
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test dosyaları oluştur
        (Path(tmpdir) / 'Left.npz').touch()
        (Path(tmpdir) / 'Right.npz').touch()
        sub = Path(tmpdir) / 'sub'
        sub.mkdir()
        (sub / 'Left.npz').touch()
        (sub / 'other.txt').touch()

        files = get_files(tmpdir, 'Left.npz')

    print(f"  Bulunan dosya sayısı : {len(files)}")
    print(f"  TEST 4 BASARILI ✓" if len(files) == 2
          else f"  TEST 4 BASARISIZ ✗")
