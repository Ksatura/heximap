"""
ext_read_sort.py

MATLAB kaynak dosyaları:
  - main/2_extract/extReadImage.m
  - main/2_extract/extSortImages.m
  - main/shared/fileExchange/vecRotMat/vecRotMat.m

Açıklama:
  Hexagon görüntü pencerelerini okuma ve stereo çift sıralaması.

Bağımlılıklar:
  - numpy (pip install numpy)
"""

import numpy as np


# ===========================================================================
# VEC ROT MAT — Möller-Hughes Vektör Rotasyon Matrisi
# MATLAB karşılığı: shared/fileExchange/vecRotMat/vecRotMat.m
# ===========================================================================

def vec_rot_mat(f, t):
    """
    f birim vektörünü t birim vektörüne döndüren rotasyon matrisini hesaplar.
    R(f,t) * f = t

    MATLAB imzası:
        R = vecRotMat(f, t)

    Parametreler
    ----------
    f : np.ndarray, shape (3,) veya (N, 3)  — kaynak birim vektör(ler)
    t : np.ndarray, shape (3,) veya (N, 3)  — hedef birim vektör(ler)

    Döndürür
    -------
    R : np.ndarray, shape (3, 3) veya (N, 3, 3)
    """
    f = np.atleast_2d(f).astype(float)
    t = np.atleast_2d(t).astype(float)

    # Birim vektör kontrolü
    f_norms = np.linalg.norm(f, axis=1)
    t_norms = np.linalg.norm(t, axis=1)
    if not (np.allclose(f_norms, 1.0, atol=1e-5) and
            np.allclose(t_norms, 1.0, atol=1e-5)):
        raise ValueError("Giriş vektörleri birim vektör olmalıdır.")

    n = f.shape[0]
    R_out = np.zeros((n, 3, 3))

    for i in range(n):
        fi = f[i]
        ti = t[i]
        c = np.dot(fi, ti)          # cos(açı)

        if abs(c) > 0.99:
            # Paralel veya anti-paralel durum
            # En küçük bileşen ekseni etrafında döndür
            p = np.zeros(3)
            idx_min = np.argmin(np.abs(fi))
            p[idx_min] = 1.0

            u = p - fi
            v = p - ti
            ru = -2.0 / np.dot(u, u)
            rv = -2.0 / np.dot(v, v)
            ruv = 4.0 * np.dot(u, v) / (np.dot(u, u) * np.dot(v, v))

            R_out[i] = (np.eye(3)
                        + ru * np.outer(u, u)
                        + rv * np.outer(v, v)
                        + ruv * np.outer(v, u))
        else:
            # Genel durum — Möller-Hughes formülü
            vv = np.cross(fi, ti)
            h = (1.0 - c) / np.dot(vv, vv)
            R_out[i] = np.array([
                [c + h*vv[0]**2,          h*vv[0]*vv[1] - vv[2],  h*vv[0]*vv[2] + vv[1]],
                [h*vv[0]*vv[1] + vv[2],   c + h*vv[1]**2,          h*vv[1]*vv[2] - vv[0]],
                [h*vv[0]*vv[2] - vv[1],   h*vv[1]*vv[2] + vv[0],   c + h*vv[2]**2       ],
            ])

    return R_out[0] if n == 1 else R_out


# ===========================================================================
# EXT READ IMAGE
# MATLAB karşılığı: extReadImage.m
# ===========================================================================

def ext_read_image(mat_obj, win_obj):
    """
    Hexagon görüntü penceresini mat dosyasından okur.

    MATLAB imzası:
        extReadImage(objM, objW, hW, cWin)

    Parametreler
    ----------
    mat_obj : dict
        Tam Hexagon görüntüsünü içeren nesne.
        Beklenen anahtar: 'Image' (np.ndarray)

    win_obj : dict
        Pencere bilgilerini içeren nesne.
        Beklenen anahtar: 'Window' — shape (2,2)
            [[x_min, y_min], [x_max, y_max]]

    Döndürür
    -------
    win_obj : dict  — 'Image' anahtarı eklenerek güncellenir
    """
    win = win_obj['Window']

    # MATLAB'da: objM.Image(mWin(3):mWin(4), mWin(1):mWin(2))
    # MATLAB 1-tabanlı, Python 0-tabanlı — 1 çıkar
    row_start = int(win[0, 1]) - 1
    row_end   = int(win[1, 1])      # Python slice: end dahil değil → -1+1 = 0
    col_start = int(win[0, 0]) - 1
    col_end   = int(win[1, 0])

    win_obj['Image'] = mat_obj['Image'][row_start:row_end,
                                        col_start:col_end].copy()
    return win_obj


# ===========================================================================
# EXT SORT IMAGES
# MATLAB karşılığı: extSortImages.m
# ===========================================================================

def ext_sort_images(hex_files, mat_objects):
    """
    Hexagon görüntülerini çekilme sırasına göre sıralar.
    Sol/sağ stereo çiftini belirlemek için kullanılır.

    MATLAB imzası:
        [cHexFile, cM] = extSortImages(cHexFile, cM, hW)

    Parametreler
    ----------
    hex_files   : list of str   — Hexagon dosya yolları
    mat_objects : list of dict  — Her görüntü için metadata nesnesi.
        Her dict şu anahtarları içermelidir:
            'Image'       : np.ndarray
            'SpatialTrans': dict  — uzamsal dönüşüm bilgisi
                {'A': np.ndarray (3,3)}  — affin dönüşüm matrisi

    Döndürür
    -------
    hex_files_sorted   : list of str
    mat_objects_sorted : list of dict
    """
    n = len(mat_objects)
    centers = np.zeros((n, 2))
    directions = np.zeros((n, 2))

    for i, obj in enumerate(mat_objects):
        img = obj['Image']
        h, w = img.shape[:2]
        T = obj['SpatialTrans']   # affin dönüşüm matrisi (3x3 homojen)

        # Görüntü köşeleri (1-tabanlı, MATLAB uyumlu)
        # y ekseni işareti ters
        corners_img = np.array([
            [1,  -1,  1],
            [w,  -h,  1],
            [w,  -1,  1],
            [1,  -h,  1],
        ], dtype=float)

        # Dünya koordinatlarına dönüştür
        corners_wld = (T @ corners_img.T).T   # (4, 3)
        corners_wld = corners_wld[:, :2]       # x, y al

        # Merkez
        centers[i] = corners_wld.mean(axis=0)

        # Köşe 1'den köşe 3'e yön vektörü
        directions[i] = corners_wld[2] - corners_wld[0]

    # Yön vektörlerinin ortalamasını al, 3D'ye genişlet
    mean_dir_3d = np.mean(
        np.hstack([directions, np.zeros((n, 1))]), axis=0
    )
    mean_dir_3d /= np.linalg.norm(mean_dir_3d)

    # x eksenine hizalayan rotasyon matrisini hesapla
    x_axis = np.array([1.0, 0.0, 0.0])
    R = vec_rot_mat(mean_dir_3d, x_axis)

    # Merkezleri döndür
    centers_3d = np.hstack([centers, np.zeros((n, 1))])
    centers_rot = (R @ centers_3d.T).T   # (n, 3)

    # x koordinatına göre sırala
    order = np.argsort(centers_rot[:, 0])

    hex_files_sorted   = [hex_files[i]   for i in order]
    mat_objects_sorted = [mat_objects[i] for i in order]

    return hex_files_sorted, mat_objects_sorted


# ===========================================================================
# BASIT TEST
# ===========================================================================

if __name__ == '__main__':
    print("ext_read_sort.py — Temel testler\n")

    # -------------------------------------------------------------------
    # TEST 1: vec_rot_mat
    # -------------------------------------------------------------------
    print("TEST 1: vec_rot_mat")

    # x eksenini y eksenine döndür
    f = np.array([1.0, 0.0, 0.0])
    t = np.array([0.0, 1.0, 0.0])
    R = vec_rot_mat(f, t)
    result = R @ f
    err = np.max(np.abs(result - t))
    print(f"  R * f = {np.round(result, 6)}, beklenen {t}")
    print(f"  Hata  : {err:.2e}")
    print(f"  TEST 1 BASARILI ✓\n" if err < 1e-10
          else f"  TEST 1 BASARISIZ ✗\n")

    # Paralel durum (aynı vektör)
    f2 = np.array([0.0, 0.0, 1.0])
    t2 = np.array([0.0, 0.0, 1.0])
    R2 = vec_rot_mat(f2, t2)
    err2 = np.max(np.abs(R2 @ f2 - t2))
    print(f"  Paralel durum hatası: {err2:.2e}")
    print(f"  TEST 1b BASARILI ✓\n" if err2 < 1e-10
          else f"  TEST 1b BASARISIZ ✗\n")

    # -------------------------------------------------------------------
    # TEST 2: ext_read_image
    # -------------------------------------------------------------------
    print("TEST 2: ext_read_image")

    fake_image = np.arange(100 * 200, dtype=np.uint8).reshape(100, 200)
    mat_obj = {'Image': fake_image}
    win_obj = {
        'Window': np.array([[10, 20], [50, 80]])   # [x_min,y_min],[x_max,y_max]
    }

    win_obj = ext_read_image(mat_obj, win_obj)
    expected_shape = (80 - 20, 50 - 10)   # (60, 40)
    assert 'Image' in win_obj, "'Image' anahtarı eklenmeli"
    assert win_obj['Image'].shape == expected_shape, \
        f"Beklenen {expected_shape}, alınan {win_obj['Image'].shape}"
    print(f"  Pencere boyutu : {win_obj['Image'].shape}, beklenen {expected_shape}")
    print(f"  TEST 2 BASARILI ✓\n")

    # -------------------------------------------------------------------
    # TEST 3: ext_sort_images
    # -------------------------------------------------------------------
    print("TEST 3: ext_sort_images")

    # İki sahte görüntü — farklı konumlarda
    def make_obj(x_offset):
        img = np.zeros((100, 200), dtype=np.uint8)
        # Basit öteleme dönüşümü: x_offset kadar sağda
        T = np.array([
            [1.0, 0.0, x_offset],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        return {'Image': img, 'SpatialTrans': T}

    # İkinci görüntü (x=500) solda, birinci (x=1000) sağda listelenmiş
    files = ['img_right.npz', 'img_left.npz']
    objs  = [make_obj(1000), make_obj(500)]

    files_s, objs_s = ext_sort_images(files, objs)

    print(f"  Sıralama öncesi: {files}")
    print(f"  Sıralama sonrası: {files_s}")
    correct = files_s[0] == 'img_left.npz'
    print(f"  TEST 3 BASARILI ✓" if correct
          else f"  TEST 3 BASARISIZ ✗")
