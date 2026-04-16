"""
ext_filter_images.py

MATLAB kaynak dosyası:
  - main/2_extract/extFilterImages.m

Açıklama:
  Stereo görüntü çiftine histogram eşleştirme, adaptif histogram
  eşitleme ve Wiener gürültü filtresi uygular.

Bağımlılıklar:
  - numpy        (pip install numpy)
  - scikit-image (pip install scikit-image)
  - scipy        (pip install scipy)
"""

import numpy as np
from skimage.exposure import match_histograms, equalize_adapthist
from scipy.signal import wiener


# ===========================================================================
# ANA FONKSİYON
# MATLAB karşılığı: extFilterImages()
# ===========================================================================

def ext_filter_images(img1, img2, empty_val=0, opt=None):
    """
    Stereo görüntü çiftine ön işleme filtreleri uygular.

    MATLAB imzası:
        [mI1, mI2] = extFilterImages(mI1, mI2, iE, sOpt)

    Parametreler
    ----------
    img1, img2  : np.ndarray  — gri tonlamalı görüntüler (uint8 veya float)
    empty_val   : int/float   — boş piksel değeri (MATLAB'da iE, genellikle 0)
    opt         : dict veya None
        'histmatch'  : bool — histogram eşleştirme (varsayılan True)
        'adapthisteq': bool — adaptif histogram eşitleme (varsayılan True)
        'wiener2'    : bool — Wiener gürültü filtresi (varsayılan True)

    Döndürür
    -------
    img1_out, img2_out : np.ndarray  — filtrelenmiş görüntüler (float64, [0,1])
    """
    if opt is None:
        opt = {'histmatch': True, 'adapthisteq': True, 'wiener2': True}

    # Float64'e çevir ve [0,1] aralığına normalize et
    img1 = _to_float(img1)
    img2 = _to_float(img2)

    # Boş piksel maskeleri
    mask1 = img1 != _to_float_val(empty_val, img1)
    mask2 = img2 != _to_float_val(empty_val, img2)

    # -----------------------------------------------------------------
    # 1. Histogram eşleştirme: img2 → img1 histogramına uydur
    # MATLAB: imhistmatch(mI2(lM2), mI1(lM1), 256)
    # -----------------------------------------------------------------
    if opt.get('histmatch', True):
        if mask1.any() and mask2.any():
            # Sadece geçerli pikseller üzerinde eşleştir
            vals2_matched = _histmatch_masked(
                img2[mask2], img1[mask1], nbins=256
            )
            img2[mask2] = vals2_matched

    # -----------------------------------------------------------------
    # 2. Adaptif histogram eşitleme (CLAHE)
    # MATLAB: adapthisteq(mI, 'ClipLimit', 0.03, 'NumTiles', [20 20])
    # -----------------------------------------------------------------
    if opt.get('adapthisteq', True):
        img1 = equalize_adapthist(img1, nbins=256,
                                  clip_limit=0.03,
                                  kernel_size=(img1.shape[0] // 20,
                                               img1.shape[1] // 20))
        img2 = equalize_adapthist(img2, nbins=256,
                                  clip_limit=0.03,
                                  kernel_size=(img2.shape[0] // 20,
                                               img2.shape[1] // 20))

    # -----------------------------------------------------------------
    # 3. Wiener gürültü filtresi
    # MATLAB: wiener2(mI, [3 3])
    # -----------------------------------------------------------------
    if opt.get('wiener2', True):
        img1 = wiener(img1, mysize=(3, 3))
        img2 = wiener(img2, mysize=(3, 3))
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)

    # Boş pikselleri orijinal değerlerine geri yükle
    empty_f = _to_float_val(empty_val, np.zeros(1, dtype=np.float64))
    img1[~mask1] = empty_f
    img2[~mask2] = empty_f

    return img1, img2


# ===========================================================================
# YARDIMCI FONKSİYONLAR
# ===========================================================================

def _to_float(img):
    """Görüntüyü [0,1] aralığında float64'e çevirir."""
    img = np.array(img, dtype=np.float64)
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _to_float_val(val, reference):
    """Boş piksel değerini referans görüntünün ölçeğine göre normalize eder."""
    if isinstance(reference, np.ndarray) and reference.max() <= 1.0:
        return float(val) / 255.0 if float(val) > 1.0 else float(val)
    return float(val)


def _histmatch_masked(src_vals, ref_vals, nbins=256):
    """
    1D piksel değeri dizisini referans dağılımına eşleştirir.
    MATLAB imhistmatch'in piksel maskeli versiyonu.

    Parametreler
    ----------
    src_vals : np.ndarray, shape (N,)  — kaynak piksel değerleri
    ref_vals : np.ndarray, shape (M,)  — referans piksel değerleri
    nbins    : int

    Döndürür
    -------
    np.ndarray, shape (N,)  — eşleştirilmiş değerler
    """
    # Referans histogramı
    ref_hist, bin_edges = np.histogram(ref_vals, bins=nbins, range=(0, 1))
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    ref_cdf /= ref_cdf[-1]

    # Kaynak histogramı
    src_hist, _ = np.histogram(src_vals, bins=nbins, range=(0, 1))
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    src_cdf /= src_cdf[-1]

    # CDF eşleştirme
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    matched = np.interp(
        np.interp(src_vals, bin_centers, src_cdf),
        ref_cdf, bin_centers
    )
    return matched


# ===========================================================================
# BASIT TEST
# ===========================================================================

if __name__ == '__main__':
    print("ext_filter_images.py — Temel testler\n")

    rng = np.random.default_rng(42)

    # Farklı parlaklık dağılımlarına sahip sentetik görüntüler
    img1 = (rng.random((200, 200)) * 200 + 30).astype(np.uint8)
    img2 = (rng.random((200, 200)) * 100).astype(np.uint8)

    # Boş piksel bölgeleri ekle
    img1[:10, :] = 0
    img2[:10, :] = 0

    # -------------------------------------------------------------------
    # TEST 1: Tüm filtreler açık
    # -------------------------------------------------------------------
    print("TEST 1: Tüm filtreler açık")
    opt = {'histmatch': True, 'adapthisteq': True, 'wiener2': True}
    out1, out2 = ext_filter_images(img1.copy(), img2.copy(), empty_val=0,
                                   opt=opt)
    assert out1.shape == img1.shape, "Çıktı boyutu değişmemeli"
    assert out2.shape == img2.shape, "Çıktı boyutu değişmemeli"
    assert out1.dtype == np.float64, "Çıktı float64 olmalı"
    assert out1.max() <= 1.0 and out1.min() >= 0.0, "Değerler [0,1] içinde olmalı"
    print(f"  Giriş img1 ort   : {img1.mean():.1f}  →  çıkış: {out1.mean():.4f}")
    print(f"  Giriş img2 ort   : {img2.mean():.1f}  →  çıkış: {out2.mean():.4f}")
    print(f"  Çıkış boyutları  : {out1.shape}, {out2.shape}")
    print(f"  Boş piksel korundu: {out1[:10,:].mean() == 0.0}")
    print(f"  TEST 1 BASARILI ✓\n")

    # -------------------------------------------------------------------
    # TEST 2: Yalnızca histogram eşleştirme
    # -------------------------------------------------------------------
    print("TEST 2: Yalnızca histogram eşleştirme")
    opt2 = {'histmatch': True, 'adapthisteq': False, 'wiener2': False}
    out1b, out2b = ext_filter_images(img1.copy(), img2.copy(),
                                     empty_val=0, opt=opt2)
    # img2'nin ortalaması img1'e yaklaşmış olmalı
    diff_before = abs(img1.mean() - img2.mean())
    diff_after  = abs(out1b.mean() - out2b.mean()) * 255
    print(f"  Eşleştirme öncesi fark : {diff_before:.1f}")
    print(f"  Eşleştirme sonrası fark: {diff_after:.1f}")
    print(f"  TEST 2 BASARILI ✓\n" if diff_after < diff_before
          else f"  TEST 2 BASARISIZ ✗\n")

    # -------------------------------------------------------------------
    # TEST 3: Tüm filtreler kapalı (passthrough)
    # -------------------------------------------------------------------
    print("TEST 3: Tüm filtreler kapalı (passthrough)")
    opt3 = {'histmatch': False, 'adapthisteq': False, 'wiener2': False}
    out1c, out2c = ext_filter_images(img1.copy(), img2.copy(),
                                     empty_val=0, opt=opt3)
    expected1 = img1.astype(np.float64) / 255.0
    assert np.allclose(out1c, expected1, atol=1e-6), \
        "Filtre kapalıyken görüntü değişmemeli"
    print(f"  Maksimum sapma: {np.max(np.abs(out1c - expected1)):.2e}")
    print(f"  TEST 3 BASARILI ✓")
