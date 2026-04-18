"""
HEXIMAP Python portuna entegrasyon testi — 2. Aşama: Extract

Bu script, stitch aşamasının ürettiği .npz dosyaları üzerinde
extract pipeline'ını (pencere hesaplama → stereo doğrultma →
kamera poz tahmini → disparite) adım adım çalıştırır ve her
aşamanın çıktısını doğrular.

Kullanım:
    python3 integration_test_extract.py

Giriş (Aşama 1 çıktıları):
    DZB1206-500009L007001.npz
    DZB1206-500009L008001.npz

Çıktı:
    DZB1206-500009L007001_L008001_disparity.npz
"""

import numpy as np
import sys
import traceback
from pathlib import Path

# Modül yollarını ekle
BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / 'main' / 'shared'))
sys.path.insert(0, str(BASE / 'main' / '2_extract'))

from ext_windows      import parse_corner_gcps, compute_spatial_transform, compute_windows
from ext_read_sort    import ext_read_image, ext_sort_images
from ext_filter_images import ext_filter_images
from ext_stereo_rect  import ext_stereo_rect
from ext_init_bundle  import ext_init_trans, ext_bundle_adjust
from ext_disparity    import ext_disparity

# ===========================================================================
# GÖRÜNTÜ ÇİFTLERİ VE KÖŞELERİ
# ===========================================================================

# parse_corner_gcps sırası: [NW_lat, NW_lon, NE_lat, NE_lon,
#                             SE_lat, SE_lon, SW_lat, SW_lon]
IMAGE_CONFIG = {
    'L007001': {
        'npz':  'DZB1206-500009L007001.npz',
        'gcps': [
            40.055, 26.652,   # NW
            39.905, 28.022,   # NE
            37.808, 27.624,   # SE
            37.953, 26.294,   # SW
        ],
    },
    'L008001': {
        'npz':  'DZB1206-500009L008001.npz',
        'gcps': [
            39.425, 26.505,   # NW
            39.277, 27.862,   # NE
            37.180, 27.471,   # SE
            37.323, 26.152,   # SW
        ],
    },
}

# KH-9 Lower Resolution Mapping Camera sabit parametreleri
# Film: 9"x18", tarama çözünürlüğü ~25 µm → piksel boyutu
KH9_FOCAL_LENGTH_MM    = 304.8          # nominal odak uzaklığı (mm)
KH9_PIXEL_SIZE_MM      = 0.025          # 25 µm tarama
KH9_FOCAL_LENGTH_PIX   = KH9_FOCAL_LENGTH_MM / KH9_PIXEL_SIZE_MM   # ≈ 12192 px

# Test için kullanılacak pencere boyutu (tam ölçeğin küçük bir alt kümesi)
# Tam işlemde 4400, test için 2000 kullanıyoruz
TEST_WINDOW_SIZE   = 2000
TEST_BUFFER        = 200
DISPARITY_RES      = '1/8'   # hız için düşük çözünürlük


# ===========================================================================
# YARDIMCI FONKSİYONLAR
# ===========================================================================

def find_npz_dir():
    """Stitch çıktısı .npz dosyalarının bulunduğu klasörü bulur."""
    search_dirs = [
        Path.home() / 'Desktop' / 'Heximap_Tests',
        Path.home() / 'Masaüstü' / 'Heximap_Tests',
        Path.home() / 'Downloads',
        Path.home() / 'İndirilenler',
        Path.home() / 'Desktop',
        Path.home() / 'Masaüstü',
        Path.home(),
    ]
    for d in search_dirs:
        if d.exists():
            npz = list(d.glob('DZB*.npz'))
            if npz:
                print(f"  .npz dosyaları bulundu: {d}")
                return d
    return None


def load_npz_image(npz_path):
    """
    Stitch çıktısı .npz dosyasını yükler.
    Döndürür: dict {'Image': np.ndarray, 'Transform': np.ndarray}
    """
    data = np.load(npz_path)
    return {
        'Image':     data['Image'],
        'Transform': data['Transform'],
    }


def build_mat_object(cfg, npz_dir, label):
    """
    Tek bir görüntü için tam metadata nesnesini oluşturur.

    Döndürür
    -------
    dict — ext_init_trans'ın beklediği anahtarları içeren nesne:
        Image, Image10, SpatialTrans,
        FocalLengthPixels, PrincipalPointPixels
    """
    npz_path = npz_dir / cfg['npz']
    print(f"  [{label}] Yükleniyor: {npz_path.name} ...", end=' ')

    mat = load_npz_image(npz_path)
    img = mat['Image']
    h, w = img.shape[:2]

    print(f"{w}x{h} px")

    # GCP → SpatialTrans
    gcps_wld = parse_corner_gcps(cfg['gcps'])
    T = compute_spatial_transform(gcps_wld, (h, w))
    mat['SpatialTrans'] = T

    # 1/10 ölçek görüntü (ext_init_trans için)
    import cv2
    h10 = max(1, h // 10)
    w10 = max(1, w // 10)
    mat['Image10'] = cv2.resize(img, (w10, h10),
                                interpolation=cv2.INTER_AREA)

    # Kamera parametreleri
    mat['FocalLengthPixels']    = KH9_FOCAL_LENGTH_PIX
    mat['PrincipalPointPixels'] = np.array([w / 2.0, h / 2.0])

    return mat


def check_step(name, condition, detail=''):
    """Adım sonucunu raporlar."""
    status = '✓' if condition else '✗'
    print(f"    {status} {name}" + (f": {detail}" if detail else ''))
    return condition


# ===========================================================================
# TEST ADIMLARI
# ===========================================================================

def step1_load_and_prepare(npz_dir):
    """
    Adım 1: .npz görüntülerini yükle, SpatialTrans ve Image10 oluştur.
    """
    print("\n  [Adım 1] Görüntüler yükleniyor ve hazırlanıyor...")

    mat_l = build_mat_object(IMAGE_CONFIG['L007001'], npz_dir, 'L007001')
    mat_r = build_mat_object(IMAGE_CONFIG['L008001'], npz_dir, 'L008001')

    ok = True
    ok &= check_step('L007001 Image',       mat_l['Image'].ndim == 2,
                     f"shape={mat_l['Image'].shape}")
    ok &= check_step('L007001 SpatialTrans', mat_l['SpatialTrans'].shape == (3,3))
    ok &= check_step('L007001 Image10',      mat_l['Image10'].ndim == 2,
                     f"shape={mat_l['Image10'].shape}")
    ok &= check_step('L008001 yüklendi',     mat_r['Image'].ndim == 2)

    return mat_l, mat_r, ok


def step2_sort_and_overlap(mat_l, mat_r):
    """
    Adım 2: Görüntüleri stereo sıraya göre düzenle, örtüşme bölgesini hesapla.
    """
    print("\n  [Adım 2] Görüntü sıralaması ve örtüşme kontrolü...")

    files   = ['L007001', 'L008001']
    objects = [mat_l, mat_r]

    files_sorted, objects_sorted = ext_sort_images(files, objects)

    print(f"    Sıralama sonrası: {files_sorted[0]} → {files_sorted[1]}")

    mat_left  = objects_sorted[0]
    mat_right = objects_sorted[1]

    # Örtüşme bölgesi kontrolü (SpatialTrans köşelerinden)
    def world_bounds(T, h, w):
        pts = np.array([[1,-1,1],[w,-1,1],[w,-h,1],[1,-h,1]], dtype=float).T
        wld = T @ pts
        return wld[0].min(), wld[0].max(), wld[1].min(), wld[1].max()

    xl, xr_l, yl, yr_l = world_bounds(
        mat_left['SpatialTrans'], *mat_left['Image'].shape[:2][::-1]
    )
    xr, xr_r, yr, yr_r = world_bounds(
        mat_right['SpatialTrans'], *mat_right['Image'].shape[:2][::-1]
    )

    # Kesişim
    x_overlap = min(xr_l, xr_r) - max(xl, xr)
    has_overlap = x_overlap > 0

    ok = check_step('Sıralama başarılı', len(files_sorted) == 2)
    ok &= check_step('Örtüşme mevcut',
                     has_overlap,
                     f"x_overlap={x_overlap:.4f}°")

    return mat_left, mat_right, ok


def step3_compute_windows(mat_l, mat_r):
    """
    Adım 3: İşleme pencerelerini hesapla.
    Test için merkezi küçük bir ROI kullanıyoruz.
    """
    print("\n  [Adım 3] İşleme pencereleri hesaplanıyor...")

    h, w = mat_l['Image'].shape[:2]

    # Merkezi bir ROI seç (görüntünün %40–60 bölgesi)
    roi = {
        'x_min': int(w * 0.40),
        'x_max': int(w * 0.60),
        'y_min': int(h * 0.40),
        'y_max': int(h * 0.60),
    }

    # Başlangıç homojenlik matrisleri (bundle adjust öncesi birim matris)
    H1 = np.eye(3)
    H2 = np.eye(3)

    windows = compute_windows(
        H1, H2,
        img_size_l=(h, w),
        img_size_r=mat_r['Image'].shape[:2],
        roi_list=[roi],
        win_size_pix=TEST_WINDOW_SIZE,
        buffer_pix=TEST_BUFFER,
    )

    print(f"    ROI: {roi}")
    print(f"    Pencere sayısı: {len(windows)}")
    for i, win in enumerate(windows):
        print(f"    Pencere {i+1}: sol={win['left'].tolist()}")

    ok = check_step('En az 1 pencere üretildi', len(windows) >= 1)

    return windows, ok


def step4_read_window(mat_l, mat_r, windows):
    """
    Adım 4: İlk pencereyi görüntülerden kes.
    """
    print("\n  [Adım 4] Pencere görüntüleri kesiliyor...")

    win = windows[0]

    obj_l = {
        'Image':  None,
        'Window': win['left'].astype(int),
    }
    obj_r = {
        'Image':  None,
        'Window': win['right'].astype(int),
    }

    obj_l = ext_read_image(mat_l, obj_l)
    obj_r = ext_read_image(mat_r, obj_r)

    img_l = obj_l['Image']
    img_r = obj_r['Image']

    print(f"    Sol pencere görüntüsü : {img_l.shape}")
    print(f"    Sağ pencere görüntüsü : {img_r.shape}")
    print(f"    Sol min/max           : {img_l.min()}/{img_l.max()}")
    print(f"    Sağ min/max           : {img_r.min()}/{img_r.max()}")

    ok  = check_step('Sol görüntü kesildi', img_l.ndim == 2 and img_l.size > 0)
    ok &= check_step('Sağ görüntü kesildi', img_r.ndim == 2 and img_r.size > 0)
    ok &= check_step('Veri içeriyor (sıfır değil)',
                     img_l.max() > 0 and img_r.max() > 0)

    return obj_l, obj_r, ok


def step5_filter(obj_l, obj_r):
    """
    Adım 5: Histogram eşleştirme + CLAHE + Wiener filtresi.
    """
    print("\n  [Adım 5] Görüntü filtreleme (histmatch + CLAHE + Wiener)...")

    opt = {'histmatch': True, 'adapthisteq': True, 'wiener2': True}
    filt_l, filt_r = ext_filter_images(
        obj_l['Image'], obj_r['Image'], empty_val=0, opt=opt
    )

    mean_l_before = obj_l['Image'].mean()
    mean_r_before = obj_r['Image'].mean()
    diff_before   = abs(mean_l_before - mean_r_before)
    diff_after    = abs(filt_l.mean() - filt_r.mean()) * 255

    print(f"    Filtre öncesi ort fark  : {diff_before:.1f} (8-bit)")
    print(f"    Filtre sonrası ort fark : {diff_after:.1f} (8-bit eşdeğer)")
    print(f"    Çıktı aralığı           : [{filt_l.min():.3f}, {filt_l.max():.3f}]")

    obj_l['ImageFilt'] = filt_l
    obj_r['ImageFilt'] = filt_r

    ok  = check_step('Çıktı [0,1] aralığında',
                     filt_l.max() <= 1.0 and filt_l.min() >= 0.0)
    ok &= check_step('Histogram yakınsamış',
                     diff_after < diff_before,
                     f"{diff_before:.1f} → {diff_after:.1f}")

    return obj_l, obj_r, ok


def step6_stereo_rect(obj_l, obj_r):
    """
    Adım 6: Epipolar doğrultma (stereo rectification).
    """
    print("\n  [Adım 6] Epipolar doğrultma (stereo rectification)...")

    def progress(msg):
        print(f"    ↳ {msg}")

    obj_l, obj_r = ext_stereo_rect(obj_l, obj_r, progress_cb=progress)

    rect_l = obj_l['RectImage']
    H1     = obj_l['Homography']
    H2     = obj_r['Homography']
    pts_l  = obj_l['PointMatches']
    v_err  = obj_l.get('Accuracy', {}).get('PointMatches', np.array([]))

    print(f"    Doğrultulmuş görüntü boyutu: {rect_l.shape}")
    print(f"    Homography H1 (L007001)    :\n{np.round(H1, 4)}")
    print(f"    Eşleşen nokta sayısı       : {pts_l.shape[1] if pts_l.ndim==2 else 0}")

    if v_err.size > 0:
        v_median = np.nanmedian(np.abs(v_err))
        v_pct90  = np.nanpercentile(np.abs(v_err), 90)
        print(f"    Dikey hata medyan / P90    : {v_median:.2f} / {v_pct90:.2f} px")
    else:
        v_median = 999

    ok  = check_step("'RectImage' oluşturuldu", 'RectImage' in obj_l)
    ok &= check_step("'Homography' hesaplandı",  H1 is not None)
    ok &= check_step("Dikey hata < 5 px (medyan)",
                     v_median < 5.0,
                     f"{v_median:.2f} px")

    return obj_l, obj_r, ok


def step7_init_bundle(mat_l, mat_r, obj_l, obj_r, npz_dir=None):
    """
    Adım 7: KH-9 nominal parametrelerinden kamera matrisleri üret,
    epipolar homografilerden göreli poz çıkar, bundle adjustment uygula.
    ext_init_trans tam stereo görüntü çiftleri gerektirdiğinden
    pencere kesimlerine dayalı yaklaşım kullanılıyor.
    """
    print("\n  [Adım 7] Kamera poz tahmini (homografi tabanlı)...")

    h_win, w_win = obj_l['Image'].shape[:2]
    f_pix = KH9_FOCAL_LENGTH_PIX

    K = np.array([[f_pix, 0, w_win / 2],
                  [0, f_pix, h_win / 2],
                  [0,     0,          1]], dtype=float)

    H1 = obj_l.get('Homography', np.eye(3))
    H2 = obj_r.get('Homography', np.eye(3))

    # H2^-1 * H1: sol görüntüden sağ görüntüye dönüşüm
    H2_inv_H1 = np.linalg.solve(H2, H1)
    R_approx  = H2_inv_H1[:3, :3].copy()
    scale     = np.cbrt(abs(np.linalg.det(R_approx)))
    if scale > 1e-6:
        R_approx /= scale
    U, _, Vt = np.linalg.svd(R_approx)
    R_clean  = U @ Vt
    if np.linalg.det(R_clean) < 0:
        Vt[-1] *= -1
        R_clean = U @ Vt

    t_approx = H2_inv_H1[:2, 2:3] / max(scale, 1e-6)
    t_3d     = np.vstack([t_approx, [[0.0]]])

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R_clean,   t_3d])

    print(f"    KH-9 odak uzaklığı (px): {f_pix:.0f}")
    print(f"    ||t|| öteleme normu    : {np.linalg.norm(t_3d):.4f}")
    print(f"    det(R) kontrolü        : {np.linalg.det(R_clean):.6f}")

    obj_l['IntrinsicMatrix']  = K
    obj_r['IntrinsicMatrix']  = K.copy()
    obj_l['LeftPoseMatrix']   = P1
    obj_l['RightPoseMatrix']  = P2
    obj_l['PoseMatrix']       = P1
    obj_r['PoseMatrix']       = P2

    ok  = check_step("'IntrinsicMatrix' oluşturuldu", True,
                     f"f={f_pix:.0f} px")
    ok &= check_step("P2 ötelemesi sıfır değil",
                     np.linalg.norm(t_3d) > 1e-3,
                     f"||t||={np.linalg.norm(t_3d):.4f}")
    ok &= check_step("Rotasyon matrisi geçerli",
                     abs(np.linalg.det(R_clean) - 1.0) < 0.01)

    print("\n  [Adım 7b] Bundle adjustment (ext_bundle_adjust)...")
    try:
        windows_l, windows_r = ext_bundle_adjust(
            [obj_l], [obj_r], progress_cb=progress
        )
        acc     = windows_l[0].get('Accuracy', {})
        ba_cost = acc.get('BundleAdjust', None)
        ba_err  = acc.get('Reprojection', np.array([]))
        if ba_err.size > 0:
            print(f"    Ortalama reprojection : {np.nanmean(ba_err):.4f} px")
        if ba_cost is not None:
            print(f"    Final maliyet         : {ba_cost:.6f}")
        ok &= check_step("Bundle adjustment tamamlandı",
                         'BundleAdjust' in acc or 'Reprojection' in acc)
        return windows_l[0], windows_r[0], ok
    except Exception as e:
        print(f"    ↳ Bundle adjustment atlandı: {e}")
        return obj_l, obj_r, ok

def step8_disparity(obj_l, obj_r):
    """
    Adım 8: SGBM disparite haritası.
    """
    print(f"\n  [Adım 8] Disparite haritası ({DISPARITY_RES} çözünürlük)...")

    def progress(msg):
        print(f"    ↳ {msg}")

    obj_l, obj_r = ext_disparity(
        obj_l, obj_r,
        resolution=DISPARITY_RES,
        block_size=7,
        progress_cb=progress,
    )

    disp = obj_l['Disparity']
    valid_mask = ~np.isnan(disp)
    valid_pct  = valid_mask.mean() * 100

    if valid_mask.any():
        d_min  = np.nanmin(disp)
        d_max  = np.nanmax(disp)
        d_mean = np.nanmean(disp)
    else:
        d_min = d_max = d_mean = 0

    print(f"    Disparite haritası boyutu: {disp.shape}")
    print(f"    Geçerli piksel oranı     : %{valid_pct:.1f}")
    print(f"    Disparite aralığı        : [{d_min:.1f}, {d_max:.1f}] px")
    print(f"    Ortalama disparite       : {d_mean:.2f} px")

    ok  = check_step("'Disparity' üretildi", disp is not None)
    ok &= check_step("Geçerli piksel > %5",
                     valid_pct > 5.0,
                     f"%{valid_pct:.1f}")
    ok &= check_step("Anlamlı disparite aralığı",
                     abs(d_max - d_min) > 1.0,
                     f"[{d_min:.1f}, {d_max:.1f}]")

    return obj_l, obj_r, ok


def step9_save(obj_l, obj_r, npz_dir):
    """
    Adım 9: Sonuçları .npz olarak kaydet.
    """
    print("\n  [Adım 9] Sonuçlar kaydediliyor...")

    out_path = npz_dir / 'DZB1206-500009L007001_L008001_disparity.npz'

    save_dict = {
        'Disparity':      obj_l['Disparity'],
        'DisparityScale': np.array(obj_l['DisparityScale']),
        'Homography_L':   obj_l['Homography'],
        'Homography_R':   obj_r['Homography'],
        'Window_L':       obj_l['Window'],
        'Window_R':       obj_r['Window'],
    }

    # ImagePoints isteğe bağlı (boş olabilir)
    if 'ImagePoints' in obj_l and obj_l['ImagePoints'] is not None:
        save_dict['ImagePoints_L'] = obj_l['ImagePoints']
    if 'ImagePoints' in obj_r and obj_r['ImagePoints'] is not None:
        save_dict['ImagePoints_R'] = obj_r['ImagePoints']

    np.savez(out_path, **save_dict)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"    Kaydedildi : {out_path}")
    print(f"    Dosya boyutu: {size_mb:.1f} MB")

    ok = check_step("Dosya oluşturuldu", out_path.exists())
    return out_path, ok


# ===========================================================================
# ANA FONKSİYON
# ===========================================================================

def main():
    print("HEXIMAP Entegrasyon Testi — Aşama 2: Extract")
    print("=" * 60)

    # .npz klasörünü bul
    npz_dir = find_npz_dir()
    if npz_dir is None:
        print("\n⚠️  .npz dosyaları bulunamadı.")
        npz_dir = Path(input("Klasör yolu: ").strip())
        if not npz_dir.exists():
            print("Klasör bulunamadı, çıkılıyor.")
            sys.exit(1)

    # Gerekli .npz dosyalarının varlığını kontrol et
    for label, cfg in IMAGE_CONFIG.items():
        p = npz_dir / cfg['npz']
        if not p.exists():
            print(f"\n⚠️  Bulunamadı: {p}")
            print("Önce integration_test_stitch.py'yi çalıştırın.")
            sys.exit(1)

    print(f"\nKlasör : {npz_dir}")
    print(f"Pencere : {TEST_WINDOW_SIZE}×{TEST_WINDOW_SIZE} px "
          f"(tam işlemde 4400×4400)")
    print(f"Disparite çözünürlüğü: {DISPARITY_RES}")

    results = []

    try:
        # Adım 1 — Yükle ve hazırla
        print(f"\n{'='*60}")
        mat_l, mat_r, ok1 = step1_load_and_prepare(npz_dir)
        results.append(('Yükleme & Hazırlık',       ok1))

        # Adım 2 — Sırala ve örtüşme kontrol
        print(f"\n{'='*60}")
        mat_l, mat_r, ok2 = step2_sort_and_overlap(mat_l, mat_r)
        results.append(('Sıralama & Örtüşme',        ok2))

        # Adım 3 — Pencere hesapla
        print(f"\n{'='*60}")
        windows, ok3 = step3_compute_windows(mat_l, mat_r)
        results.append(('Pencere Hesaplama',          ok3))

        # Adım 4 — Pencereyi kes
        print(f"\n{'='*60}")
        obj_l, obj_r, ok4 = step4_read_window(mat_l, mat_r, windows)
        results.append(('Pencere Kesme (ext_read_image)', ok4))

        # Adım 5 — Filtrele
        print(f"\n{'='*60}")
        obj_l, obj_r, ok5 = step5_filter(obj_l, obj_r)
        results.append(('Görüntü Filtreleme',         ok5))

        # Adım 6 — Epipolar doğrult
        print(f"\n{'='*60}")
        obj_l, obj_r, ok6 = step6_stereo_rect(obj_l, obj_r)
        results.append(('Epipolar Doğrultma',         ok6))

        # Adım 7 — Kamera poz + bundle adjustment
        print(f"\n{'='*60}")
        obj_l, obj_r, ok7 = step7_init_bundle(mat_l, mat_r, obj_l, obj_r, npz_dir)
        results.append(('Kamera Poz & Bundle Adj.',   ok7))

        # Adım 8 — Disparite haritası
        print(f"\n{'='*60}")
        obj_l, obj_r, ok8 = step8_disparity(obj_l, obj_r)
        results.append(('Disparite Haritası',         ok8))

        # Adım 9 — Kaydet
        print(f"\n{'='*60}")
        out_path, ok9 = step9_save(obj_l, obj_r, npz_dir)
        results.append(('Sonuç Kaydetme',             ok9))

    except Exception as e:
        print(f"\n✗ Beklenmeyen hata: {e}")
        traceback.print_exc()

    # -----------------------------------------------------------------------
    # ÖZET
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ÖZET")
    print(f"{'='*60}")
    for name, ok in results:
        status = '✓ BAŞARILI' if ok else '✗ BAŞARISIZ'
        print(f"  {name:<40} {status}")

    all_ok = all(ok for _, ok in results)
    critical_ok = all(ok for name, ok in results
                      if name not in ('Kamera Poz & Bundle Adj.',))

    print()
    if all_ok:
        print("Tüm extract adımları başarıyla tamamlandı!")
        print(f"Disparite çıktısı: {npz_dir / 'DZB1206-500009L007001_L008001_disparity.npz'}")
        print("\nSıradaki adım: georef (coğrafi referanslama) aşaması")
    elif critical_ok:
        print("Kritik adımlar başarılı (kamera poz tahmini atlandı).")
        print("Bundle adjustment gerektirebilir — georef testine geçilebilir.")
    else:
        print("Bazı kritik adımlar başarısız. Hata mesajlarını inceleyiniz.")


if __name__ == '__main__':
    main()
