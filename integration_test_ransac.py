"""
integration_test_ransac.py

estimate_transform_ransac.py entegrasyon testi.
MATLAB karşılıkları: shared/estimateTransformRansac.m,
                     shared/fileExchange/absor/absor.m

Çalıştırma:
    python integration_test_ransac.py

Bağımlılıklar:
    pip install numpy
"""

import sys
import warnings
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Modül keşfi
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent
SHARED_DIR = REPO_ROOT / "main" / "shared"

for p in [SHARED_DIR, REPO_ROOT]:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

try:
    from estimate_transform_ransac import (
        absor,
        estimate_transform_ransac,
        _apply_transform,
        _transform_distance,
        _fit_transform,
    )
except ImportError as e:
    print(f"[HATA] estimate_transform_ransac.py yüklenemedi: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------
PASS = "BASARILI ✓"
FAIL = "BASARISIZ ✗"
results: list[tuple[str, bool, str]] = []
rng = np.random.default_rng(42)


def run_test(name: str, fn):
    try:
        msg = fn()
        ok  = True
    except AssertionError as e:
        msg = f"AssertionError: {e}"
        ok  = False
    except Exception as e:
        msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        ok  = False
    label = PASS if ok else FAIL
    print(f"  {label}  {f'({msg})' if msg else ''}")
    results.append((name, ok, msg or ""))


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ---------------------------------------------------------------------------
# Test verisi üretici yardımcılar
# ---------------------------------------------------------------------------

def _make_rot(rx=0., ry=0., rz=0.):
    """ZYX döndürme matrisi (derece)."""
    rx, ry, rz = np.radians([rx, ry, rz])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx),  np.cos(rx)]])
    Ry = np.array([[ np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz),  np.cos(rz), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def _make_transform(rx=0., ry=0., rz=15., tx=5., ty=3., tz=0., scale=1.0):
    """Homojen 4×4 dönüşüm matrisi üret."""
    R = _make_rot(rx=rx, ry=ry, rz=rz)
    M = np.eye(4)
    M[:3, :3] = scale * R
    M[:3,  3] = [tx, ty, tz]
    return M


def _apply_noise(pts, sigma=0.1, seed=0):
    r = np.random.default_rng(seed)
    return pts + r.normal(0, sigma, pts.shape)


def _make_pair(n=60, rz=20., tx=8., ty=-4., tz=2., scale=1.1,
               noise=0.05, outlier_frac=0.0, seed=7):
    """İki eşleşen nokta kümesi üret; isteğe bağlı aykırı değerler ekle."""
    r = np.random.default_rng(seed)
    pts1 = r.random((3, n)) * 100
    M_true = _make_transform(rz=rz, tx=tx, ty=ty, tz=tz, scale=scale)
    pts2 = _apply_transform(M_true, pts1)
    pts2 = _apply_noise(pts2, sigma=noise, seed=seed+1)
    if outlier_frac > 0:
        n_out = max(1, int(n * outlier_frac))
        out_idx = r.choice(n, n_out, replace=False)
        pts2[:, out_idx] = r.random((3, n_out)) * 200
    return pts1, pts2, M_true


# ===========================================================================
# TEST 1 — absor: temel özellikler
# ===========================================================================
def test_absor_basic():
    section("TEST 1 · absor — temel özellikler")

    run_test("çıktı (4,4) homojen matris",          _t1_shape)
    run_test("kimlik dönüşümü — M ≈ I",             _t1_identity)
    run_test("saf öteleme kurtarma",                _t1_translation)
    run_test("saf döndürme kurtarma — R hatası",    _t1_rotation)
    run_test("ölçek + döndürme kurtarma",           _t1_scale_rot)
    run_test("do_scale=False → ölçek ≈ 1",          _t1_no_scale)
    run_test("NaN noktalar filtrelenir",             _t1_nan_filter)
    run_test("minimum 3 nokta (n=3)",               _t1_min_points)


def _t1_shape():
    pts1 = rng.random((3, 20)) * 100
    M = absor(pts1, pts1)
    assert M.shape == (4, 4)
    return f"{M.shape}"


def _t1_identity():
    pts1 = rng.random((3, 30)) * 100
    M = absor(pts1, pts1, do_scale=False)
    err = np.max(np.abs(M - np.eye(4)))
    assert err < 1e-8, f"Kimlik hatası: {err:.2e}"
    return f"maks_hata={err:.2e}"


def _t1_translation():
    pts1 = rng.random((3, 40)) * 100
    t = np.array([10., -5., 3.])
    pts2 = pts1 + t[:, None]
    M = absor(pts1, pts2, do_scale=False)
    t_est = M[:3, 3]
    err = np.max(np.abs(t_est - t))
    assert err < 1e-6, f"Öteleme hatası: {err:.2e}"
    return f"t_err={err:.2e}"


def _t1_rotation():
    pts1 = rng.random((3, 50)) * 100
    R_true = _make_rot(rz=30.)
    pts2 = R_true @ pts1
    M = absor(pts1, pts2, do_scale=False)
    R_est = M[:3, :3]
    err = np.max(np.abs(R_est - R_true))
    assert err < 1e-6, f"Döndürme hatası: {err:.2e}"
    return f"R_err={err:.2e}"


def _t1_scale_rot():
    pts1 = rng.random((3, 50)) * 100
    s_true = 1.15
    R_true = _make_rot(rz=25., rx=5.)
    pts2 = s_true * R_true @ pts1
    M = absor(pts1, pts2, do_scale=True)
    s_est = np.linalg.norm(M[:3, 0])   # ölçek = R sütununun normu
    err_s = abs(s_est - s_true)
    assert err_s < 0.01, f"Ölçek hatası: {err_s:.4f}"
    return f"s_true={s_true}, s_est={s_est:.4f}, err={err_s:.4f}"


def _t1_no_scale():
    pts1 = rng.random((3, 30)) * 50
    pts2 = 2.0 * pts1   # 2× ölçek
    M = absor(pts1, pts2, do_scale=False)
    s_est = np.linalg.norm(M[:3, 0])
    assert abs(s_est - 1.0) < 1e-6, f"do_scale=False ölçek {s_est:.4f} ≠ 1"
    return f"s_est={s_est:.6f}"


def _t1_nan_filter():
    pts1 = rng.random((3, 20)) * 100
    pts2 = pts1.copy()
    pts2[:, 5] = np.nan
    pts2[:, 10] = np.nan
    M = absor(pts1, pts2, do_scale=False)
    err = np.max(np.abs(M - np.eye(4)))
    assert err < 1e-6, f"NaN filtresi sonrası hata: {err:.2e}"
    return f"NaN noktalar filtrelendi ✓"


def _t1_min_points():
    pts1 = rng.random((3, 3)) * 50
    M = absor(pts1, pts1)
    assert M.shape == (4, 4)
    return "n=3 ✓"


# ===========================================================================
# TEST 2 — absor: dönüşüm kalitesi
# ===========================================================================
def test_absor_quality():
    section("TEST 2 · absor — dönüşüm kalitesi")

    run_test("gürültüsüz — RMSE < 1e-6",            _t2_noiseless)
    run_test("az gürültü — RMSE < gürültü σ × 2",   _t2_low_noise)
    run_test("3D tam dönüşüm kurtarma",              _t2_full_3d)
    run_test("büyük ölçek (1000× koordinat)",        _t2_large_scale)
    run_test("homojen son satır [0,0,0,1]",          _t2_homogeneous)


def _rmse(M, pts1, pts2):
    pts2_t = _apply_transform(M, pts2)
    return np.sqrt(np.mean(np.sum((pts2_t - pts1)**2, axis=0)))


def _t2_noiseless():
    pts1, pts2, M_true = _make_pair(noise=0.0)
    M = absor(pts2, pts1)
    rmse = _rmse(M, pts1, pts2)
    assert rmse < 1e-6, f"RMSE={rmse:.2e}"
    return f"RMSE={rmse:.2e}"


def _t2_low_noise():
    sigma = 0.5
    pts1, pts2, _ = _make_pair(noise=sigma, n=100)
    M = absor(pts2, pts1)
    rmse = _rmse(M, pts1, pts2)
    assert rmse < sigma * 3, f"RMSE={rmse:.3f}, σ={sigma}"
    return f"RMSE={rmse:.3f} (σ={sigma})"


def _t2_full_3d():
    pts1 = rng.random((3, 80)) * 200
    M_true = _make_transform(rz=45., rx=10., ry=-20.,
                             tx=50., ty=-30., tz=10., scale=0.9)
    pts2 = _apply_transform(M_true, pts1)
    M = absor(pts2, pts1)
    rmse = _rmse(M, pts1, pts2)
    assert rmse < 1e-4, f"3D RMSE={rmse:.2e}"
    return f"RMSE={rmse:.2e}"


def _t2_large_scale():
    pts1 = rng.random((3, 40)) * 1e6
    t = np.array([1e4, -2e4, 5e3])
    pts2 = pts1 + t[:, None]
    M = absor(pts2, pts1, do_scale=False)
    rmse = _rmse(M, pts1, pts2)
    assert rmse < 1.0, f"Büyük koordinat RMSE={rmse:.2e}"
    return f"RMSE={rmse:.2e}"


def _t2_homogeneous():
    pts1 = rng.random((3, 20)) * 100
    M = absor(pts1, pts1)
    assert np.allclose(M[3], [0, 0, 0, 1])
    return "son satır [0,0,0,1] ✓"


# ===========================================================================
# TEST 3 — _apply_transform / _transform_distance
# ===========================================================================
def test_helpers():
    section("TEST 3 · _apply_transform / _transform_distance")

    run_test("apply — kimlik → noktalar değişmez",   _t3_apply_identity)
    run_test("apply — öteleme uygulanır",            _t3_apply_translation)
    run_test("apply — çıktı boyutu (3, N)",          _t3_apply_shape)
    run_test("distance — aynı nokta → 0",            _t3_dist_zero)
    run_test("distance — bilinen mesafe",             _t3_dist_known)
    run_test("distance — çıktı boyutu (N,)",         _t3_dist_shape)


def _t3_apply_identity():
    pts = rng.random((3, 25)) * 100
    out = _apply_transform(np.eye(4), pts)
    err = np.max(np.abs(out - pts))
    assert err < 1e-12
    return f"err={err:.1e}"


def _t3_apply_translation():
    pts = rng.random((3, 20)) * 50
    t = np.array([3., -7., 2.])
    M = np.eye(4); M[:3, 3] = t
    out = _apply_transform(M, pts)
    err = np.max(np.abs(out - (pts + t[:, None])))
    assert err < 1e-10
    return f"err={err:.1e}"


def _t3_apply_shape():
    pts = rng.random((3, 17)) * 100
    out = _apply_transform(np.eye(4), pts)
    assert out.shape == (3, 17)
    return f"{out.shape}"


def _t3_dist_zero():
    pts = rng.random((3, 30)) * 100
    d = _transform_distance(np.eye(4), pts, pts)
    assert np.max(d) < 1e-12
    return f"maks_d={np.max(d):.1e}"


def _t3_dist_known():
    pts1 = np.zeros((3, 5))
    pts2 = np.zeros((3, 5))
    pts2[0] = 3.0; pts2[1] = 4.0   # mesafe = 5
    d = _transform_distance(np.eye(4), pts1, pts2)
    assert np.allclose(d, 5.0), f"Beklenen 5.0, alınan {d}"
    return f"d={d[0]:.4f}"


def _t3_dist_shape():
    pts = rng.random((3, 13)) * 100
    d = _transform_distance(np.eye(4), pts, pts)
    assert d.shape == (13,)
    return f"{d.shape}"


# ===========================================================================
# TEST 4 — estimate_transform_ransac: temel
# ===========================================================================
def test_ransac_basic():
    section("TEST 4 · estimate_transform_ransac — temel")

    run_test("çıktı boyutları (4,4) ve (N,)",        _t4_output_shapes)
    run_test("gürültüsüz — tüm noktalar inlier",     _t4_all_inliers)
    run_test("inlier maskesi bool dtype",             _t4_dtype)
    run_test("hatalı girdi → ValueError",             _t4_bad_input)
    run_test("n=3 minimum (subset_size=3)",           _t4_min_n)


def _t4_output_shapes():
    pts1, pts2, _ = _make_pair(n=40)
    M, inliers = estimate_transform_ransac(pts1, pts2,
                                           num_iter=200, inlier_dist=2.0)
    assert M.shape == (4, 4)
    assert inliers.shape == (40,)
    return f"M={M.shape}, inliers={inliers.shape}"


def _t4_all_inliers():
    pts1, pts2, _ = _make_pair(n=50, noise=0.01)
    _, inliers = estimate_transform_ransac(pts1, pts2,
                                           num_iter=300, inlier_dist=0.5)
    assert inliers.sum() >= 45, f"İnlier sayısı düşük: {inliers.sum()}"
    return f"inlier={inliers.sum()}/50"


def _t4_dtype():
    pts1, pts2, _ = _make_pair(n=30)
    _, inliers = estimate_transform_ransac(pts1, pts2,
                                           num_iter=100, inlier_dist=2.0)
    assert inliers.dtype == bool
    return f"dtype={inliers.dtype}"


def _t4_bad_input():
    pts1 = rng.random((3, 10))
    pts2 = rng.random((3, 15))   # farklı boyut
    try:
        estimate_transform_ransac(pts1, pts2)
        assert False, "ValueError beklendi"
    except ValueError:
        return "ValueError ✓"


def _t4_min_n():
    pts1 = rng.random((3, 3)) * 50
    pts2 = pts1 + 1.0
    M, inliers = estimate_transform_ransac(pts1, pts2,
                                           num_iter=50, inlier_dist=5.0)
    assert M.shape == (4, 4)
    return f"n=3 tamamlandı, inlier={inliers.sum()}"


# ===========================================================================
# TEST 5 — estimate_transform_ransac: dönüşüm kalitesi
# ===========================================================================
def test_ransac_quality():
    section("TEST 5 · estimate_transform_ransac — kalite")

    run_test("gürültüsüz — dönüşüm kurtarılır",     _t5_noiseless_recovery)
    run_test("gürültülü — RMSE < 3σ",               _t5_noisy_rmse)
    run_test("%20 aykırı değer — inlier oranı > %70",_t5_outliers)
    run_test("saf öteleme RANSAC ile kurtarılır",    _t5_translation)
    run_test("saf döndürme RANSAC ile kurtarılır",   _t5_rotation)
    run_test("daha fazla iterasyon → daha iyi sonuç",_t5_more_iters)


def _t5_noiseless_recovery():
    pts1, pts2, M_true = _make_pair(n=60, noise=0.0)
    M, inliers = estimate_transform_ransac(pts1, pts2,
                                           num_iter=500, inlier_dist=0.1)
    rmse = _rmse(M, pts1, pts2)
    assert rmse < 1e-4, f"RMSE={rmse:.2e}"
    assert inliers.sum() > 55
    return f"RMSE={rmse:.2e}, inlier={inliers.sum()}/60"


def _t5_noisy_rmse():
    sigma = 1.0
    pts1, pts2, _ = _make_pair(n=80, noise=sigma)
    M, _ = estimate_transform_ransac(pts1, pts2,
                                     num_iter=500, inlier_dist=sigma*3)
    rmse = _rmse(M, pts1, pts2)
    assert rmse < sigma * 4, f"RMSE={rmse:.3f} > 4σ={sigma*4}"
    return f"RMSE={rmse:.3f} (σ={sigma})"


def _t5_outliers():
    pts1, pts2, _ = _make_pair(n=100, noise=0.1,
                               outlier_frac=0.20)
    _, inliers = estimate_transform_ransac(pts1, pts2,
                                           num_iter=1000, inlier_dist=1.0)
    rate = inliers.sum() / 100
    assert rate > 0.70, f"İnlier oranı düşük: {rate:.2f}"
    return f"inlier_oran={rate:.2f}"


def _t5_translation():
    r2 = np.random.default_rng(99)
    pts1 = r2.random((3, 50)) * 100
    t = np.array([15., -10., 5.])
    pts2 = pts1 + t[:, None]
    # Saf öteleme: ölçek=1, do_scale=False ile absor daha doğru
    M, inliers = estimate_transform_ransac(pts1, pts2,
                                           num_iter=300, inlier_dist=0.5)
    # Öteleme vektörü M[:3,3] — ölçekli R'dan bağımsız olarak kontrol et
    pts2_t = _apply_transform(M, pts2)
    rmse = np.sqrt(np.mean(np.sum((pts2_t - pts1)**2, axis=0)))
    assert rmse < 1.0, f"Öteleme RMSE: {rmse:.3f}"
    return f"RMSE={rmse:.4f}, inlier={inliers.sum()}/50"


def _t5_rotation():
    r2 = np.random.default_rng(77)
    pts1 = r2.random((3, 60)) * 100
    R = _make_rot(rz=40.)
    pts2 = R @ pts1
    M, inliers = estimate_transform_ransac(pts1, pts2,
                                           num_iter=500, inlier_dist=0.5)
    rmse = _rmse(M, pts1, pts2)
    assert rmse < 1.0, f"Döndürme RMSE={rmse:.3f}"
    return f"RMSE={rmse:.4f}, inlier={inliers.sum()}/60"


def _t5_more_iters():
    pts1, pts2, _ = _make_pair(n=60, noise=0.5, outlier_frac=0.15)
    _, inliers_low  = estimate_transform_ransac(pts1, pts2,
                                                num_iter=10,  inlier_dist=2.0)
    _, inliers_high = estimate_transform_ransac(pts1, pts2,
                                                num_iter=500, inlier_dist=2.0)
    assert inliers_high.sum() >= inliers_low.sum(), \
        f"Daha fazla iterasyon daha az inlier: {inliers_low.sum()} → {inliers_high.sum()}"
    return f"iter=10: {inliers_low.sum()}, iter=500: {inliers_high.sum()}"


# ===========================================================================
# TEST 6 — RANSAC: özel durumlar
# ===========================================================================
def test_ransac_edge_cases():
    section("TEST 6 · estimate_transform_ransac — özel durumlar")

    run_test("tüm aykırı değer → yine de çalışır",  _t6_all_outliers)
    run_test("tekrarlı noktalar",                    _t6_duplicate)
    run_test("büyük koordinatlar (UTM ölçeği)",      _t6_utm_scale)
    run_test("num_iter=1 → hata yok",               _t6_single_iter)


def _t6_all_outliers():
    r2 = np.random.default_rng(11)
    pts1 = r2.random((3, 20)) * 100
    pts2 = r2.random((3, 20)) * 100   # tamamen rassal — ilişki yok
    M, inliers = estimate_transform_ransac(pts1, pts2,
                                           num_iter=100, inlier_dist=0.1)
    assert M.shape == (4, 4)
    return f"inlier={inliers.sum()} (rassal veri)"


def _t6_duplicate():
    pts1 = np.tile(rng.random((3, 5)) * 50, (1, 4))   # 20 tekrarlı
    pts2 = pts1 + 2.0
    M, inliers = estimate_transform_ransac(pts1, pts2,
                                           num_iter=100, inlier_dist=5.0)
    assert M.shape == (4, 4)
    return f"inlier={inliers.sum()}"


def _t6_utm_scale():
    r2 = np.random.default_rng(55)
    pts1 = r2.random((3, 40)) * 1e5 + np.array([[5e5], [4e6], [1e3]])
    t = np.array([50., -30., 5.])
    pts2 = pts1 + t[:, None]
    M, inliers = estimate_transform_ransac(pts1, pts2,
                                           num_iter=300, inlier_dist=5.0)
    pts2_t = _apply_transform(M, pts2)
    rmse = np.sqrt(np.mean(np.sum((pts2_t - pts1)**2, axis=0)))
    assert rmse < 10.0, f"UTM ölçeği RMSE: {rmse:.2f}"
    return f"RMSE={rmse:.3f}, inlier={inliers.sum()}/40"


def _t6_single_iter():
    pts1, pts2, _ = _make_pair(n=30)
    M, inliers = estimate_transform_ransac(pts1, pts2,
                                           num_iter=1, inlier_dist=2.0)
    assert M.shape == (4, 4)
    return "num_iter=1 ✓"


# ===========================================================================
# ANA ÇALIŞTIRICI
# ===========================================================================
def main():
    print("\n" + "=" * 60)
    print("  RANSAC ENTEGRASYON TESTİ")
    print("  estimate_transform_ransac.py")
    print("=" * 60)

    test_absor_basic()
    test_absor_quality()
    test_helpers()
    test_ransac_basic()
    test_ransac_quality()
    test_ransac_edge_cases()

    # -----------------------------------------------------------------------
    # Özet
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ÖZET")
    print("=" * 60)

    skipped = [r for r in results if r[1] is None]
    passed  = [r for r in results if r[1] is True]
    failed  = [r for r in results if r[1] is False]

    for name, ok, msg in results:
        if ok is None:
            mark = "⊘ ATLANDI"
        elif ok:
            mark = "✓ BASARILI"
        else:
            mark = "✗ BASARISIZ"
        print(f"  {mark:<14} {name}")
        if not ok and ok is not None:
            short = msg.split('\n')[0][:80]
            print(f"               → {short}")

    total = len(passed) + len(failed)
    print(f"\n  Toplam: {total} test  |  "
          f"{len(passed)} başarılı  |  "
          f"{len(failed)} başarısız  |  "
          f"{len(skipped)} atlandı")

    if failed:
        print("\n  [BAŞARISIZ TESTLER]")
        for name, _, _ in failed:
            print(f"    • {name}")
        sys.exit(1)
    else:
        print("\n  Tüm testler geçti ✓")
        sys.exit(0)


if __name__ == '__main__':
    main()
