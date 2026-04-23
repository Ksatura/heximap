"""
integration_test_utils.py

utils.py entegrasyon testi.
MATLAB karşılıkları: shared/makeRotMat.m, shared/triangulate.m,
                     shared/fileExchange/voicebox/rotro2eu.m,
                     shared/getFiles.m

Çalıştırma:
    python integration_test_utils.py

Bağımlılıklar:
    pip install numpy scipy
"""

import sys
import traceback
import tempfile
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
    from utils import make_rot_mat, rotro2eu, triangulate, get_files
except ImportError as e:
    print(f"[HATA] utils.py yüklenemedi: {e}")
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


# ===========================================================================
# TEST 1 — make_rot_mat: temel özellikler
# ===========================================================================
def test_make_rot_mat_basic():
    section("TEST 1 · make_rot_mat — temel özellikler")

    run_test("sıfır açı → birim matris",             _t1_zero)
    run_test("çıktı şekli (3, 3)",                   _t1_shape)
    run_test("ortogonal matris — R^T R = I",         _t1_orthogonal)
    run_test("determinant = +1",                     _t1_det)
    run_test("X ekseni 90° döndürme",                _t1_rx90)
    run_test("Y ekseni 90° döndürme",                _t1_ry90)
    run_test("Z ekseni 90° döndürme",                _t1_rz90)
    run_test("180° döndürme — öz değer",             _t1_180)


def _t1_zero():
    R = make_rot_mat(0., 0., 0.)
    err = np.max(np.abs(R - np.eye(3)))
    assert err < 1e-14
    return f"err={err:.1e}"


def _t1_shape():
    R = make_rot_mat(10., 20., 30.)
    assert R.shape == (3, 3)
    return f"{R.shape}"


def _t1_orthogonal():
    for rx, ry, rz in [(15., -5., 30.), (90., 0., 0.), (0., 0., 45.)]:
        R = make_rot_mat(rx, ry, rz)
        err = np.max(np.abs(R.T @ R - np.eye(3)))
        assert err < 1e-12, f"Ortogonalite hatası ({rx},{ry},{rz}): {err:.2e}"
    return "R^T R = I ✓"


def _t1_det():
    for rx, ry, rz in [(10., 20., 30.), (0., 90., 0.), (45., 45., 45.)]:
        R = make_rot_mat(rx, ry, rz)
        d = np.linalg.det(R)
        assert abs(d - 1.0) < 1e-12, f"det={d:.6f}"
    return "det=1 ✓"


def _t1_rx90():
    R = make_rot_mat(90., 0., 0.)
    expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    err = np.max(np.abs(R - expected))
    assert err < 1e-10, f"Hata: {err:.2e}"
    return f"err={err:.1e}"


def _t1_ry90():
    R = make_rot_mat(0., 90., 0.)
    expected = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    err = np.max(np.abs(R - expected))
    assert err < 1e-10, f"Hata: {err:.2e}"
    return f"err={err:.1e}"


def _t1_rz90():
    R = make_rot_mat(0., 0., 90.)
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    err = np.max(np.abs(R - expected))
    assert err < 1e-10, f"Hata: {err:.2e}"
    return f"err={err:.1e}"


def _t1_180():
    R = make_rot_mat(180., 0., 0.)
    # v = [0, 1, 0] → [0, -1, 0]
    v = np.array([0., 1., 0.])
    v_rot = R @ v
    assert np.allclose(v_rot, [0., -1., 0.], atol=1e-10)
    return "180° X ✓"


# ===========================================================================
# TEST 2 — make_rot_mat: bileşim özellikleri
# ===========================================================================
def test_make_rot_mat_composition():
    section("TEST 2 · make_rot_mat — bileşim")

    run_test("sıra bağımlılığı — Rx*Ry ≠ Ry*Rx",    _t2_noncommutative)
    run_test("ters döndürme — R(θ)R(-θ) = I",        _t2_inverse)
    run_test("360° → birim matris",                  _t2_full_circle)
    run_test("küçük açı lineerlik",                  _t2_small_angle)


def _t2_noncommutative():
    Rxy = make_rot_mat(30., 45., 0.)
    Ryx = make_rot_mat(45., 30., 0.)   # rx,ry değerleri farklı sıra
    # Bunlar genel olarak farklı olmalı
    assert not np.allclose(Rxy, Ryx, atol=1e-6)
    return "Rx*Ry ≠ Ry*Rx ✓"


def _t2_inverse():
    for rx, ry, rz in [(15., -20., 35.), (0., 90., 0.)]:
        R  = make_rot_mat( rx,  ry,  rz)
        Ri = make_rot_mat(-rx, -ry, -rz)
        # R @ Ri yaklaşık birim matris olmalı (ZYX ters sıra nedeniyle yaklaşık)
        prod = R @ Ri
        # En azından her birinin tersi: R^-1 = R^T (ortogonal)
        err = np.max(np.abs(R.T @ R - np.eye(3)))
        assert err < 1e-12
    return "R^T = R^-1 ✓"


def _t2_full_circle():
    R = make_rot_mat(360., 0., 0.)
    err = np.max(np.abs(R - np.eye(3)))
    assert err < 1e-10
    return f"err={err:.1e}"


def _t2_small_angle():
    """Küçük açı: sin(θ)≈θ, cos(θ)≈1."""
    theta = 0.001   # derece — çok küçük
    R = make_rot_mat(theta, 0., 0.)
    theta_r = np.radians(theta)
    assert abs(R[1, 2] + theta_r) < 1e-8
    assert abs(R[2, 1] - theta_r) < 1e-8
    return f"θ={theta}° lineer ✓"


# ===========================================================================
# TEST 3 — rotro2eu
# ===========================================================================
def test_rotro2eu():
    section("TEST 3 · rotro2eu")

    run_test("çıktı boyutu (3,)",                    _t3_shape)
    run_test("gidiş-dönüş — make_rot_mat ile",       _t3_roundtrip)
    run_test("sıfır açı → sıfır çıktı",              _t3_zero)
    run_test("birden fazla açı kombinasyonu",         _t3_multiple)
    run_test("birim matris → sıfır açı",             _t3_identity_input)


def _t3_shape():
    R = make_rot_mat(10., 20., 30.)
    e = rotro2eu('zyx', R)
    assert e.shape == (3,)
    return f"{e.shape}"


def _t3_roundtrip():
    for angles in [(15., -5., 30.), (45., 0., -90.), (1., 2., 3.)]:
        R = make_rot_mat(*angles)
        e = rotro2eu('zyx', R)
        R_back = make_rot_mat(
            np.degrees(e[0]),
            np.degrees(e[1]),
            np.degrees(e[2]),
        )
        err = np.max(np.abs(R - R_back))
        assert err < 1e-8, f"Gidiş-dönüş hatası {angles}: {err:.2e}"
    return "gidiş-dönüş ✓"


def _t3_zero():
    R = np.eye(3)
    e = rotro2eu('zyx', R)
    assert np.max(np.abs(e)) < 1e-10
    return f"maks_açı={np.max(np.abs(e)):.1e}"


def _t3_multiple():
    combos = [(0., 0., 45.), (30., 0., 0.), (0., 60., 0.), (10., 20., 30.)]
    for rx, ry, rz in combos:
        R = make_rot_mat(rx, ry, rz)
        e = rotro2eu('zyx', R)
        R_back = make_rot_mat(*np.degrees(e))
        err = np.max(np.abs(R - R_back))
        assert err < 1e-8, f"({rx},{ry},{rz}) gidiş-dönüş hatası: {err:.2e}"
    return f"{len(combos)} kombinasyon ✓"


def _t3_identity_input():
    e = rotro2eu('zyx', np.eye(3))
    assert np.allclose(e, 0., atol=1e-10)
    return "sıfır açı ✓"


# ===========================================================================
# TEST 4 — triangulate: temel
# ===========================================================================
def test_triangulate_basic():
    section("TEST 4 · triangulate — temel")

    run_test("çıktı şekli (4, N)",                   _t4_shape)
    run_test("homojen — son koordinat 1",             _t4_homogeneous)
    run_test("compute_error=False → boş hata",        _t4_no_error)
    run_test("compute_error=True → hata vektörü",     _t4_with_error)
    run_test("tek nokta (N=1)",                       _t4_single_point)


def _make_cameras():
    K = np.array([[1000., 0., 320.],
                  [0., 1000., 240.],
                  [0.,    0.,   1.]], dtype=float)
    P1 = np.eye(3, 4, dtype=float)
    P2 = np.array([[1, 0, 0, -100],
                   [0, 1, 0,    0],
                   [0, 0, 1,    0]], dtype=float)
    return K, P1, P2


def _make_projections(pts_true, K, P1, P2):
    pts_h = np.vstack([pts_true, np.ones((1, pts_true.shape[1]))])
    proj1 = K @ P1 @ pts_h; proj1 /= proj1[2:3]
    proj2 = K @ P2 @ pts_h; proj2 /= proj2[2:3]
    return proj1, proj2


_PTS_TRUE = np.array([[10., -5., 20.],
                      [ 0.,  0., 50.],
                      [30., 10., 80.]], dtype=float).T   # (3, 3)


def _t4_shape():
    K, P1, P2 = _make_cameras()
    p1, p2 = _make_projections(_PTS_TRUE, K, P1, P2)
    pts3d, _ = triangulate(p1, p2, P1, P2, K, K)
    assert pts3d.shape == (4, 3)
    return f"{pts3d.shape}"


def _t4_homogeneous():
    K, P1, P2 = _make_cameras()
    p1, p2 = _make_projections(_PTS_TRUE, K, P1, P2)
    pts3d, _ = triangulate(p1, p2, P1, P2, K, K)
    assert np.allclose(pts3d[3], 1.0, atol=1e-8)
    return "son koordinat=1 ✓"


def _t4_no_error():
    K, P1, P2 = _make_cameras()
    p1, p2 = _make_projections(_PTS_TRUE, K, P1, P2)
    _, error = triangulate(p1, p2, P1, P2, K, K, compute_error=False)
    assert error.size == 0
    return "boş hata ✓"


def _t4_with_error():
    K, P1, P2 = _make_cameras()
    p1, p2 = _make_projections(_PTS_TRUE, K, P1, P2)
    _, error = triangulate(p1, p2, P1, P2, K, K, compute_error=True)
    assert error.size == 4 * 3   # 4 * N
    return f"hata boyutu={error.size}"


def _t4_single_point():
    K, P1, P2 = _make_cameras()
    pt = np.array([[10., 5., 30.]]).T
    p1, p2 = _make_projections(pt, K, P1, P2)
    pts3d, _ = triangulate(p1, p2, P1, P2, K, K)
    assert pts3d.shape == (4, 1)
    return f"{pts3d.shape}"


# ===========================================================================
# TEST 5 — triangulate: doğruluk
# ===========================================================================
def test_triangulate_accuracy():
    section("TEST 5 · triangulate — doğruluk")

    run_test("gürültüsüz — koordinat hatası < 1e-4",  _t5_noiseless)
    run_test("gürültüsüz — projeksiyon hatası < 1e-6", _t5_proj_error)
    run_test("farklı kamera taban çizgisi",            _t5_baseline)
    run_test("çok nokta (N=20)",                       _t5_many_points)
    run_test("büyük derinlik (Z=1000)",                _t5_large_depth)


def _t5_noiseless():
    K, P1, P2 = _make_cameras()
    p1, p2 = _make_projections(_PTS_TRUE, K, P1, P2)
    pts3d, _ = triangulate(p1, p2, P1, P2, K, K)
    pts_xyz = pts3d[:3] / pts3d[3:4]
    err = np.max(np.abs(pts_xyz - _PTS_TRUE))
    assert err < 1e-4, f"Koordinat hatası: {err:.2e}"
    return f"maks_err={err:.2e}"


def _t5_proj_error():
    K, P1, P2 = _make_cameras()
    p1, p2 = _make_projections(_PTS_TRUE, K, P1, P2)
    _, error = triangulate(p1, p2, P1, P2, K, K, compute_error=True)
    assert error.mean() < 1e-6, f"Proj hatası: {error.mean():.2e}"
    return f"ort_proj_err={error.mean():.2e}"


def _t5_baseline():
    K = np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]], dtype=float)
    P1 = np.eye(3, 4, dtype=float)
    # Geniş taban çizgisi
    P2 = np.array([[1, 0, 0, -300], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    pts = np.array([[0., 0., 100.], [50., 30., 200.]]).T
    p1, p2 = _make_projections(pts, K, P1, P2)
    pts3d, _ = triangulate(p1, p2, P1, P2, K, K)
    pts_xyz = pts3d[:3] / pts3d[3:4]
    err = np.max(np.abs(pts_xyz - pts))
    assert err < 1e-4, f"Taban çizgisi hatası: {err:.2e}"
    return f"err={err:.2e}"


def _t5_many_points():
    K, P1, P2 = _make_cameras()
    pts = rng.random((3, 20)) * 50 + np.array([[0.], [0.], [50.]])
    p1, p2 = _make_projections(pts, K, P1, P2)
    pts3d, _ = triangulate(p1, p2, P1, P2, K, K)
    pts_xyz = pts3d[:3] / pts3d[3:4]
    err = np.max(np.abs(pts_xyz - pts))
    assert err < 1e-3, f"N=20 hatası: {err:.2e}"
    return f"N=20, maks_err={err:.2e}"


def _t5_large_depth():
    K, P1, P2 = _make_cameras()
    pts = np.array([[0., 0., 1000.]]).T
    p1, p2 = _make_projections(pts, K, P1, P2)
    pts3d, _ = triangulate(p1, p2, P1, P2, K, K)
    pts_xyz = pts3d[:3] / pts3d[3:4]
    err = abs(pts_xyz[2, 0] - 1000.)
    assert err < 1.0, f"Büyük derinlik Z hatası: {err:.4f}"
    return f"Z_err={err:.4f}"


# ===========================================================================
# TEST 6 — get_files
# ===========================================================================
def test_get_files():
    section("TEST 6 · get_files")

    run_test("düz klasör — uzantı filtresi",          _t6_flat)
    run_test("özyinelemeli alt klasör",               _t6_recursive)
    run_test("son ek filtresi (Left.npz)",            _t6_suffix)
    run_test("eşleşme yok → boş liste",              _t6_no_match)
    run_test("var olmayan klasör → boş liste",        _t6_missing)
    run_test("çıktı sıralı",                         _t6_sorted)
    run_test("tam yol döndürülür",                   _t6_full_path)


def _t6_flat():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / 'a.tif').touch()
        (Path(d) / 'b.tif').touch()
        (Path(d) / 'c.shp').touch()
        files = get_files(d, '.tif')
        assert len(files) == 2
        return f"{len(files)} .tif"


def _t6_recursive():
    with tempfile.TemporaryDirectory() as d:
        sub = Path(d) / 'sub'
        sub.mkdir()
        (Path(d) / 'a.npz').touch()
        (sub / 'b.npz').touch()
        (sub / 'c.txt').touch()
        files = get_files(d, '.npz')
        assert len(files) == 2
        return f"{len(files)} .npz"


def _t6_suffix():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / 'Left.mat').touch()
        (Path(d) / 'Right.mat').touch()
        sub = Path(d) / 'sub'
        sub.mkdir()
        (sub / 'Left.mat').touch()
        files = get_files(d, 'Left.mat')
        assert len(files) == 2
        return f"{len(files)} Left.mat"


def _t6_no_match():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / 'a.txt').touch()
        files = get_files(d, '.tif')
        assert files == []
        return "boş liste ✓"


def _t6_missing():
    files = get_files('/tmp/bu_klasor_yoktur_xyz789', '.tif')
    assert files == []
    return "boş liste ✓"


def _t6_sorted():
    with tempfile.TemporaryDirectory() as d:
        for name in ['c.tif', 'a.tif', 'b.tif']:
            (Path(d) / name).touch()
        files = get_files(d, '.tif')
        assert files == sorted(files)
        return "sıralı ✓"


def _t6_full_path():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'test.tif'
        f.touch()
        files = get_files(d, '.tif')
        assert len(files) == 1
        assert Path(files[0]).is_absolute()
        return "tam yol ✓"


# ===========================================================================
# ANA ÇALIŞTIRICI
# ===========================================================================
def main():
    print("\n" + "=" * 60)
    print("  UTILS ENTEGRASYON TESTİ")
    print("  utils.py")
    print("=" * 60)

    test_make_rot_mat_basic()
    test_make_rot_mat_composition()
    test_rotro2eu()
    test_triangulate_basic()
    test_triangulate_accuracy()
    test_get_files()

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
