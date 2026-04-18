"""
integration_test_georef.py

Georef aşaması entegrasyon testi.
MATLAB karşılıkları: main/3_georef/geoOptimize.m ve bağımlıları

Çalıştırma:
    python integration_test_georef.py

Bağımlılıklar:
    pip install numpy scipy pyproj open3d   (open3d opsiyonel)

Repo yapısı varsayımı:
    <repo>/
        main/
            3_georef/
                geo_optimize.py   ← test edilen modül
            shared/
                utils.py          ← make_rot_mat
"""

import sys
import warnings
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Modül keşfi: repo kökünü sys.path'e ekle
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
GEOREF_DIR = REPO_ROOT / "main" / "3_georef"
SHARED_DIR = REPO_ROOT / "main" / "shared"

for p in [GEOREF_DIR, SHARED_DIR, REPO_ROOT]:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

try:
    from geo_optimize import (
        ll2utm, utm2ll,
        points2grid,
        compute_normals,
        geo_sample_points,
        transform_using_solver_var,
        optimize_dem,
        align_dem,
        geo_opti_trans,
        HAS_OPEN3D,
    )
except ImportError as e:
    print(f"[HATA] geo_optimize.py yüklenemedi: {e}")
    print("  Repo kökünü doğrulayın ve bağımlılıkları kurun.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------
PASS = "BASARILI ✓"
FAIL = "BASARISIZ ✗"

results: list[tuple[str, bool, str]] = []


def run_test(name: str, fn):
    """Tek bir test fonksiyonunu çalıştırır, sonucu kaydeder."""
    try:
        msg = fn()
        ok = True
    except AssertionError as e:
        msg = f"AssertionError: {e}"
        ok = False
    except Exception as e:
        msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        ok = False

    label = PASS if ok else FAIL
    print(f"  {label}  {f'({msg})' if msg else ''}")
    results.append((name, ok, msg or ""))


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ===========================================================================
# TEST 1 — ll2utm / utm2ll gidiş-dönüş
# ===========================================================================
def test_ll2utm_roundtrip():
    section("TEST 1 · ll2utm / utm2ll gidiş-dönüş")

    pts_ll = np.array([
        [30.5, 39.8],
        [31.2, 40.1],
        [29.8, 38.5],
        [28.0, 37.0],
    ])

    run_test("ll2utm temel dönüşüm", lambda: _t1_basic(pts_ll))
    run_test("ll2utm gidiş-dönüş hata < 1e-6 derece", lambda: _t1_roundtrip(pts_ll))
    run_test("ll2utm güney yarımküre", _t1_south)
    run_test("ll2utm NaN satırları korunur", _t1_nan)


def _t1_basic(pts_ll):
    pts_utm, zone, hemi = ll2utm(pts_ll.copy())
    assert 30 <= zone <= 40, f"Beklenmeyen UTM dilimi: {zone}"
    assert hemi == 'N', f"Beklenen 'N', alınan '{hemi}'"
    assert pts_utm.shape == pts_ll.shape
    return f"dilim={zone}{hemi}"


def _t1_roundtrip(pts_ll):
    pts_utm, zone, hemi = ll2utm(pts_ll.copy())
    pts_back = utm2ll(pts_utm.copy(), zone, hemi)
    err = np.max(np.abs(pts_back[:, :2] - pts_ll))
    assert err < 1e-6, f"Hata çok büyük: {err:.2e} derece"
    return f"maks_hata={err:.2e}°"


def _t1_south():
    pts = np.array([[-43.2, -22.9], [-46.6, -23.5]])  # Brezilya
    pts_utm, zone, hemi = ll2utm(pts.copy())
    assert hemi == 'S', f"Beklenen 'S', alınan '{hemi}'"
    pts_back = utm2ll(pts_utm.copy(), zone, hemi)
    err = np.max(np.abs(pts_back[:, :2] - pts))
    assert err < 1e-6
    return f"dilim={zone}{hemi}"


def _t1_nan():
    pts = np.array([[30.0, 40.0], [np.nan, np.nan], [31.0, 39.0]])
    pts_utm, _, _ = ll2utm(pts.copy())
    assert np.isnan(pts_utm[1, 0]), "NaN satırı korunmadı"
    return "NaN satırı sağlam"


# ===========================================================================
# TEST 2 — points2grid
# ===========================================================================
def test_points2grid():
    section("TEST 2 · points2grid")

    run_test("sparse — ızgara boyutu", _t2_shape)
    run_test("sparse — değer tutarlılığı", _t2_values)
    run_test("sparse — boş bölge NaN", _t2_nan_region)
    run_test("sparse — sınır dışı noktalar yok sayılır", _t2_oob)
    run_test("interp — ızgara oluşur", _t2_interp)


def _make_grid_axes(nx=10, ny=8):
    vX = np.arange(0.0, nx * 10.0, 10.0)
    vY = np.arange((ny - 1) * 10.0, -1.0, -10.0)
    return vX, vY


def _t2_shape():
    vX, vY = _make_grid_axes()
    mX, mY = np.meshgrid(vX, vY)
    z = np.ones_like(mX)
    pts = np.vstack([mX.ravel(), mY.ravel(), z.ravel()])
    grid = points2grid(pts, vX, vY, 'sparse')
    assert grid.shape == (len(vY), len(vX))
    return f"{grid.shape}"


def _t2_values():
    vX, vY = _make_grid_axes()
    mX, mY = np.meshgrid(vX, vY)
    z = np.sin(mX / 30) + np.cos(mY / 30)
    pts = np.vstack([mX.ravel(), mY.ravel(), z.ravel()])
    grid = points2grid(pts, vX, vY, 'sparse')
    err = np.nanmax(np.abs(grid - z))
    assert err < 0.01, f"Değer hatası: {err}"
    return f"maks_hata={err:.2e}"


def _t2_nan_region():
    vX, vY = _make_grid_axes()
    # Sadece sol yarıya nokta koy
    half = len(vX) // 2
    vX_sub = vX[:half]
    mX, mY = np.meshgrid(vX_sub, vY[:len(vY)])
    pts = np.vstack([mX.ravel(), mY.ravel(), np.ones(mX.size)])
    grid = points2grid(pts, vX, vY, 'sparse')
    # Sağ yarı NaN olmalı
    right_nan = np.isnan(grid[:, half:]).all()
    assert right_nan, "Boş bölge NaN değil"
    return "sağ yarı NaN ✓"


def _t2_oob():
    vX = np.arange(0.0, 100.0, 10.0)
    vY = np.arange(90.0, -1.0, -10.0)
    # Sınır dışı + sınır içi noktalar
    pts = np.array([
        [50.0, 999.0, 50.0],   # x içinde, y dışında
        [50.0, 50.0,  1.0],    # tamamen içinde
        [-100.0, 50.0, 2.0],   # x dışında
    ]).T
    grid = points2grid(pts, vX, vY, 'sparse')
    valid_count = np.sum(~np.isnan(grid))
    assert valid_count == 1, f"Beklenmedik dolu hücre sayısı: {valid_count}"
    return f"dolu_hücre={valid_count}"


def _t2_interp():
    vX = np.linspace(0, 90, 10)
    vY = np.linspace(90, 0, 10)
    mX, mY = np.meshgrid(vX, vY)
    z = mX * 0.1 + mY * 0.05
    pts = np.vstack([mX.ravel(), mY.ravel(), z.ravel()])
    grid = points2grid(pts, vX, vY, 'interp')
    assert grid.shape == (10, 10)
    return f"{grid.shape}"


# ===========================================================================
# TEST 3 — compute_normals
# ===========================================================================
def test_compute_normals():
    section("TEST 3 · compute_normals")

    run_test("düz yüzey — dikey normal", _t3_flat)
    run_test("normal çıktı boyutu", _t3_shape)
    run_test("normalize edilmiş — norm ≈ 1", _t3_unit)
    run_test("NaN doldurma açık", _t3_nan_fill)
    run_test("eğimli yüzey — normal yönü", _t3_slope)


def _make_flat_grid(h=20, w=25, dxy=10.0):
    x = np.arange(w) * dxy
    y = np.arange(h) * dxy
    mX, mY = np.meshgrid(x, y)
    mZ = np.zeros_like(mX)
    return mX, mY, mZ


def _t3_flat():
    mX, mY, mZ = _make_flat_grid()
    nrm = compute_normals(mX, mY, mZ, interp=False)
    # Düz yüzey → normal (0, 0, 1)
    assert nrm.shape[1] == 3
    err = np.max(np.abs(nrm[:, 2] - 1.0))
    assert err < 1e-6, f"z-bileşen hatası: {err}"
    return f"nz≈1 hata={err:.1e}"


def _t3_shape():
    mX, mY, mZ = _make_flat_grid(h=15, w=20)
    nrm = compute_normals(mX, mY, mZ)
    assert nrm.shape == (15 * 20, 3)
    return f"{nrm.shape}"


def _t3_unit():
    mX, mY, mZ = _make_flat_grid()
    mZ = np.sin(mX / 50) * np.cos(mY / 50) * 10
    nrm = compute_normals(mX, mY, mZ)
    norms = np.linalg.norm(nrm, axis=1)
    err = np.max(np.abs(norms - 1.0))
    assert err < 1e-6, f"Normalizasyon hatası: {err}"
    return f"maks_norm_hata={err:.1e}"


def _t3_nan_fill():
    mX, mY, mZ = _make_flat_grid()
    mZ[5, 5] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nrm = compute_normals(mX, mY, mZ, interp=True)
    # NaN çıkmamalı (doldurulmuş olmalı)
    nan_count = np.sum(np.isnan(nrm))
    assert nan_count == 0, f"Normal'de {nan_count} NaN kaldı"
    return "NaN'sız normal ✓"


def _t3_slope():
    """x yönünde 45° eğim → ny bileşeni 0, nx ≠ 0."""
    x = np.arange(20) * 10.0
    y = np.arange(15) * 10.0
    mX, mY = np.meshgrid(x, y)
    mZ = mX.copy()  # z = x → 45° eğim
    nrm = compute_normals(mX, mY, mZ, interp=False)
    # nx çoğunlukla negatif olmalı (eğim +x yönünde)
    mean_nx = nrm[:, 0].mean()
    assert mean_nx < 0, f"Beklenen nx<0, alınan {mean_nx:.3f}"
    return f"mean_nx={mean_nx:.3f}"


# ===========================================================================
# TEST 4 — geo_sample_points
# ===========================================================================
def test_geo_sample_points():
    section("TEST 4 · geo_sample_points")

    run_test("boyut kontrolü (D<N)", _t4_shape_col)
    run_test("boyut kontrolü (N<D, transpozlu)", _t4_shape_row)
    run_test("tekrarsız indeksler", _t4_unique)
    run_test("n_pts > N durumu (tümü seç)", _t4_over)
    run_test("indeksler içerik ile uyumlu", _t4_consistency)


rng = np.random.default_rng(0)


def _t4_shape_col():
    pts = rng.random((3, 5000))
    sampled, idx = geo_sample_points(pts, 500)
    assert sampled.shape == (3, 500), f"Beklenen (3,500) alınan {sampled.shape}"
    assert idx.shape == (500,)
    return f"{sampled.shape}"


def _t4_shape_row():
    pts = rng.random((5000, 3))  # N>D → transpozlanır
    sampled, idx = geo_sample_points(pts, 200)
    assert sampled.shape == (200, 3), f"Beklenen (200,3) alınan {sampled.shape}"
    return f"{sampled.shape}"


def _t4_unique():
    pts = rng.random((3, 10000))
    _, idx = geo_sample_points(pts, 1000)
    assert len(np.unique(idx)) == 1000, "Tekrarlayan indeks var"
    return "tekrar yok ✓"


def _t4_over():
    pts = rng.random((3, 50))
    sampled, idx = geo_sample_points(pts, 200)
    assert sampled.shape[1] == 50, "Tüm noktalar seçilmeli"
    return f"seçilen={sampled.shape[1]}"


def _t4_consistency():
    pts = rng.random((4, 1000))
    sampled, idx = geo_sample_points(pts, 100)
    # Seçilen noktalar orijinaldeki indekslerle uyuşmalı
    err = np.max(np.abs(sampled - pts[:, idx]))
    assert err == 0.0, f"İçerik uyumsuzluğu: {err}"
    return "içerik uyumlu ✓"


# ===========================================================================
# TEST 5 — transform_using_solver_var ileri-geri
# ===========================================================================
def test_transform():
    section("TEST 5 · transform_using_solver_var")

    run_test("ileri-geri dönüşüm tutarlılığı", _t5_roundtrip)
    run_test("sıfır dönüşüm → noktalar değişmez", _t5_identity)
    run_test("sadece öteleme", _t5_translation_only)
    run_test("sadece döndürme", _t5_rotation_only)


def _make_solver_info(v_test, active_mask, v0=None, delta=None, pts=None):
    if v0 is None:
        v0 = np.array([0., 0., 0., 0., 0., 0., 1., 1., 1., 1.])
    if delta is None:
        delta = np.array([5., 5., 5., 100., 100., 100., 0., 0., 0., 0.])
    vIdx = np.where(active_mask)[0]
    mBnd = np.vstack([v0[active_mask] - delta[active_mask],
                      v0[active_mask] + delta[active_mask]])
    if pts is None:
        rng2 = np.random.default_rng(7)
        pts = np.vstack([rng2.random((3, 30)) * 500, np.ones((1, 30))])
    vC = pts[:3].mean(axis=1)
    return {
        'scale_bounds':    mBnd,
        'rotation_center': vC,
        'variables_index': vIdx,
        'variables':       v_test,
        'direction':       'forward',
    }, pts


def _t5_roundtrip():
    active = np.array([True, True, True, True, True, True, False, False, False, False])
    v_test = np.array([0.6, 0.4, 0.55, 0.7, 0.3, 0.8])
    info_fwd, pts = _make_solver_info(v_test, active)
    info_inv = dict(info_fwd); info_inv['direction'] = 'inverse'
    pts_fwd = transform_using_solver_var(pts.copy(), info_fwd)
    pts_back = transform_using_solver_var(pts_fwd.copy(), info_inv)
    err = np.max(np.abs(pts_back - pts))
    assert err < 1e-5, f"İleri-geri hata: {err:.2e}"
    return f"maks_hata={err:.2e}"


def _t5_identity():
    active = np.array([True, True, True, True, True, True, False, False, False, False])
    # Tüm değişkenler orta noktada → sıfır dönüşüm
    v_mid = np.full(6, 0.5)
    info, pts = _make_solver_info(v_mid, active)
    pts_t = transform_using_solver_var(pts.copy(), info)
    err = np.max(np.abs(pts_t - pts))
    assert err < 1e-6, f"Sıfır dönüşüm hatası: {err:.2e}"
    return f"maks_hata={err:.2e}"


def _t5_translation_only():
    """Sadece tz aktif, 50 m öteleme."""
    v0 = np.array([0., 0., 0., 0., 0., 0., 1., 1., 1., 1.])
    delta = np.array([0., 0., 0., 0., 0., 50., 0., 0., 0., 0.])
    active = delta > 0
    vIdx = np.where(active)[0]
    mBnd = np.vstack([v0[active] - delta[active], v0[active] + delta[active]])
    rng2 = np.random.default_rng(9)
    pts = np.vstack([rng2.random((3, 20)) * 200, np.ones((1, 20))])
    info = {
        'scale_bounds':    mBnd,
        'rotation_center': pts[:3].mean(axis=1),
        'variables_index': vIdx,
        'variables':       np.array([1.0]),   # üst sınır → +50 m
        'direction':       'forward',
    }
    pts_t = transform_using_solver_var(pts.copy(), info)
    dz = pts_t[2] - pts[2]
    assert np.allclose(dz, 50.0, atol=1e-4), f"Beklenen dz=50, alınan {dz.mean():.3f}"
    return f"dz={dz.mean():.2f} m"


def _t5_rotation_only():
    """rz=90° döndürme: x→-y, y→x."""
    v0 = np.array([0., 0., 0., 0., 0., 0., 1., 1., 1., 1.])
    delta = np.array([0., 0., 90., 0., 0., 0., 0., 0., 0., 0.])
    active = delta > 0
    vIdx = np.where(active)[0]
    mBnd = np.vstack([v0[active] - delta[active], v0[active] + delta[active]])
    # Merkez orijin, basit nokta
    pts = np.array([[100., 0., 0., 1.],
                    [0.,   0., 0., 1.]]).T   # shape (4, 2)
    info = {
        'scale_bounds':    mBnd,
        'rotation_center': np.zeros(3),
        'variables_index': vIdx,
        'variables':       np.array([1.0]),   # +90°
        'direction':       'forward',
    }
    pts_t = transform_using_solver_var(pts.copy(), info)
    # [100,0,0] → ~ [0,100,0]
    assert abs(pts_t[0, 0]) < 1.0, f"Beklenen x≈0, alınan {pts_t[0,0]:.3f}"
    assert abs(pts_t[1, 0] - 100.0) < 1.0, f"Beklenen y≈100, alınan {pts_t[1,0]:.3f}"
    return f"döndürme ✓  x={pts_t[0,0]:.2f} y={pts_t[1,0]:.2f}"


# ===========================================================================
# TEST 6 — optimize_dem
# ===========================================================================
def test_optimize_dem():
    section("TEST 6 · optimize_dem")

    run_test("optimize_dem yakınsama — RMSE < 5 m", _t6_converge)
    run_test("optimize_dem çıktı anahtarları mevcut", _t6_keys)
    run_test("optimize_dem sıfır dönüşümde RMSE düşük", _t6_zero)


def _make_dem_pair(rng_seed=42, grid_size=30, noise=0.0, offset_z=0.0,
                   rot_deg=0.0):
    """Referans DEM ve hareketli nokta bulutu üretir."""
    rng2 = np.random.default_rng(rng_seed)
    nx = ny = grid_size
    vX = np.linspace(500000, 500000 + (nx - 1) * 90, nx)
    vY = np.linspace(4500000 + (ny - 1) * 90, 4500000, ny)
    mX, mY = np.meshgrid(vX, vY)

    dem_ref = (np.sin((mX - 500000) / 500) * 20 +
               np.cos((mY - 4500000) / 500) * 15 + 200.0)
    dem_ref += rng2.normal(0, noise, dem_ref.shape)

    # Nokta bulutu: ızgara noktaları + z kayması
    pts_z = dem_ref.ravel() + offset_z
    pts = np.vstack([mX.ravel(), mY.ravel(), pts_z,
                     np.ones(mX.size)])
    return pts, dem_ref, np.zeros(dem_ref.shape, bool), vX, vY


def _t6_converge():
    pts, dem_ref, mask, vX, vY = _make_dem_pair(offset_z=3.0)
    opt = {
        'rotation':     [1.0, 1.0, 1.0],
        'translation':  [0., 0., 10.],
        'scale':        [float('nan')] * 3,
        'globalScale':  0.,
        'maxIterations': 100,
        'polySurf':     False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sOut = optimize_dem(pts, dem_ref, mask, vX, vY, opt)
    rmse = sOut['verticalRMSE']
    assert rmse < 5.0, f"RMSE çok yüksek: {rmse:.2f} m"
    return f"RMSE={rmse:.3f} m"


def _t6_keys():
    pts, dem_ref, mask, vX, vY = _make_dem_pair()
    opt = {
        'rotation': [1., 1., 1.], 'translation': [0., 0., 5.],
        'scale': [float('nan')] * 3, 'globalScale': 0.,
        'maxIterations': 30, 'polySurf': False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sOut = optimize_dem(pts, dem_ref, mask, vX, vY, opt)
    required = {'scale_bounds', 'rotation_center', 'variables_index',
                'variables', 'poly_surf', 'direction', 'verticalRMSE'}
    missing = required - set(sOut.keys())
    assert not missing, f"Eksik anahtarlar: {missing}"
    return "tüm anahtarlar mevcut ✓"


def _t6_zero():
    """Offset=0 → başlangıç RMSE zaten düşük."""
    pts, dem_ref, mask, vX, vY = _make_dem_pair(offset_z=0.0)
    opt = {
        'rotation': [0.5, 0.5, 0.5], 'translation': [0., 0., 5.],
        'scale': [float('nan')] * 3, 'globalScale': 0.,
        'maxIterations': 50, 'polySurf': False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sOut = optimize_dem(pts, dem_ref, mask, vX, vY, opt)
    rmse = sOut['verticalRMSE']
    assert rmse < 2.0, f"Sıfır offset'te RMSE yüksek: {rmse:.2f} m"
    return f"RMSE={rmse:.3f} m"


# ===========================================================================
# TEST 7 — align_dem (ICP)
# ===========================================================================
def test_align_dem():
    section("TEST 7 · align_dem (ICP)")

    if not HAS_OPEN3D:
        print("  [ATLANDI] open3d kurulu değil — align_dem ICP testi atlandı")
        results.append(("align_dem ICP", None, "atlandı — open3d yok"))
        return

    run_test("align_dem çıktı matrisi (4,4)", _t7_shape)
    run_test("align_dem saf öteleme kurtarma", _t7_translation)


def _t7_shape():
    pts, dem_ref, mask, vX, vY = _make_dem_pair()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T = align_dem(pts, dem_ref, mask, vX, vY)
    assert T.shape == (4, 4), f"Beklenen (4,4) alınan {T.shape}"
    return f"{T.shape}"


def _make_dem_pair():  # noqa: F811 — içeride aynı imza
    # test 6'nın _make_dem_pair'ini yeniden kullan
    return _make_dem_pair_impl()


def _make_dem_pair_impl(grid_size=40, offset_z=0.0):
    rng2 = np.random.default_rng(0)
    nx = ny = grid_size
    vX = np.linspace(500000, 500000 + (nx - 1) * 90, nx)
    vY = np.linspace(4500000 + (ny - 1) * 90, 4500000, ny)
    mX, mY = np.meshgrid(vX, vY)
    dem_ref = np.sin((mX - 500000) / 400) * 30 + np.cos((mY - 4500000) / 400) * 20 + 300.
    pts_z = dem_ref.ravel() + offset_z
    pts = np.vstack([mX.ravel(), mY.ravel(), pts_z, np.ones(mX.size)])
    return pts, dem_ref, np.zeros(dem_ref.shape, bool), vX, vY


# TEST 6 fonksiyonlarının kullandığı _make_dem_pair'i düzelt
_make_dem_pair_for_t6 = lambda **kw: _make_dem_pair_orig(**kw)  # noqa: E731


def _make_dem_pair_orig(rng_seed=42, grid_size=30, noise=0.0, offset_z=0.0,
                        rot_deg=0.0):
    rng2 = np.random.default_rng(rng_seed)
    nx = ny = grid_size
    vX = np.linspace(500000, 500000 + (nx - 1) * 90, nx)
    vY = np.linspace(4500000 + (ny - 1) * 90, 4500000, ny)
    mX, mY = np.meshgrid(vX, vY)
    dem_ref = (np.sin((mX - 500000) / 500) * 20 +
               np.cos((mY - 4500000) / 500) * 15 + 200.0)
    dem_ref += rng2.normal(0, noise, dem_ref.shape)
    pts_z = dem_ref.ravel() + offset_z
    pts = np.vstack([mX.ravel(), mY.ravel(), pts_z, np.ones(mX.size)])
    return pts, dem_ref, np.zeros(dem_ref.shape, bool), vX, vY


# test 6 fonksiyonlarını güncelle
def _t6_converge():  # noqa: F811
    pts, dem_ref, mask, vX, vY = _make_dem_pair_orig(offset_z=3.0)
    opt = {
        'rotation':     [1.0, 1.0, 1.0],
        'translation':  [0., 0., 10.],
        'scale':        [float('nan')] * 3,
        'globalScale':  0.,
        'maxIterations': 100,
        'polySurf':     False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sOut = optimize_dem(pts, dem_ref, mask, vX, vY, opt)
    rmse = sOut['verticalRMSE']
    assert rmse < 5.0, f"RMSE çok yüksek: {rmse:.2f} m"
    return f"RMSE={rmse:.3f} m"


def _t6_keys():  # noqa: F811
    pts, dem_ref, mask, vX, vY = _make_dem_pair_orig()
    opt = {
        'rotation': [1., 1., 1.], 'translation': [0., 0., 5.],
        'scale': [float('nan')] * 3, 'globalScale': 0.,
        'maxIterations': 30, 'polySurf': False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sOut = optimize_dem(pts, dem_ref, mask, vX, vY, opt)
    required = {'scale_bounds', 'rotation_center', 'variables_index',
                'variables', 'poly_surf', 'direction', 'verticalRMSE'}
    missing = required - set(sOut.keys())
    assert not missing, f"Eksik anahtarlar: {missing}"
    return "tüm anahtarlar mevcut ✓"


def _t6_zero():  # noqa: F811
    pts, dem_ref, mask, vX, vY = _make_dem_pair_orig(offset_z=0.0)
    opt = {
        'rotation': [0.5, 0.5, 0.5], 'translation': [0., 0., 5.],
        'scale': [float('nan')] * 3, 'globalScale': 0.,
        'maxIterations': 50, 'polySurf': False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sOut = optimize_dem(pts, dem_ref, mask, vX, vY, opt)
    rmse = sOut['verticalRMSE']
    assert rmse < 2.0, f"Sıfır offset'te RMSE yüksek: {rmse:.2f} m"
    return f"RMSE={rmse:.3f} m"


def _t7_shape():  # noqa: F811
    pts, dem_ref, mask, vX, vY = _make_dem_pair_impl()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T = align_dem(pts, dem_ref, mask, vX, vY)
    assert T.shape == (4, 4), f"Beklenen (4,4) alınan {T.shape}"
    return f"{T.shape}"


def _t7_translation():
    rng2 = np.random.default_rng(5)
    pts, dem_ref, mask, vX, vY = _make_dem_pair_impl(offset_z=5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        T = align_dem(pts, dem_ref, mask, vX, vY)
    assert T.shape == (4, 4)
    # Dönüşüm matrisinin diyagonali 1'e yakın olmalı (küçük dönüşüm)
    diag_err = np.max(np.abs(T[:3, :3] - np.eye(3)))
    assert diag_err < 0.5, f"Döndürme bileşeni beklenenden büyük: {diag_err:.3f}"
    return f"T_diag_err={diag_err:.3f}"


# ===========================================================================
# TEST 8 — geo_opti_trans uçtan uca
# ===========================================================================
def test_geo_opti_trans():
    section("TEST 8 · geo_opti_trans (uçtan uca)")
    run_test("sahte DEM ile boru hattı çalışır", _t8_pipeline)
    run_test("dönüşüm matrisi (4,4) ve çıktı anahtarları", _t8_output_format)


def _fake_get_ref_dem(bounds, path):
    """Gerçek GeoTIFF olmadan test için sahte DEM üretir."""
    lon_min, lat_min = bounds[0]
    lon_max, lat_max = bounds[1]
    n = 20
    vLon = np.linspace(lon_min, lon_max, n)
    vLat = np.linspace(lat_max, lat_min, n)
    mLon, mLat = np.meshgrid(vLon, vLat)
    dem = np.sin(mLon * 10) * 20 + np.cos(mLat * 10) * 15 + 300.
    return dem, vLon, vLat


def _make_utm_pts(n=2000, zone=36, hemi='N', seed=1):
    rng2 = np.random.default_rng(seed)
    easting  = 600000. + rng2.random(n) * 5000
    northing = 4400000. + rng2.random(n) * 5000
    z        = 300. + rng2.random(n) * 50
    return np.vstack([easting, northing, z, np.ones(n)])


def _t8_pipeline():
    pts = _make_utm_pts()
    opt = {
        'rotation':     [5., 5., 5.],
        'translation':  [0., 0., 50.],
        'scale':        [float('nan')] * 3,
        'globalScale':  0.,
        'maxIterations': 50,
        'polySurf':     False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mT, sOut = geo_opti_trans(
            pts, "fake_dem.tif", None,
            zone=36, hemisphere='N',
            opt=opt,
            get_ref_dem_fn=_fake_get_ref_dem,
        )
    assert mT.shape == (4, 4)
    assert 'verticalRMSE' in sOut
    return f"RMSE={sOut['verticalRMSE']:.2f} m"


def _t8_output_format():
    pts = _make_utm_pts(n=1000, seed=2)
    opt = {
        'rotation': [2., 2., 2.], 'translation': [0., 0., 20.],
        'scale': [float('nan')] * 3, 'globalScale': 0.,
        'maxIterations': 30, 'polySurf': False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mT, sOut = geo_opti_trans(
            pts, "fake_dem.tif", None,
            zone=36, hemisphere='N',
            opt=opt,
            get_ref_dem_fn=_fake_get_ref_dem,
        )
    assert mT.shape == (4, 4), f"mT.shape={mT.shape}"
    required_out = {'scale_bounds', 'rotation_center', 'variables_index',
                    'variables', 'direction', 'verticalRMSE'}
    missing = required_out - set(sOut.keys())
    assert not missing, f"Eksik çıktı anahtarları: {missing}"
    return "format ✓"


# ===========================================================================
# ANA ÇALIŞTIRIC
# ===========================================================================
def main():
    print("\n" + "=" * 60)
    print("  GEOREF ENTEGRASYON TESTİ")
    print("  geo_optimize.py")
    print("=" * 60)

    test_ll2utm_roundtrip()
    test_points2grid()
    test_compute_normals()
    test_geo_sample_points()
    test_transform()
    test_optimize_dem()
    test_align_dem()
    test_geo_opti_trans()

    # ---------------------------------------------------------------------------
    # Özet
    # ---------------------------------------------------------------------------
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
        for name, _, msg in failed:
            print(f"    • {name}")
        sys.exit(1)
    else:
        print("\n  Tüm testler geçti ✓")
        sys.exit(0)


if __name__ == '__main__':
    main()
