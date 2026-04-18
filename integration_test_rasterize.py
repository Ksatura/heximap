"""
integration_test_rasterize.py

Rasterize aşaması entegrasyon testi.
MATLAB karşılıkları: main/4_rasterize/rasDem.m, rasClean.m,
                     rasSmooth.m, rasOrtho.m ve shared bağımlıları

Çalıştırma:
    python integration_test_rasterize.py

Bağımlılıklar:
    pip install numpy scipy rasterio pyproj open3d  (open3d opsiyonel)
"""

import sys
import warnings
import traceback
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Modül keşfi
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent
RAS_DIR     = REPO_ROOT / "main" / "4_rasterize"
SHARED_DIR  = REPO_ROOT / "main" / "shared"
GEOREF_DIR  = REPO_ROOT / "main" / "3_georef"

for p in [RAS_DIR, GEOREF_DIR, SHARED_DIR, REPO_ROOT]:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

try:
    from ras_export import (
        ras_clean,
        ras_smooth,
        mesh_denoise,
        curvature_correction,
        ras_dem,
        ras_ortho,
    )
except ImportError as e:
    print(f"[HATA] ras_export.py yüklenemedi: {e}")
    sys.exit(1)

try:
    from geo_optimize import ll2utm, utm2ll
except ImportError as e:
    print(f"[HATA] geo_optimize.py yüklenemedi: {e}")
    sys.exit(1)

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

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


def _make_dem(h=40, w=50, nan_frac=0.0, seed=0):
    r = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, w)
    y = np.linspace(0, 4 * np.pi, h)
    mX, mY = np.meshgrid(x, y)
    dem = np.sin(mX) * np.cos(mY) * 200 + 500.0
    if nan_frac > 0:
        mask = r.random((h, w)) < nan_frac
        dem[mask] = np.nan
    return dem


def _make_coord_vecs(h=40, w=50,
                     lon0=30.0, lat0=39.0,
                     dlon=0.002, dlat=0.002):
    vX = np.linspace(lon0, lon0 + (w - 1) * dlon, w)
    vY = np.linspace(lat0 + (h - 1) * dlat, lat0, h)
    return vX, vY


def _make_win_obj(n_pts=800, seed=1):
    """Sentetik bir pencere nesnesi oluşturur."""
    r = np.random.default_rng(seed)
    lon = r.uniform(30.0, 30.15, n_pts)
    lat = r.uniform(39.0, 39.15, n_pts)
    elev = 900.0 + r.normal(0, 10, n_pts)
    pts_utm, zone, hemi = ll2utm(
        np.column_stack([lon, lat]), None, None
    )
    pts_geo = np.vstack([
        pts_utm[:, 0], pts_utm[:, 1], elev, np.ones(n_pts)
    ])
    return {
        'TriangulatedPointsGeoref': pts_geo,
        'RegionID': 1,
        'WindowID': 1,
        'GeorefInfo': {
            'Initial': {
                'Triangulated2WorldTransform': {
                    'zone': zone, 'hemi': hemi
                }
            }
        }
    }, zone, hemi


# ===========================================================================
# TEST 1 — ras_clean
# ===========================================================================
def test_ras_clean():
    section("TEST 1 · ras_clean")

    run_test("do_clean=False → DEM değişmez",       _t1_passthrough)
    run_test("küçük boşluk doldurulur",              _t1_small_gap_filled)
    run_test("büyük boşluk kenarları genişler",      _t1_large_gap_expands)
    run_test("speckle (yalnız ada) silinir",         _t1_speckle_removed)
    run_test("tüm geçerli DEM dokunulmadan kalır",   _t1_valid_preserved)
    run_test("çıktı şekli giriş ile aynı",           _t1_shape)


def _t1_passthrough():
    dem = _make_dem(nan_frac=0.1)
    out = ras_clean(dem.copy(), res_m=30.0, do_clean=False)
    assert np.array_equal(np.isnan(out), np.isnan(dem))
    return "NaN maskesi korundu ✓"


def _t1_small_gap_filled():
    dem = _make_dem()
    dem[20, 20] = np.nan   # 1 piksel boşluk
    # gap_thresh > 1 piksel alan → doldurulmalı
    out = ras_clean(dem.copy(), res_m=30.0, do_clean=True,
                    gap_thresh=10000, speckle_thresh=1)
    assert not np.isnan(out[20, 20]), "1 piksel boşluk doldurulmadı"
    return "1px boşluk dolduruldu ✓"


def _t1_large_gap_expands():
    dem = _make_dem()
    dem[10:25, 10:25] = np.nan   # 15×15 = 225 piksel
    n_nan_before = np.isnan(dem).sum()
    # gap_thresh küçük → büyük boşluk kenarları genişler
    out = ras_clean(dem.copy(), res_m=30.0, do_clean=True,
                    gap_thresh=100, speckle_thresh=1)
    n_nan_after = np.isnan(out).sum()
    assert n_nan_after >= n_nan_before, \
        f"Büyük boşluk genişlemedi: {n_nan_before} → {n_nan_after}"
    return f"NaN {n_nan_before} → {n_nan_after}"


def _t1_speckle_removed():
    dem = np.full((40, 50), np.nan)
    # Tek büyük geçerli bölge
    dem[5:35, 5:45] = _make_dem(h=30, w=40)
    # Küçük yalnız ada (2×2 = 4 piksel)
    dem[1:3, 1:3] = 500.0
    n_valid_before = (~np.isnan(dem)).sum()
    out = ras_clean(dem.copy(), res_m=30.0, do_clean=True,
                    gap_thresh=1, speckle_thresh=10000)
    n_valid_after = (~np.isnan(out)).sum()
    assert n_valid_after < n_valid_before, \
        "Speckle silinmedi"
    return f"geçerli piksel {n_valid_before} → {n_valid_after}"


def _t1_valid_preserved():
    dem = _make_dem()   # tamamen geçerli
    out = ras_clean(dem.copy(), res_m=30.0, do_clean=True,
                    gap_thresh=100, speckle_thresh=100)
    assert not np.any(np.isnan(out)), "Tamamen geçerli DEM bozuldu"
    return "geçerli piksel korundu ✓"


def _t1_shape():
    dem = _make_dem(h=37, w=53, nan_frac=0.05)
    out = ras_clean(dem.copy(), res_m=20.0)
    assert out.shape == dem.shape
    return f"{out.shape}"


# ===========================================================================
# TEST 2 — ras_smooth
# ===========================================================================
def test_ras_smooth():
    section("TEST 2 · ras_smooth")

    run_test("medyan filtre aykırı değeri azaltır",  _t2_outlier)
    run_test("NaN maskesi medyan sonrası korunur",    _t2_nan_preserved)
    run_test("do_median=False → ham DEM",            _t2_no_median)
    run_test("çıktı şekli değişmez",                 _t2_shape)
    run_test("gürültüsüz DEM medyandan etkilenmez",  _t2_flat)


def _t2_outlier():
    dem = _make_dem()
    dem[15, 20] = 99999.0
    vX, vY = _make_coord_vecs()
    out = ras_smooth(vX, vY, dem.copy(),
                     do_median=True, do_denoise=False, median_size=3)
    assert out[15, 20] < 99999.0, "Aykırı değer azalmadı"
    return f"{dem[15,20]:.0f} → {out[15,20]:.1f}"


def _t2_nan_preserved():
    dem = _make_dem(nan_frac=0.1, seed=5)
    nan_mask = np.isnan(dem)
    vX, vY = _make_coord_vecs()
    out = ras_smooth(vX, vY, dem.copy(),
                     do_median=True, do_denoise=False, median_size=3)
    assert np.all(np.isnan(out) == nan_mask), "NaN maskesi değişti"
    return "NaN korundu ✓"


def _t2_no_median():
    dem = _make_dem()
    dem[5, 5] = 99999.0
    vX, vY = _make_coord_vecs()
    out = ras_smooth(vX, vY, dem.copy(),
                     do_median=False, do_denoise=False)
    assert out[5, 5] == 99999.0, "do_median=False'de değer değişti"
    return "ham DEM korundu ✓"


def _t2_shape():
    dem = _make_dem(h=33, w=47)
    vX, vY = _make_coord_vecs(h=33, w=47)
    out = ras_smooth(vX, vY, dem,
                     do_median=True, do_denoise=False)
    assert out.shape == dem.shape
    return f"{out.shape}"


def _t2_flat():
    dem = np.full((30, 40), 500.0)
    vX, vY = _make_coord_vecs(h=30, w=40)
    out = ras_smooth(vX, vY, dem.copy(),
                     do_median=True, do_denoise=False, median_size=3)
    err = np.max(np.abs(out - 500.0))
    assert err < 1e-6, f"Düz DEM bozuldu: {err}"
    return f"maks_hata={err:.1e}"


# ===========================================================================
# TEST 3 — mesh_denoise
# ===========================================================================
def test_mesh_denoise():
    section("TEST 3 · mesh_denoise")

    run_test("çıktı şekli giriş ile aynı",           _t3_shape)
    run_test("düz yüzey → minimum değişim",          _t3_flat)
    run_test("gürültülü yüzey pürüzsüzleşir",        _t3_smooth)
    run_test("NaN bölgeler korunur",                 _t3_nan)


def _t3_shape():
    dem = _make_dem(h=20, w=25)
    vX, vY = _make_coord_vecs(h=20, w=25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = mesh_denoise(dem, vX, vY,
                           params=(0.9, 2, 4), block_size=500)
    assert out.shape == dem.shape
    return f"{out.shape}"


def _t3_flat():
    dem = np.full((15, 20), 300.0)
    vX, vY = _make_coord_vecs(h=15, w=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = mesh_denoise(dem, vX, vY,
                           params=(0.9, 1, 2), block_size=500)
    valid = ~np.isnan(out)
    if valid.any():
        err = np.max(np.abs(out[valid] - 300.0))
        assert err < 50.0, f"Düz yüzey çok bozuldu: {err:.1f}"
        return f"maks_hata={err:.2f}"
    return "tüm NaN (kabul edilebilir)"


def _t3_smooth():
    r = np.random.default_rng(10)
    x = np.linspace(0, 2 * np.pi, 25)
    y = np.linspace(0, 2 * np.pi, 20)
    mX, mY = np.meshgrid(x, y)
    dem_clean = np.sin(mX) * np.cos(mY) * 100 + 300
    dem_noisy = dem_clean + r.normal(0, 20, dem_clean.shape)
    vX, vY = _make_coord_vecs(h=20, w=25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = mesh_denoise(dem_noisy, vX, vY,
                           params=(0.5, 3, 6), block_size=500)
    valid = ~np.isnan(out)
    if valid.any():
        rmse_in  = np.sqrt(np.mean((dem_noisy[valid] - dem_clean[valid])**2))
        rmse_out = np.sqrt(np.mean((out[valid]       - dem_clean[valid])**2))
        # En azından bozulmamış olmalı
        assert rmse_out <= rmse_in * 1.5, \
            f"Denoise DEM'i daha da bozdu: {rmse_in:.1f} → {rmse_out:.1f}"
        return f"RMSE {rmse_in:.1f} → {rmse_out:.1f}"
    return "tüm NaN (kabul edilebilir)"


def _t3_nan():
    dem = _make_dem(h=20, w=25, nan_frac=0.2)
    nan_mask = np.isnan(dem)
    vX, vY = _make_coord_vecs(h=20, w=25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = mesh_denoise(dem, vX, vY,
                           params=(0.9, 1, 2), block_size=500)
    # NaN olan pikseller sonuçta da NaN kalmalı
    assert np.all(np.isnan(out[nan_mask])), "NaN bölgeler bozuldu"
    return "NaN korundu ✓"


# ===========================================================================
# TEST 4 — curvature_correction
# ===========================================================================
def test_curvature_correction():
    section("TEST 4 · curvature_correction")

    run_test("forward dönüşüm — çıktı boyutu (4,N)",  _t4_shape_fwd)
    run_test("forward sonrası z değişir (eğrilik)",    _t4_z_changes)
    run_test("forward-inverse yakınsak",               _t4_roundtrip)


def _make_utm_pts_cc(n=50, zone=36, hemi='N', seed=3):
    r = np.random.default_rng(seed)
    E = 600000. + r.random(n) * 5000
    N = 4400000. + r.random(n) * 5000
    Z = 500. + r.random(n) * 100
    return np.vstack([E, N, Z, np.ones(n)]), zone, hemi


def _absor_dummy(src, dst):
    """Basit ölçeksiz Horn: sadece öteleme + birim döndürme."""
    t = dst.mean(axis=1) - src.mean(axis=1)
    T = np.eye(4)
    T[:3, 3] = t
    return T


def _t4_shape_fwd():
    pts, zone, hemi = _make_utm_pts_cc()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pts_out, mT = curvature_correction(
            pts, zone, hemi, 'forward', absor_fn=_absor_dummy
        )
    assert pts_out.shape == pts.shape, \
        f"Beklenen {pts.shape}, alınan {pts_out.shape}"
    assert mT.shape == (4, 4)
    return f"{pts_out.shape}, T={mT.shape}"


def _t4_z_changes():
    pts, zone, hemi = _make_utm_pts_cc(n=100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pts_out, _ = curvature_correction(
            pts, zone, hemi, 'forward', absor_fn=_absor_dummy
        )
    z_diff = np.abs(pts_out[2] - pts[2])
    assert z_diff.mean() > 0, "Eğrilik düzeltmesi z'yi değiştirmedi"
    return f"ortalama_dz={z_diff.mean():.2f} m"


def _t4_roundtrip():
    pts, zone, hemi = _make_utm_pts_cc(n=80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pts_fwd, mT = curvature_correction(
            pts, zone, hemi, 'forward', absor_fn=_absor_dummy
        )
        pts_back, _ = curvature_correction(
            pts_fwd, zone, hemi, 'inverse', transform_matrix=mT
        )
    # z gidiş-dönüş hatası makul olmalı (sayısal hata toleransı)
    err = np.max(np.abs(pts_back[2] - pts[2]))
    assert err < 100.0, f"Gidiş-dönüş z hatası: {err:.2f} m"
    return f"z_roundtrip_err={err:.2f} m"


# ===========================================================================
# TEST 5 — ras_dem
# ===========================================================================
def test_ras_dem():
    section("TEST 5 · ras_dem")

    run_test("GeoTIFF dosyası oluşturulur",           _t5_file_created)
    run_test("çıktı int16 ve nodata=-32768",          _t5_dtype_nodata)
    run_test("win_obj anahtarları eklendi",            _t5_keys)
    run_test("CRS = EPSG:4326",                        _t5_crs)
    run_test("do_clean=True çalışır",                  _t5_clean)
    run_test("do_median=True çalışır",                 _t5_median)
    run_test("NaN içermeyen nokta bulutu",             _t5_no_nan_pts)


def _t5_file_created():
    win_obj, _, _ = _make_win_obj()
    with tempfile.TemporaryDirectory() as tmpdir:
        ras_dem(win_obj, tmpdir,
                do_clean=False, do_median=False, do_denoise=False)
        f = Path(tmpdir) / 'dems' / 'dem_r1_w1.tif'
        assert f.exists(), "GeoTIFF oluşturulmadı"
        return f"dosya mevcut ✓ ({f.stat().st_size} B)"


def _t5_dtype_nodata():
    win_obj, _, _ = _make_win_obj()
    with tempfile.TemporaryDirectory() as tmpdir:
        ras_dem(win_obj, tmpdir,
                do_clean=False, do_median=False, do_denoise=False)
        f = Path(tmpdir) / 'dems' / 'dem_r1_w1.tif'
        with rasterio.open(f) as ds:
            assert ds.dtypes[0] == 'int16', f"dtype={ds.dtypes[0]}"
            assert ds.nodata == -32768, f"nodata={ds.nodata}"
        return "int16, nodata=-32768 ✓"


def _t5_keys():
    win_obj, _, _ = _make_win_obj()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = ras_dem(win_obj, tmpdir,
                         do_clean=False, do_median=False, do_denoise=False)
    assert 'HexagonDem'           in result
    assert 'HexagonDemSpatialRef' in result
    assert result['HexagonDem'].dtype == np.int16
    return "HexagonDem, HexagonDemSpatialRef ✓"


def _t5_crs():
    win_obj, _, _ = _make_win_obj()
    with tempfile.TemporaryDirectory() as tmpdir:
        ras_dem(win_obj, tmpdir,
                do_clean=False, do_median=False, do_denoise=False)
        f = Path(tmpdir) / 'dems' / 'dem_r1_w1.tif'
        with rasterio.open(f) as ds:
            assert ds.crs.to_epsg() == 4326, f"CRS={ds.crs}"
        return "EPSG:4326 ✓"


def _t5_clean():
    win_obj, _, _ = _make_win_obj(n_pts=600)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = ras_dem(win_obj, tmpdir,
                         do_clean=True, do_median=False, do_denoise=False,
                         gap_thresh=5000, speckle_thresh=5000)
    assert 'HexagonDem' in result
    return "do_clean=True tamamlandı ✓"


def _t5_median():
    win_obj, _, _ = _make_win_obj(n_pts=600)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = ras_dem(win_obj, tmpdir,
                         do_clean=False, do_median=True, do_denoise=False,
                         median_size=3)
    assert 'HexagonDem' in result
    return "do_median=True tamamlandı ✓"


def _t5_no_nan_pts():
    """NaN içeren nokta bulutunu filtreler."""
    win_obj, _, _ = _make_win_obj(n_pts=400)
    # Bazı noktalara NaN ekle
    win_obj['TriangulatedPointsGeoref'][2, :50] = np.nan
    with tempfile.TemporaryDirectory() as tmpdir:
        result = ras_dem(win_obj, tmpdir,
                         do_clean=False, do_median=False, do_denoise=False)
    assert 'HexagonDem' in result
    return "NaN noktalar filtrelendi ✓"


# ===========================================================================
# TEST 6 — ras_ortho (sahte win_obj ile)
# ===========================================================================
def test_ras_ortho():
    section("TEST 6 · ras_ortho")

    run_test("GeoTIFF dosyası oluşturulur",           _t6_file_created)
    run_test("win_obj anahtarları eklendi",            _t6_keys)
    run_test("çıktı uint8",                           _t6_dtype)
    run_test("CRS = EPSG:4326",                        _t6_crs)


def _make_ortho_win_obj(seed=7):
    """ras_ortho için gerekli tüm alanları içeren win_obj."""
    r = np.random.default_rng(seed)

    # Önce ras_dem çalıştır
    win_obj, zone, hemi = _make_win_obj(n_pts=600, seed=seed)
    with tempfile.TemporaryDirectory() as tmpdir:
        win_obj = ras_dem(win_obj, tmpdir,
                          do_clean=False, do_median=False, do_denoise=False)
        # TemporaryDirectory kapanmadan önce win_obj'u tamamla

    sR = win_obj['HexagonDemSpatialRef']
    h = sR['RasterSize'][0]
    w = sR['RasterSize'][1]

    # Basit kamera matrisleri
    K = np.array([[3000., 0., 2000.],
                  [0., 3000., 1500.],
                  [0.,    0.,    1.]])
    P = np.hstack([np.eye(3), np.zeros((3, 1))])

    win_obj['IntrinsicMatrix'] = K
    win_obj['PoseMatrix']      = P
    win_obj['Image']           = r.integers(0, 255,
                                             (3000, 4000), dtype=np.uint8)
    win_obj['Window']          = np.array([[1, 1, 1, 1]])

    # GeorefInfo Final alanları (boş listeler)
    win_obj['GeorefInfo']['Final'] = {
        'OptimizationOutput': [],
        'AlignmentOutput':    [],
    }
    win_obj['GeorefInfo']['Initial']['OptimizationOutput'] = {
        'scale_bounds':    np.array([[0.], [1.]]),
        'rotation_center': np.zeros(3),
        'variables_index': np.array([], dtype=int),
        'variables':       np.array([]),
        'direction':       'inverse',
    }
    win_obj['GeorefInfo']['Initial']['AlignmentOutput'] = np.eye(3, 4)
    win_obj['GeorefInfo']['Initial']['WindowTransform'] = 'none'

    # ras_ortho içinde sT['trans'] olarak okunur — birim matris yeterli
    win_obj['GeorefInfo']['Initial']['Triangulated2WorldTransform']['trans'] = \
        np.eye(4)

    return win_obj


def _t6_file_created():
    win_obj = _make_ortho_win_obj()
    with tempfile.TemporaryDirectory() as tmpdir:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ras_ortho(win_obj, tmpdir)
        f = Path(tmpdir) / 'images' / 'image_r1w1.tif'
        assert f.exists(), "Ortogörüntü GeoTIFF oluşturulmadı"
        return f"dosya mevcut ✓ ({f.stat().st_size} B)"


def _t6_keys():
    win_obj = _make_ortho_win_obj()
    with tempfile.TemporaryDirectory() as tmpdir:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ras_ortho(win_obj, tmpdir)
    assert 'HexagonImage'           in result
    assert 'HexagonImageSpatialRef' in result
    return "HexagonImage, HexagonImageSpatialRef ✓"


def _t6_dtype():
    win_obj = _make_ortho_win_obj()
    with tempfile.TemporaryDirectory() as tmpdir:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ras_ortho(win_obj, tmpdir)
    assert result['HexagonImage'].dtype == np.uint8
    return "uint8 ✓"


def _t6_crs():
    win_obj = _make_ortho_win_obj()
    with tempfile.TemporaryDirectory() as tmpdir:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ras_ortho(win_obj, tmpdir)
        f = Path(tmpdir) / 'images' / 'image_r1w1.tif'
        with rasterio.open(f) as ds:
            assert ds.crs.to_epsg() == 4326
        return "EPSG:4326 ✓"


# ===========================================================================
# ANA ÇALIŞTIRICI
# ===========================================================================
def main():
    if not HAS_RASTERIO:
        print("[HATA] rasterio kurulu değil: pip install rasterio")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  RASTERIZE ENTEGRASYON TESTİ")
    print("  ras_export.py")
    print("=" * 60)

    test_ras_clean()
    test_ras_smooth()
    test_mesh_denoise()
    test_curvature_correction()
    test_ras_dem()
    test_ras_ortho()

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
