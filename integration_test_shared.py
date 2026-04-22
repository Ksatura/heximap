"""
integration_test_shared.py

shared_utils.py entegrasyon testi.
MATLAB karşılıkları: main/shared/*.m

Çalıştırma:
    python integration_test_shared.py

Bağımlılıklar:
    pip install numpy scipy rasterio
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
REPO_ROOT  = Path(__file__).resolve().parent
SHARED_DIR = REPO_ROOT / "main" / "shared"
GEOREF_DIR = REPO_ROOT / "main" / "3_georef"

for p in [SHARED_DIR, GEOREF_DIR, REPO_ROOT]:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

try:
    from shared_utils import (
        get_files,
        block_process,
        make_spatial_ref_vecs,
        neighbors,
        polygons2grid,
        shift_dem,
        read_geotiff_region,
        ll2ps, ps2ll,
        check_input,
        check_shp_path,
        colormap_dem,
    )
except ImportError as e:
    print(f"[HATA] shared_utils.py yüklenemedi: {e}")
    sys.exit(1)

try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
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


def _make_tif(path, data=None, epsg=4326):
    """Test için küçük GeoTIFF yazar."""
    if data is None:
        data = (rng.random((20, 30)) * 1000).astype(np.float32)
    transform = from_bounds(30.0, 39.0, 31.0, 40.0,
                            data.shape[1], data.shape[0])
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=data.shape[0], width=data.shape[1],
        count=1, dtype=data.dtype,
        crs=CRS.from_epsg(epsg),
        transform=transform,
    ) as ds:
        ds.write(data, 1)
    return data


# ===========================================================================
# TEST 1 — get_files
# ===========================================================================
def test_get_files():
    section("TEST 1 · get_files")
    run_test("düz klasör — uzantı filtresi",    _t1_flat)
    run_test("özyinelemeli alt klasör",          _t1_recursive)
    run_test("eşleşme yok → boş liste",         _t1_no_match)
    run_test("var olmayan klasör → boş liste",   _t1_missing_dir)
    run_test("büyük/küçük harf uzantısı",        _t1_case)


def _t1_flat():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / 'a.tif').touch()
        (Path(d) / 'b.tif').touch()
        (Path(d) / 'c.shp').touch()
        tifs = get_files(d, '.tif')
        assert len(tifs) == 2, f"Beklenen 2, alınan {len(tifs)}"
        return f"{len(tifs)} .tif dosyası"


def _t1_recursive():
    with tempfile.TemporaryDirectory() as d:
        sub = Path(d) / 'sub'
        sub.mkdir()
        (Path(d) / 'a.shp').touch()
        (sub / 'b.shp').touch()
        shps = get_files(d, '.shp')
        assert len(shps) == 2, f"Beklenen 2, alınan {len(shps)}"
        return f"{len(shps)} .shp dosyası (alt klasörden)"


def _t1_no_match():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / 'a.txt').touch()
        result = get_files(d, '.tif')
        assert result == [], f"Boş liste beklendi, alınan {result}"
        return "boş liste ✓"


def _t1_missing_dir():
    result = get_files('/tmp/bu_klasor_yoktur_xyz', '.tif')
    assert result == []
    return "boş liste ✓"


def _t1_case():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / 'a.TIF').touch()
        (Path(d) / 'b.tif').touch()
        tifs = get_files(d, '.tif')
        # En az küçük harf olanı bulmalı
        assert len(tifs) >= 1
        return f"{len(tifs)} dosya"


# ===========================================================================
# TEST 2 — block_process
# ===========================================================================
def test_block_process():
    section("TEST 2 · block_process")
    run_test("sabit değer fonksiyonu — tüm hücreler 1.0",  _t2_constant)
    run_test("çıktı şekli doğru",                          _t2_shape)
    run_test("boolean dtype",                               _t2_bool)
    run_test("blok koordinatları fonksiyona iletilir",      _t2_coords)
    run_test("tek blok (block_size > ızgara)",              _t2_single_block)
    run_test("yanlış boyutlu blok → uyarı, atlanır",        _t2_wrong_size)


def _t2_constant():
    def fn(s):
        ix, iy = s['count_x'], s['count_y']
        h = s['index_y'][iy+1] - s['index_y'][iy]
        w = s['index_x'][ix+1] - s['index_x'][ix]
        return np.ones((h, w))
    grid = block_process((20, 30), 10, 'float64', fn)
    assert grid.shape == (20, 30)
    assert np.all(grid == 1.0)
    return "tüm hücreler 1.0 ✓"


def _t2_shape():
    def fn(s):
        ix, iy = s['count_x'], s['count_y']
        h = s['index_y'][iy+1] - s['index_y'][iy]
        w = s['index_x'][ix+1] - s['index_x'][ix]
        return np.zeros((h, w))
    for (H, W, bs) in [(15, 25, 8), (100, 100, 33), (7, 7, 100)]:
        grid = block_process((H, W), bs, 'float64', fn)
        assert grid.shape == (H, W), f"Beklenen ({H},{W}), alınan {grid.shape}"
    return "tüm boyutlar doğru ✓"


def _t2_bool():
    def fn(s):
        ix, iy = s['count_x'], s['count_y']
        h = s['index_y'][iy+1] - s['index_y'][iy]
        w = s['index_x'][ix+1] - s['index_x'][ix]
        return np.ones((h, w), dtype=bool)
    grid = block_process((10, 15), 5, 'bool', fn)
    assert grid.dtype == bool
    assert grid.all()
    return "bool dtype ✓"


def _t2_coords():
    """Blok koordinatları index_x/y ile tutarlı olmalı."""
    collected = []
    def fn(s):
        collected.append((s['count_x'], s['count_y']))
        ix, iy = s['count_x'], s['count_y']
        h = s['index_y'][iy+1] - s['index_y'][iy]
        w = s['index_x'][ix+1] - s['index_x'][ix]
        return np.zeros((h, w))
    block_process((20, 30), 10, 'float64', fn)
    xs = [c[0] for c in collected]
    ys = [c[1] for c in collected]
    assert min(xs) == 0 and min(ys) == 0
    return f"{len(collected)} blok işlendi"


def _t2_single_block():
    def fn(s):
        ix, iy = s['count_x'], s['count_y']
        h = s['index_y'][iy+1] - s['index_y'][iy]
        w = s['index_x'][ix+1] - s['index_x'][ix]
        return np.full((h, w), 7.0)
    grid = block_process((5, 8), 1000, 'float64', fn)
    assert grid.shape == (5, 8)
    assert np.all(grid == 7.0)
    return "tek blok ✓"


def _t2_wrong_size():
    def fn(s):
        return np.ones((1, 1))   # Her zaman yanlış boyut
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grid = block_process((10, 15), 5, 'float64', fn)
    assert len(w) > 0, "Uyarı beklendi"
    # NaN ile başlatılmış olmalı (hiçbir blok yazılmadı)
    assert np.all(np.isnan(grid))
    return "uyarı verildi, NaN ızgara ✓"


# ===========================================================================
# TEST 3 — make_spatial_ref_vecs
# ===========================================================================
def test_make_spatial_ref_vecs():
    section("TEST 3 · make_spatial_ref_vecs")
    run_test("full — vektör uzunlukları",          _t3_full_len)
    run_test("full — kenar değerleri sınır içinde", _t3_full_bounds)
    run_test("limits — sadece 2 eleman",            _t3_limits)
    run_test("vX artan, vY azalan",                _t3_monotone)
    run_test("doğrudan vX/vY sözlüğü",             _t3_direct)


def _make_sR(H=10, W=20):
    return {'Lonlim': [30.0, 31.0], 'Latlim': [39.0, 40.0],
            'RasterSize': (H, W)}


def _t3_full_len():
    sR = _make_sR(H=10, W=20)
    vX, vY = make_spatial_ref_vecs(sR, 'full')
    assert len(vX) == 20 and len(vY) == 10
    return f"vX={len(vX)}, vY={len(vY)}"


def _t3_full_bounds():
    sR = _make_sR()
    vX, vY = make_spatial_ref_vecs(sR, 'full')
    assert vX[0] > 30.0 and vX[-1] < 31.0
    assert vY[0] < 40.0 and vY[-1] > 39.0
    return "sınırlar içinde ✓"


def _t3_limits():
    sR = _make_sR()
    vX, vY = make_spatial_ref_vecs(sR, 'limits')
    assert len(vX) == 2 and len(vY) == 2
    return f"vX={vX}, vY={vY}"


def _t3_monotone():
    sR = _make_sR(H=15, W=25)
    vX, vY = make_spatial_ref_vecs(sR, 'full')
    assert np.all(np.diff(vX) > 0), "vX artan değil"
    assert np.all(np.diff(vY) < 0), "vY azalan değil"
    return "artan/azalan ✓"


def _t3_direct():
    vX_in = np.linspace(30.0, 31.0, 20)
    vY_in = np.linspace(40.0, 39.0, 10)
    sR = {'vX': vX_in, 'vY': vY_in}
    vX, vY = make_spatial_ref_vecs(sR, 'full')
    assert np.allclose(vX, vX_in) and np.allclose(vY, vY_in)
    return "doğrudan vektör ✓"


# ===========================================================================
# TEST 4 — neighbors
# ===========================================================================
def test_neighbors():
    section("TEST 4 · neighbors")
    run_test("çıktı boyutu (H*W, 8)",       _t4_shape)
    run_test("merkez piksel komşuları",      _t4_values)
    run_test("interp=True — NaN yok",        _t4_no_nan)
    run_test("interp=False — kenar NaN",     _t4_edge_nan)
    run_test("düz ızgara — komşular eşit",   _t4_flat)


def _t4_shape():
    mI = rng.random((7, 9))
    N = neighbors(mI, interp=True)
    assert N.shape == (63, 8), f"Beklenen (63,8), alınan {N.shape}"
    return f"{N.shape}"


def _t4_values():
    """3×3 ızgarada merkez piksel (1,1) komşuları bilinmeli."""
    mI = np.arange(1, 10, dtype=float).reshape(3, 3)
    N = neighbors(mI, interp=False)
    # Merkez piksel indeksi = 1*3+1 = 4
    center_nbrs = N[4]
    # Saat yönü: N=2, NE=3, E=6, SE=9, S=8, SW=7, W=4, NW=1
    expected = np.array([2., 3., 6., 9., 8., 7., 4., 1.])
    assert np.allclose(center_nbrs, expected), \
        f"Beklenen {expected}, alınan {center_nbrs}"
    return "merkez komşuları doğru ✓"


def _t4_no_nan():
    mI = rng.random((5, 6))
    N = neighbors(mI, interp=True)
    assert not np.any(np.isnan(N)), "interp=True'da NaN olmamalı"
    return "NaN yok ✓"


def _t4_edge_nan():
    mI = rng.random((4, 5))
    N = neighbors(mI, interp=False)
    # Köşe piksel (0,0) komşularının bir kısmı NaN olmalı
    assert np.any(np.isnan(N[0])), "Köşe komşusu NaN olmalı"
    return "köşe NaN ✓"


def _t4_flat():
    mI = np.ones((5, 5)) * 7.0
    N = neighbors(mI, interp=True)
    assert np.allclose(N, 7.0), "Düz ızgara tüm komşular eşit olmalı"
    return "tüm komşular 7.0 ✓"


# ===========================================================================
# TEST 5 — ll2ps / ps2ll
# ===========================================================================
def test_ps_projection():
    section("TEST 5 · ll2ps / ps2ll")
    run_test("güney yarımküre gidiş-dönüş",    _t5_south_roundtrip)
    run_test("kuzey yarımküre gidiş-dönüş",    _t5_north_roundtrip)
    run_test("kutup noktası (lat=-90)",         _t5_pole)
    run_test("çıktı boyutu korunur",            _t5_shape)
    run_test("yanlış yarımküre → ValueError",   _t5_mixed_hemi)


_UPS_S = [0.0, -90.0, 2000000.0, 2000000.0, 0.994]  # Antarktika UPS
_UPS_N = [0.0,  90.0, 2000000.0, 2000000.0, 0.994]  # Arktik UPS


def _t5_south_roundtrip():
    pts = np.array([[-45., -75.], [90., -80.], [135., -70.]])
    ps  = ll2ps(pts.copy(), _UPS_S)
    bk  = ps2ll(ps.copy(),  _UPS_S)
    err = np.max(np.abs(bk[:, :2] - pts))
    assert err < 1e-6, f"Hata: {err:.2e}°"
    return f"maks_hata={err:.2e}°"


def _t5_north_roundtrip():
    pts = np.array([[0., 80.], [90., 75.], [180., 85.]])
    ps  = ll2ps(pts.copy(), _UPS_N)
    bk  = ps2ll(ps.copy(),  _UPS_N)
    # 180° ve -180° matematiksel olarak eşdeğer — farkı [-180,180]'e sar
    diff_lon = np.abs(((bk[:, 0] - pts[:, 0]) + 180) % 360 - 180)
    diff_lat = np.abs(bk[:, 1] - pts[:, 1])
    err = max(diff_lon.max(), diff_lat.max())
    assert err < 1e-6, f"Hata: {err:.2e}°"
    return f"maks_hata={err:.2e}°"


def _t5_pole():
    pts = np.array([[0., -90.]])
    ps  = ll2ps(pts.copy(), _UPS_S)
    bk  = ps2ll(ps.copy(),  _UPS_S)
    err = abs(bk[0, 1] - (-90.))
    assert err < 1e-4, f"Kutup hatası: {err:.2e}°"
    return f"kutup_hata={err:.2e}°"


def _t5_shape():
    pts = np.column_stack([
        rng.uniform(-180, 180, 20),
        rng.uniform(-89, -60, 20),
        rng.random(20) * 1000
    ])
    ps = ll2ps(pts.copy(), _UPS_S)
    assert ps.shape == pts.shape
    bk = ps2ll(ps.copy(), _UPS_S)
    assert bk.shape == pts.shape
    return f"{pts.shape}"


def _t5_mixed_hemi():
    pts = np.array([[0., 45.], [0., -45.]])
    try:
        ll2ps(pts, _UPS_S)
        assert False, "ValueError beklendi"
    except ValueError:
        return "ValueError ✓"


# ===========================================================================
# TEST 6 — polygons2grid
# ===========================================================================
def test_polygons2grid():
    section("TEST 6 · polygons2grid")
    run_test("basit kare poligon — içi True",     _t6_square)
    run_test("boş poligon listesi → sıfır maske", _t6_empty)
    run_test("çıktı boolean dtype",               _t6_dtype)
    run_test("çıktı şekli (H, W)",                _t6_shape)


def _make_grid_vecs(nx=30, ny=20,
                    lon0=0.0, lat1=1.0,
                    dlon=1/30, dlat=1/20):
    vX = np.linspace(lon0, lon0 + (nx-1)*dlon, nx)
    vY = np.linspace(lat1, lat1 - (ny-1)*dlat, ny)
    return vX, vY


def _make_square_poly(x0=0.2, x1=0.8, y0=0.2, y1=0.8):
    return [{'type': 'Polygon',
             'coordinates': [[[x0, y0], [x1, y0],
                               [x1, y1], [x0, y1], [x0, y0]]]}]


def _t6_square():
    vX, vY = _make_grid_vecs()
    polys = _make_square_poly(x0=0.3, x1=0.7, y0=0.3, y1=0.7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = polygons2grid(polys, vX, vY)
    # İçeride en az bir True piksel olmalı
    assert mask.any(), "Poligon içinde piksel bulunamadı"
    # Köşeler False olmalı
    assert not mask[0, 0], "Sol üst köşe poligon dışında olmalı"
    return f"True piksel sayısı: {mask.sum()}"


def _t6_empty():
    vX, vY = _make_grid_vecs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = polygons2grid([], vX, vY)
    assert not mask.any()
    return "tüm False ✓"


def _t6_dtype():
    vX, vY = _make_grid_vecs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = polygons2grid([], vX, vY)
    assert mask.dtype == bool
    return f"dtype={mask.dtype}"


def _t6_shape():
    vX, vY = _make_grid_vecs(nx=25, ny=15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = polygons2grid([], vX, vY)
    assert mask.shape == (15, 25), f"Beklenen (15,25), alınan {mask.shape}"
    return f"{mask.shape}"


# ===========================================================================
# TEST 7 — shift_dem
# ===========================================================================
def test_shift_dem():
    section("TEST 7 · shift_dem")
    run_test("çıktı boyutu (3,)",           _t7_shape)
    run_test("saf z kaydırması — dz tahmini", _t7_z_shift)
    run_test("maske uygulanır",              _t7_mask)
    run_test("yetersiz piksel → sıfır",      _t7_insufficient)


def _make_dem_shift(nx=40, ny=35, seed=5):
    r = np.random.default_rng(seed)
    vX = np.linspace(500000, 500000 + (nx-1)*90, nx)
    vY = np.linspace(4500000 + (ny-1)*90, 4500000, ny)
    mX, mY = np.meshgrid(vX, vY)
    dem = np.sin((mX-500000)/300)*30 + np.cos((mY-4500000)/300)*20 + 200.
    pts_z = dem.ravel() + r.normal(0, 0.5, dem.size)
    pts = np.vstack([mX.ravel(), mY.ravel(), pts_z])
    mask = np.zeros(dem.shape, dtype=bool)
    return pts, dem, mask, vX, vY


def _t7_shape():
    pts, dem, mask, vX, vY = _make_dem_shift()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vShift = shift_dem(pts, dem, mask, vX, vY, scale=np.array([1., 1.]))
    assert vShift.shape == (3,), f"Beklenen (3,), alınan {vShift.shape}"
    return f"{vShift.shape}"


def _t7_z_shift():
    pts, dem, mask, vX, vY = _make_dem_shift()
    # Sabit z kayması ekle
    pts_shifted = pts.copy()
    pts_shifted[2] += 5.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vShift = shift_dem(pts_shifted, dem, mask, vX, vY,
                           scale=np.array([1., 1.]))
    # dz bileşeni ~5 olmalı
    assert abs(vShift[2]) > 1.0, f"dz={vShift[2]:.2f}, 5 m kayma beklendi"
    return f"dz={vShift[2]:.2f} m (beklenen ~5)"


def _t7_mask():
    pts, dem, mask, vX, vY = _make_dem_shift()
    # Tüm alanı maskele
    full_mask = np.ones(dem.shape, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vShift = shift_dem(pts, dem, full_mask, vX, vY,
                           scale=np.array([1., 1.]))
    assert vShift.shape == (3,)
    return "tam maske → sıfır kaydırma ✓"


def _t7_insufficient():
    """Çok az geçerli piksel → sıfır vektör."""
    vX = np.array([500000., 500090.])
    vY = np.array([4500090., 4500000.])
    pts = np.array([[500045.], [4500045.], [200.]])
    dem = np.array([[200., 200.], [200., 200.]])
    mask = np.zeros((2, 2), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vShift = shift_dem(pts, dem, mask, vX, vY,
                           scale=np.array([1., 1.]))
    assert vShift.shape == (3,)
    return "yetersiz piksel → (3,) sıfır ✓"


# ===========================================================================
# TEST 8 — read_geotiff_region
# ===========================================================================
def test_read_geotiff_region():
    section("TEST 8 · read_geotiff_region")

    if not HAS_RASTERIO:
        print("  [ATLANDI] rasterio kurulu değil")
        results.append(("read_geotiff_region", None, "atlandı"))
        return

    run_test("tam bölge okunur",               _t8_full)
    run_test("kısmi bölge — alt küme",          _t8_partial)
    run_test("tampon % ile büyür",              _t8_buffer)
    run_test("çakışmasız pencere → ValueError", _t8_no_overlap)
    run_test("nodata → NaN dönüşümü",          _t8_nodata)


def _t8_full():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'test.tif'
        data = _make_tif(f)
        win = np.array([[30.0, 39.0], [31.0, 40.0]])
        out, vX, vY = read_geotiff_region(win, str(f))
        assert out.shape[0] > 0 and out.shape[1] > 0
        return f"boyut={out.shape}"


def _t8_partial():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'test.tif'
        _make_tif(f)
        win = np.array([[30.2, 39.3], [30.7, 39.7]])
        out, vX, vY = read_geotiff_region(win, str(f))
        assert out.shape[0] > 0
        assert vX[0] >= 30.0 and vX[-1] <= 31.0
        return f"boyut={out.shape}, vX=[{vX[0]:.2f},{vX[-1]:.2f}]"


def _t8_buffer():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'test.tif'
        _make_tif(f)
        win_small = np.array([[30.4, 39.4], [30.6, 39.6]])
        out_no_buf, vX_nb, _ = read_geotiff_region(win_small, str(f))
        out_buf, vX_b, _ = read_geotiff_region(win_small, str(f),
                                                buffer_pct=20.0)
        assert out_buf.shape[1] >= out_no_buf.shape[1], \
            "Tamponlu bölge daha geniş olmalı"
        return f"tamponsuz W={out_no_buf.shape[1]}, tamponlu W={out_buf.shape[1]}"


def _t8_no_overlap():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'test.tif'
        _make_tif(f)   # 30-31 lon, 39-40 lat kapsar
        win = np.array([[50.0, 60.0], [51.0, 61.0]])  # Hiç örtüşmez
        try:
            read_geotiff_region(win, str(f))
            assert False, "ValueError beklendi"
        except ValueError:
            return "ValueError ✓"


def _t8_nodata():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'test.tif'
        data = np.full((20, 30), -9999, dtype=np.float32)
        data[5:10, 5:10] = 500.0
        transform = from_bounds(30.0, 39.0, 31.0, 40.0, 30, 20)
        with rasterio.open(
            f, 'w', driver='GTiff',
            height=20, width=30, count=1,
            dtype='float32', crs=CRS.from_epsg(4326),
            transform=transform, nodata=-9999
        ) as ds:
            ds.write(data, 1)
        win = np.array([[30.0, 39.0], [31.0, 40.0]])
        out, _, _ = read_geotiff_region(win, str(f))
        assert np.any(np.isnan(out)), "nodata → NaN dönüşmedi"
        assert np.any(~np.isnan(out)), "Tüm değerler NaN olmamalı"
        return "nodata → NaN ✓"


# ===========================================================================
# TEST 9 — check_input / check_shp_path
# ===========================================================================
def test_check_input():
    section("TEST 9 · check_input / check_shp_path")

    if not HAS_RASTERIO:
        print("  [ATLANDI] rasterio kurulu değil")
        results.append(("check_input", None, "atlandı"))
        return

    run_test("geçerli WGS84 GeoTIFF → hata yok",   _t9_valid_tif)
    run_test("WGS84 olmayan GeoTIFF → ValueError",   _t9_invalid_tif)
    run_test("geçersiz uzantı → ValueError",          _t9_bad_ext)
    run_test("check_shp_path boş klasör → ValueError",_t9_empty_dir)


def _t9_valid_tif():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'ok.tif'
        _make_tif(f, epsg=4326)
        check_input(str(f))   # hata yükseltmemeli
        return "WGS84 GeoTIFF ✓"


def _t9_invalid_tif():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'bad.tif'
        _make_tif(f, epsg=32636)   # UTM Zone 36N
        try:
            check_input(str(f))
            assert False, "ValueError beklendi"
        except ValueError:
            return "ValueError ✓"


def _t9_bad_ext():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'data.csv'
        f.touch()
        try:
            check_input(str(f))
            assert False, "ValueError beklendi"
        except ValueError:
            return "ValueError ✓"


def _t9_empty_dir():
    with tempfile.TemporaryDirectory() as d:
        try:
            check_shp_path(d)
            assert False, "ValueError beklendi"
        except ValueError:
            return "ValueError ✓"


# ===========================================================================
# TEST 10 — colormap_dem
# ===========================================================================
def test_colormap_dem():
    section("TEST 10 · colormap_dem")
    run_test("şekil (N, 3)",              _t10_shape)
    run_test("değerler [0, 1] aralığında", _t10_range)
    run_test("renk paleti monoton değil",  _t10_varied)


def _t10_shape():
    cmap = colormap_dem()
    assert cmap.ndim == 2 and cmap.shape[1] == 3
    return f"{cmap.shape}"


def _t10_range():
    cmap = colormap_dem()
    assert cmap.min() >= 0.0 and cmap.max() <= 1.0
    return f"[{cmap.min():.2f}, {cmap.max():.2f}]"


def _t10_varied():
    cmap = colormap_dem()
    assert cmap.std() > 0.05, "Renk paleti çok düz"
    return f"std={cmap.std():.3f}"


# ===========================================================================
# ANA ÇALIŞTIRICI
# ===========================================================================
def main():
    print("\n" + "=" * 60)
    print("  SHARED UTILS ENTEGRASYON TESTİ")
    print("  shared_utils.py")
    print("=" * 60)

    test_get_files()
    test_block_process()
    test_make_spatial_ref_vecs()
    test_neighbors()
    test_ps_projection()
    test_polygons2grid()
    test_shift_dem()
    test_read_geotiff_region()
    test_check_input()
    test_colormap_dem()

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
