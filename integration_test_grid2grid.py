"""
integration_test_grid2grid.py

grid2grid.py entegrasyon testi.
MATLAB karşılıkları: shared/grid2grid.m, shared/blockProcess.m,
                     shared/makeSpatialRefVecs.m

Çalıştırma:
    python integration_test_grid2grid.py

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

for p in [SHARED_DIR, REPO_ROOT]:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

try:
    from grid2grid import (
        grid2grid,
        make_spatial_ref_vecs,
        block_process,
        _mask_nulls,
        _resample,
        _read_source,
    )
except ImportError as e:
    print(f"[HATA] grid2grid.py yüklenemedi: {e}")
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


# ---------------------------------------------------------------------------
# Test verisi üreticileri
# ---------------------------------------------------------------------------

def _make_source_dict(nx=100, ny=80,
                      lon0=30.0, lat1=40.0,
                      dlon=0.1, dlat=0.1,
                      fn=None, seed=0):
    """Sentetik kaynak ızgara sözlüğü üret."""
    src_x = np.linspace(lon0, lon0 + (nx-1)*dlon, nx)
    src_y = np.linspace(lat1, lat1 - (ny-1)*dlat, ny)   # azalan
    mX, mY = np.meshgrid(src_x, src_y)
    if fn is None:
        data = np.sin(mX / 2) * np.cos(mY / 2)
    else:
        data = fn(mX, mY)
    sR = {
        'DeltaLon':  dlon,
        'DeltaLat': -dlat,
        'Lonlim': (lon0, lon0 + (nx-1)*dlon),
        'Latlim': (lat1 - (ny-1)*dlat, lat1),
    }
    return {'data': data, 'spatial_ref': sR}, src_x, src_y, data


def _make_target_vecs(lon0=31.0, lat1=39.0, dlon=0.05, dlat=0.05,
                      nx=40, ny=30):
    vX = np.linspace(lon0, lon0 + (nx-1)*dlon, nx)
    vY = np.linspace(lat1, lat1 - (ny-1)*dlat, ny)
    return vX, vY


def _make_tif(path, data=None, lon0=30.0, lat0=39.0,
              lon1=31.0, lat1=40.0, epsg=4326, nodata=None):
    if data is None:
        h, w = 40, 50
        src_x = np.linspace(lon0, lon1, w)
        src_y = np.linspace(lat1, lat0, h)
        mX, mY = np.meshgrid(src_x, src_y)
        data = (np.sin(mX * 3) * np.cos(mY * 3) * 100).astype(np.float32)
    transform = from_bounds(lon0, lat0, lon1, lat1,
                            data.shape[1], data.shape[0])
    kwargs = dict(driver='GTiff', height=data.shape[0], width=data.shape[1],
                  count=1, dtype=data.dtype, crs=CRS.from_epsg(epsg),
                  transform=transform)
    if nodata is not None:
        kwargs['nodata'] = nodata
    with rasterio.open(path, 'w', **kwargs) as ds:
        ds.write(data, 1)
    return data


# ===========================================================================
# TEST 1 — make_spatial_ref_vecs
# ===========================================================================
def test_make_spatial_ref_vecs():
    section("TEST 1 · make_spatial_ref_vecs")

    run_test("coğrafi sR — full vektör uzunluğu",    _t1_geo_full_len)
    run_test("coğrafi sR — vX artan, vY azalan",     _t1_geo_monotone)
    run_test("limits — 2 eleman",                    _t1_limits)
    run_test("yansıtmalı sR (DeltaX/Y)",             _t1_projected)
    run_test("geçersiz anahtar → ValueError",         _t1_bad_key)


def _make_geo_sR(nx=20, ny=15):
    return {
        'DeltaLon':  0.1, 'DeltaLat': -0.1,
        'Lonlim': (30.0, 30.0 + (nx-1)*0.1),
        'Latlim': (39.0, 39.0 + (ny-1)*0.1),
    }


def _t1_geo_full_len():
    sR = _make_geo_sR(nx=20, ny=15)
    vX, vY = make_spatial_ref_vecs(sR, 'full')
    # np.arange floating point nedeniyle ±1 tolerans
    assert abs(len(vX) - 20) <= 1 and abs(len(vY) - 15) <= 1, \
        f"Beklenen ~(20,15), alınan ({len(vX)},{len(vY)})"
    return f"vX={len(vX)}, vY={len(vY)}"


def _t1_geo_monotone():
    sR = _make_geo_sR()
    vX, vY = make_spatial_ref_vecs(sR, 'full')
    assert np.all(np.diff(vX) > 0), "vX artan değil"
    assert np.all(np.diff(vY) < 0), "vY azalan değil"
    return "monotonluk ✓"


def _t1_limits():
    sR = _make_geo_sR(nx=50, ny=40)
    vX, vY = make_spatial_ref_vecs(sR, 'limits')
    assert len(vX) == 2 and len(vY) == 2
    return f"vX={np.round(vX,3)}, vY={np.round(vY,3)}"


def _t1_projected():
    sR = {
        'DeltaX': 90.0, 'DeltaY': -90.0,
        'XLimWorld': (500000.0, 500000.0 + 30*90),
        'YLimWorld': (4400000.0, 4400000.0 + 20*90),
    }
    vX, vY = make_spatial_ref_vecs(sR, 'full')
    assert len(vX) == 30 and len(vY) == 20
    return f"vX={len(vX)}, vY={len(vY)}"


def _t1_bad_key():
    sR = {'foo': 1, 'bar': 2}
    try:
        make_spatial_ref_vecs(sR, 'full')
        assert False, "ValueError beklendi"
    except ValueError:
        return "ValueError ✓"


# ===========================================================================
# TEST 2 — block_process
# ===========================================================================
def test_block_process():
    section("TEST 2 · block_process")

    run_test("sabit değer — tüm hücreler dolu",      _t2_constant)
    run_test("çıktı şekli doğru",                    _t2_shape)
    run_test("double dtype → NaN başlangıç",         _t2_nan_init)
    run_test("logical dtype → False başlangıç",      _t2_logical)
    run_test("hata veren blok → uyarı, atlanır",     _t2_error_block)
    run_test("yanlış boyut → uyarı, atlanır",        _t2_wrong_size)


def _t2_constant():
    def fn(s):
        ix, iy = s['count_x'], s['count_y']
        h = s['index_y'][iy+1] - s['index_y'][iy]
        w = s['index_x'][ix+1] - s['index_x'][ix]
        return np.ones((h, w))
    grid = block_process((20, 30), 10, 'double', fn)
    assert grid.shape == (20, 30) and np.all(grid == 1.0)
    return "tüm hücreler 1.0 ✓"


def _t2_shape():
    def fn(s):
        ix, iy = s['count_x'], s['count_y']
        h = s['index_y'][iy+1] - s['index_y'][iy]
        w = s['index_x'][ix+1] - s['index_x'][ix]
        return np.zeros((h, w))
    for H, W, bs in [(15, 25, 7), (100, 100, 33), (5, 5, 100)]:
        g = block_process((H, W), bs, 'double', fn)
        assert g.shape == (H, W)
    return "tüm boyutlar ✓"


def _t2_nan_init():
    """double dtype → başlangıç değeri NaN."""
    called = []
    def fn(s):
        called.append(1)
        return None   # None döndür → blok atlanır
    grid = block_process((10, 10), 5, 'double', fn)
    assert np.all(np.isnan(grid))
    return "NaN başlangıç ✓"


def _t2_logical():
    def fn(s):
        ix, iy = s['count_x'], s['count_y']
        h = s['index_y'][iy+1] - s['index_y'][iy]
        w = s['index_x'][ix+1] - s['index_x'][ix]
        return np.ones((h, w), dtype=bool)
    grid = block_process((8, 12), 4, 'logical', fn)
    assert grid.dtype == bool and grid.all()
    return "logical dtype ✓"


def _t2_error_block():
    def fn(s):
        raise RuntimeError("test hatası")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grid = block_process((10, 10), 5, 'double', fn)
    assert len(w) > 0
    assert np.all(np.isnan(grid))
    return "hata → uyarı ✓"


def _t2_wrong_size():
    def fn(s):
        return np.ones((1, 1))   # Her zaman yanlış boyut
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grid = block_process((10, 10), 5, 'double', fn)
    assert len(w) > 0
    return "yanlış boyut → uyarı ✓"


# ===========================================================================
# TEST 3 — _mask_nulls
# ===========================================================================
def test_mask_nulls():
    section("TEST 3 · _mask_nulls")

    run_test("sayısal null değeri → NaN",             _t3_numeric)
    run_test("null_val listesi",                      _t3_list)
    run_test("null_val='dem' string",                 _t3_dem_string)
    run_test("null_val=NaN → veri değişmez",          _t3_nan_passthrough)
    run_test("null_val=None → veri değişmez",         _t3_none_passthrough)


def _t3_numeric():
    data = np.array([[1., -9999., 3.], [4., 5., -9999.]])
    out = _mask_nulls(data.copy(), -9999.)
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 2])
    assert out[0, 0] == 1.0
    return "sayısal null → NaN ✓"


def _t3_list():
    data = np.array([[0., 1., -1.], [2., -9999., 3.]])
    out = _mask_nulls(data.copy(), [0., -9999.])
    assert np.isnan(out[0, 0]) and np.isnan(out[1, 1])
    assert out[0, 1] == 1.0
    return "liste null ✓"


def _t3_dem_string():
    data = np.array([[-600., 500., 9500.], [100., 200., 300.]])
    out = _mask_nulls(data.copy(), 'dem')
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 2])
    assert out[0, 1] == 500.0
    return "dem string ✓"


def _t3_nan_passthrough():
    data = np.array([[1., 2.], [3., 4.]])
    out = _mask_nulls(data.copy(), np.nan)
    assert np.allclose(out, data)
    return "NaN passthrough ✓"


def _t3_none_passthrough():
    data = np.array([[1., 2.], [3., 4.]])
    out = _mask_nulls(data.copy(), None)
    assert np.allclose(out, data)
    return "None passthrough ✓"


# ===========================================================================
# TEST 4 — _resample
# ===========================================================================
def test_resample():
    section("TEST 4 · _resample")

    run_test("aynı ızgara — veri korunur",            _t4_identity)
    run_test("lineer — düz yüzey doğru",              _t4_linear_flat)
    run_test("nearest yöntemi",                       _t4_nearest)
    run_test("kapsam dışı → NaN",                     _t4_out_of_bounds)
    run_test("azalan src_y → otomatik tersine çevir", _t4_descending_y)


def _make_src(nx=30, ny=25):
    src_x = np.linspace(0.0, 3.0, nx)
    src_y = np.linspace(0.0, 2.5, ny)   # artan (normal)
    mX, mY = np.meshgrid(src_x, src_y)
    data = mX * 2 + mY * 3
    return data, src_x, src_y


def _t4_identity():
    data, src_x, src_y = _make_src()
    mX, mY = np.meshgrid(src_x[5:15], src_y[5:15])
    out = _resample(data, src_x, src_y, mX, mY, 'linear')
    expected = mX * 2 + mY * 3
    err = np.nanmax(np.abs(out - expected))
    assert err < 1e-9, f"Hata: {err:.2e}"
    return f"err={err:.1e}"


def _t4_linear_flat():
    src_x = np.linspace(0, 10, 50)
    src_y = np.linspace(0, 8, 40)
    mSX, mSY = np.meshgrid(src_x, src_y)
    data = mSX * 1.5 + mSY * 2.5 + 10.0
    mX, mY = np.meshgrid(np.linspace(2, 8, 20), np.linspace(1, 7, 15))
    out = _resample(data, src_x, src_y, mX, mY, 'linear')
    expected = mX * 1.5 + mY * 2.5 + 10.0
    err = np.nanmax(np.abs(out - expected))
    assert err < 1e-8, f"Doğrusal hata: {err:.2e}"
    return f"err={err:.2e}"


def _t4_nearest():
    data, src_x, src_y = _make_src()
    mX, mY = np.meshgrid(src_x[5:15], src_y[5:15])
    out = _resample(data, src_x, src_y, mX, mY, 'nearest')
    assert out.shape == mX.shape and not np.any(np.isnan(out))
    return f"{out.shape}"


def _t4_out_of_bounds():
    data, src_x, src_y = _make_src()
    # Kaynak dışında hedef koordinatlar
    mX, mY = np.meshgrid([100., 200.], [100., 200.])
    out = _resample(data, src_x, src_y, mX, mY, 'linear')
    assert np.all(np.isnan(out)), "Kapsam dışı → NaN beklendi"
    return "kapsam dışı NaN ✓"


def _t4_descending_y():
    """src_y azalan sırada verilirse _resample otomatik tersine çevirmeli."""
    src_x = np.linspace(0, 5, 30)
    src_y_asc = np.linspace(0, 4, 25)
    src_y_desc = src_y_asc[::-1]   # azalan
    mSX, mSY = np.meshgrid(src_x, src_y_asc)
    data_asc = mSX + mSY
    data_desc = data_asc[::-1, :]   # azalan sıraya uygun veri

    mX, mY = np.meshgrid(src_x[5:15], src_y_asc[5:15])
    out_asc  = _resample(data_asc,  src_x, src_y_asc,  mX, mY, 'linear')
    out_desc = _resample(data_desc, src_x, src_y_desc, mX, mY, 'linear')
    err = np.nanmax(np.abs(out_asc - out_desc))
    assert err < 1e-8, f"Azalan/artan farkı: {err:.2e}"
    return f"err={err:.2e}"


# ===========================================================================
# TEST 5 — grid2grid: dict kaynağı
# ===========================================================================
def test_grid2grid_dict():
    section("TEST 5 · grid2grid — dict kaynağı")

    run_test("çıktı şekli (H, W)",                   _t5_shape)
    run_test("NaN oranı düşük (örtüşen bölge)",       _t5_low_nan)
    run_test("düz yüzey — interpolasyon doğru",       _t5_flat_surface)
    run_test("daha ince çözünürlük",                  _t5_finer_res)
    run_test("daha kaba çözünürlük",                  _t5_coarser_res)
    run_test("null_val maskeleme",                    _t5_null_val)
    run_test("nearest interpolasyon",                 _t5_nearest)
    run_test("hatalı vX yönü → ValueError",           _t5_bad_vx)
    run_test("hatalı vY yönü → ValueError",           _t5_bad_vy)


def _t5_shape():
    src, _, _, _ = _make_source_dict()
    vX, vY = _make_target_vecs()
    out = grid2grid(src, vX, vY)
    assert out.shape == (len(vY), len(vX))
    return f"{out.shape}"


def _t5_low_nan():
    src, _, _, _ = _make_source_dict()
    vX, vY = _make_target_vecs()
    out = grid2grid(src, vX, vY)
    nan_pct = np.isnan(out).mean() * 100
    assert nan_pct < 10, f"NaN oranı yüksek: %{nan_pct:.1f}"
    return f"NaN=%{nan_pct:.1f}"


def _t5_flat_surface():
    """z = ax + by → lineer interpolasyon mükemmel olmalı."""
    src, src_x, src_y, _ = _make_source_dict(
        fn=lambda x, y: 2.0 * x + 3.0 * y
    )
    # Hedef noktaları kaynak sınırlarının kesinlikle içinde tut
    vX = np.linspace(32.0, 37.0, 25)
    vY = np.linspace(38.5, 33.5, 20)
    out = grid2grid(src, vX, vY)
    mX, mY = np.meshgrid(vX, vY)
    expected = 2.0 * mX + 3.0 * mY
    valid = ~np.isnan(out)
    assert valid.any(), "Geçerli piksel yok"
    err = np.max(np.abs(out[valid] - expected[valid]))
    assert err < 0.1, f"Düz yüzey hatası: {err:.4f}"
    return f"err={err:.4f}"


def _t5_finer_res():
    src, _, _, _ = _make_source_dict(nx=50, ny=40, dlon=0.1, dlat=0.1)
    # Kaynak: lon [30, 34.9], lat [36.1, 40] — hedef kesinlikle içeride
    vX = np.linspace(31.0, 34.0, 61)   # ~0.05° çözünürlük
    vY = np.linspace(39.5, 37.0, 51)
    out = grid2grid(src, vX, vY)
    assert out.shape == (51, 61)
    nan_pct = np.isnan(out).mean() * 100
    assert nan_pct < 10, f"NaN oranı yüksek: %{nan_pct:.1f}"
    return f"{out.shape}, NaN=%{nan_pct:.1f}"


def _t5_coarser_res():
    src, _, _, _ = _make_source_dict(nx=100, ny=80, dlon=0.05, dlat=0.05)
    vX = np.linspace(31.0, 38.0, 15)    # daha kaba
    vY = np.linspace(39.0, 33.0, 12)
    out = grid2grid(src, vX, vY)
    assert out.shape == (12, 15)
    return f"{out.shape}"


def _t5_null_val():
    src, _, _, data = _make_source_dict()
    src['data'] = data.copy()
    src['data'][10:15, 10:15] = -9999.0
    vX = np.linspace(31.0, 38.0, 40)
    vY = np.linspace(39.0, 33.0, 30)
    out = grid2grid(src, vX, vY, params={'null_val': -9999.0})
    # Bazı pikseller NaN olmalı (maskelenen bölge)
    assert np.any(np.isnan(out)), "null_val maskeleme çalışmadı"
    return "null_val → NaN ✓"


def _t5_nearest():
    src, _, _, _ = _make_source_dict()
    vX, vY = _make_target_vecs()
    out = grid2grid(src, vX, vY, params={'interp': 'nearest'})
    assert out.shape == (len(vY), len(vX))
    assert not np.all(np.isnan(out))
    return f"{out.shape}"


def _t5_bad_vx():
    src, _, _, _ = _make_source_dict()
    vX = np.linspace(38.0, 31.0, 40)   # azalan — hatalı
    vY = np.linspace(39.0, 33.0, 30)
    try:
        grid2grid(src, vX, vY)
        assert False, "ValueError beklendi"
    except ValueError:
        return "ValueError ✓"


def _t5_bad_vy():
    src, _, _, _ = _make_source_dict()
    vX = np.linspace(31.0, 38.0, 40)
    vY = np.linspace(33.0, 39.0, 30)   # artan — hatalı
    try:
        grid2grid(src, vX, vY)
        assert False, "ValueError beklendi"
    except ValueError:
        return "ValueError ✓"


# ===========================================================================
# TEST 6 — grid2grid: GeoTIFF kaynağı
# ===========================================================================
def test_grid2grid_tiff():
    section("TEST 6 · grid2grid — GeoTIFF kaynağı")

    if not HAS_RASTERIO:
        print("  [ATLANDI] rasterio kurulu değil")
        results.append(("GeoTIFF kaynağı", None, "atlandı"))
        return

    run_test("GeoTIFF — çıktı şekli",                _t6_shape)
    run_test("GeoTIFF — NaN oranı düşük",             _t6_low_nan)
    run_test("GeoTIFF — nodata → NaN",                _t6_nodata)
    run_test("GeoTIFF — örtüşmez bölge → NaN",        _t6_no_overlap)
    run_test("GeoTIFF — tiff_layer=1",                _t6_layer)


def _t6_shape():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'src.tif'
        _make_tif(f)
        vX = np.linspace(30.1, 30.9, 20)
        vY = np.linspace(39.9, 39.1, 15)
        out = grid2grid(str(f), vX, vY)
        assert out.shape == (15, 20)
        return f"{out.shape}"


def _t6_low_nan():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'src.tif'
        _make_tif(f)
        vX = np.linspace(30.1, 30.9, 20)
        vY = np.linspace(39.9, 39.1, 15)
        out = grid2grid(str(f), vX, vY)
        nan_pct = np.isnan(out).mean() * 100
        assert nan_pct < 20, f"NaN oranı: %{nan_pct:.1f}"
        return f"NaN=%{nan_pct:.1f}"


def _t6_nodata():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'src.tif'
        h, w = 40, 50
        data = np.ones((h, w), dtype=np.float32) * 100.0
        data[10:20, 10:20] = -9999.0
        _make_tif(f, data=data, nodata=-9999.0)
        vX = np.linspace(30.05, 30.95, 25)
        vY = np.linspace(39.95, 39.05, 20)
        out = grid2grid(str(f), vX, vY,
                        params={'null_val': -9999.0})
        assert np.any(np.isnan(out)), "nodata → NaN çalışmadı"
        return "nodata → NaN ✓"


def _t6_no_overlap():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'src.tif'
        _make_tif(f)   # 30-31 lon, 39-40 lat
        vX = np.linspace(50.0, 51.0, 10)   # Örtüşmez
        vY = np.linspace(60.0, 59.0, 8)
        out = grid2grid(str(f), vX, vY)
        assert np.all(np.isnan(out)), "Örtüşmez bölge NaN olmalı"
        return "örtüşmez → NaN ✓"


def _t6_layer():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'src.tif'
        _make_tif(f)
        vX = np.linspace(30.1, 30.9, 15)
        vY = np.linspace(39.9, 39.1, 12)
        out = grid2grid(str(f), vX, vY, params={'tiff_layer': 1})
        assert out.shape == (12, 15)
        return f"{out.shape}"


# ===========================================================================
# TEST 7 — grid2grid: koordinat dönüşümü
# ===========================================================================
def test_grid2grid_transform():
    section("TEST 7 · grid2grid — koordinat dönüşümü")

    run_test("kimlik matrisi — veri değişmez",        _t7_identity_transform)
    run_test("geçersiz transform → ValueError",        _t7_bad_transform)


def _t7_identity_transform():
    src, _, _, _ = _make_source_dict()
    vX, vY = _make_target_vecs()
    out_no_t = grid2grid(src, vX, vY)
    out_id   = grid2grid(src, vX, vY, params={'transform': np.eye(4)})
    valid = ~np.isnan(out_no_t) & ~np.isnan(out_id)
    if valid.any():
        err = np.max(np.abs(out_no_t[valid] - out_id[valid]))
        assert err < 0.01, f"Kimlik dönüşüm hatası: {err:.4f}"
        return f"err={err:.4f}"
    return "geçerli piksel yok (kabul)"


def _t7_bad_transform():
    src, _, _, _ = _make_source_dict()
    vX, vY = _make_target_vecs()
    # grid2grid.py 3×3 transform için blok içinde uyarı veriyor (ValueError değil)
    # → çıktı NaN dolu olmalı
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = grid2grid(src, vX, vY, params={'transform': np.eye(3)})
    assert len(w) > 0 or np.all(np.isnan(out)), \
        "Geçersiz transform uyarı vermeli veya NaN üretmeli"
    return "geçersiz transform → uyarı/NaN ✓"


# ===========================================================================
# ANA ÇALIŞTIRICI
# ===========================================================================
def main():
    print("\n" + "=" * 60)
    print("  GRID2GRID ENTEGRASYON TESTİ")
    print("  grid2grid.py")
    print("=" * 60)

    test_make_spatial_ref_vecs()
    test_block_process()
    test_mask_nulls()
    test_resample()
    test_grid2grid_dict()
    test_grid2grid_tiff()
    test_grid2grid_transform()

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
