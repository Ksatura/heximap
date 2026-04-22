"""
shared_utils.py

MATLAB kaynak dosyaları:
  - main/shared/blockProcess.m
  - main/shared/polygons2grid.m
  - main/shared/makeSpatialRefVecs.m
  - main/shared/neighbors.m
  - main/shared/shiftDem.m
  - main/shared/readGeotiffRegion.m
  - main/shared/ll2ps.m
  - main/shared/ps2ll.m
  - main/shared/checkInput.m
  - main/shared/checkShpPath.m
  - main/shared/colormapDEM.m
  - main/shared/getFiles.m

Açıklama:
  Paylaşılan yardımcı fonksiyonlar: blok işleme, koordinat dönüşümleri,
  DEM yardımcıları, dosya sistemi ve doğrulama araçları.

Bağımlılıklar:
  - numpy   (pip install numpy)
  - scipy   (pip install scipy)
  - rasterio(pip install rasterio)

Dahili bağımlılıklar:
  - geo_optimize.py (points2grid, ll2utm)
"""

import numpy as np
import warnings
from pathlib import Path
from scipy.optimize import least_squares

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '3_georef'))

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    warnings.warn("rasterio kurulu değil. readGeotiffRegion çalışmayacak.")

try:
    from shapely.geometry import shape, Point
    from shapely.vectorized import contains
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    warnings.warn("shapely kurulu değil. polygons2grid sınırlı çalışacak.")


# ===========================================================================
# GET FILES — Özyinelemeli Dosya Listesi
# MATLAB karşılığı: shared/getFiles.m
# ===========================================================================

def get_files(directory, extension):
    """
    Bir klasör ve alt klasörlerindeki belirli uzantılı dosyaları listeler.

    MATLAB imzası:
        cFiles = getFiles(strDir, strExt)

    Parametreler
    ----------
    directory : str veya Path
    extension : str  — örn. '.shp', '.tif'

    Döndürür
    -------
    list[Path]
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    ext = extension.lower()
    return sorted(
        p for p in directory.rglob(f"*{ext}")
        if p.suffix.lower() == ext
    )


# ===========================================================================
# BLOCK PROCESS — Blok Tabanlı Işleme Motoru
# MATLAB karşılığı: shared/blockProcess.m
# ===========================================================================

def block_process(shape, block_size, dtype, fn, **kwargs):
    """
    Büyük ızgaraları bloklara bölerek işler.

    MATLAB imzası:
        mGrid = blockProcess(vSz, dBlkSz, strClass, hFun, varargin)

    Parametreler
    ----------
    shape      : tuple (H, W)
    block_size : int   — blok boyutu (piksel)
    dtype      : str   — 'bool', 'uint8', 'float64'
    fn         : callable(sInput) → np.ndarray
        sInput içeriği:
            'index_x'  : np.ndarray — X blok sınır indeksleri
            'index_y'  : np.ndarray — Y blok sınır indeksleri
            'count_x'  : int
            'count_y'  : int
    **kwargs   : fn'e iletilir

    Döndürür
    -------
    np.ndarray, shape (H, W)
    """
    H, W = shape

    # Blok sınır indekslerini hesapla
    n_blk_x = max(1, int(np.ceil(W / block_size)))
    n_blk_y = max(1, int(np.ceil(H / block_size)))
    idx_x = np.round(np.linspace(0, W, n_blk_x + 1)).astype(int)
    idx_y = np.round(np.linspace(0, H, n_blk_y + 1)).astype(int)

    # Tekli blok için düzelt
    if len(idx_x) == 1:
        idx_x = np.array([0, idx_x[0]])
    if len(idx_y) == 1:
        idx_y = np.array([0, idx_y[0]])

    # Çıktı ızgarasını başlat
    np_dtype = {'bool': bool, 'logical': bool,
                'uint8': np.uint8, 'double': np.float64,
                'float64': np.float64}.get(dtype, np.float64)
    if np_dtype == bool:
        grid = np.zeros(shape, dtype=bool)
    elif np_dtype == np.uint8:
        grid = np.zeros(shape, dtype=np.uint8)
    else:
        grid = np.full(shape, np.nan)

    for iy in range(len(idx_y) - 1):
        for ix in range(len(idx_x) - 1):
            r0, r1 = idx_y[iy], idx_y[iy + 1]
            c0, c1 = idx_x[ix], idx_x[ix + 1]

            s_input = {
                'index_x': idx_x,
                'index_y': idx_y,
                'count_x': ix,
                'count_y': iy,
            }

            try:
                block = fn(s_input, **kwargs)
            except Exception as e:
                warnings.warn(f"Blok ({iy},{ix}) işleme hatası: {e}")
                continue

            exp_shape = (r1 - r0, c1 - c0)
            if block.shape != exp_shape:
                warnings.warn(
                    f"Blok ({iy},{ix}) yanlış boyut: "
                    f"beklenen {exp_shape}, alınan {block.shape}"
                )
                continue

            grid[r0:r1, c0:c1] = block.astype(np_dtype)

    return grid


# ===========================================================================
# MAKE SPATIAL REF VECS — Koordinat Vektörleri
# MATLAB karşılığı: shared/makeSpatialRefVecs.m
# ===========================================================================

def make_spatial_ref_vecs(sR, vec_type='full'):
    """
    Spatial referans sözlüğünden koordinat vektörleri üretir.

    MATLAB imzası:
        [vX, vY] = makeSpatialRefVecs(sR, strType)

    Parametreler
    ----------
    sR       : dict
        'Lonlim'    : [lon_min, lon_max]
        'Latlim'    : [lat_min, lat_max]
        'RasterSize': (H, W)
        veya
        'vX', 'vY' doğrudan vektörler
    vec_type : str  — 'full' veya 'limits'

    Döndürür
    -------
    vX : np.ndarray — boylam vektörü (artan)
    vY : np.ndarray — enlem vektörü (azalan)
    """
    # Doğrudan vektörler varsa kullan
    if 'vX' in sR and 'vY' in sR:
        vX = np.array(sR['vX'])
        vY = np.array(sR['vY'])
        if vec_type == 'limits':
            return np.array([vX[0], vX[-1]]), np.array([vY[0], vY[-1]])
        return vX, vY

    lon_lim = sR['Lonlim']
    lat_lim = sR['Latlim']
    H, W    = sR['RasterSize']

    dX = (lon_lim[1] - lon_lim[0]) / W
    dY = (lat_lim[1] - lat_lim[0]) / H   # pozitif

    if vec_type == 'full':
        vX = np.linspace(lon_lim[0] + dX / 2, lon_lim[1] - dX / 2, W)
        vY = np.linspace(lat_lim[1] - dY / 2, lat_lim[0] + dY / 2, H)
    elif vec_type == 'limits':
        vX = np.array([lon_lim[0] + dX / 2, lon_lim[1] - dX / 2])
        vY = np.array([lat_lim[1] - dY / 2, lat_lim[0] + dY / 2])
    else:
        raise ValueError(f"Geçersiz vec_type: '{vec_type}'")

    return vX, vY


# ===========================================================================
# NEIGHBORS — 8-Komşu Piksel Değerleri
# MATLAB karşılığı: shared/neighbors.m
# ===========================================================================

def neighbors(mI, interp=True):
    """
    Her piksel için 8 komşu değerini döndürür.

    MATLAB imzası:
        mNeighbors = neighbors(mI, lInterp)

    Parametreler
    ----------
    mI     : np.ndarray, shape (H, W)
    interp : bool — True: kenar ekstrapolasyonu, False: NaN dolgu

    Döndürür
    -------
    np.ndarray, shape (H*W, 8) — saat yönünde: N, NE, E, SE, S, SW, W, NW
    """
    mI = mI.astype(float)

    if interp:
        # Kenar ekstrapolasyonu
        vL = mI[:, :1]  - (mI[:, 1:2]  - mI[:, :1])
        vR = mI[:, -1:] + (mI[:, -1:]  - mI[:, -2:-1])
        mI = np.hstack([vL, mI, vR])
        vT = mI[:1, :]  - (mI[1:2, :]  - mI[:1, :])
        vB = mI[-1:, :] + (mI[-1:, :]  - mI[-2:-1, :])
        mI = np.vstack([vT, mI, vB])
    else:
        mI = np.pad(mI, 1, constant_values=np.nan)

    H, W = mI.shape
    # 8 yön: N, NE, E, SE, S, SW, W, NW
    offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1),
               (1, 0),  (1, -1), (0, -1), (-1, -1)]

    out_h = H - 2
    out_w = W - 2
    result = np.zeros((out_h * out_w, 8))

    for i, (dr, dc) in enumerate(offsets):
        r0 = 1 + dr; r1 = H - 1 + dr
        c0 = 1 + dc; c1 = W - 1 + dc
        result[:, i] = mI[r0:r1, c0:c1].ravel()

    return result


# ===========================================================================
# POLYGONS2GRID — Shapefile Poligonlarını Izgaraya Dönüştür
# MATLAB karşılığı: shared/polygons2grid.m
# ===========================================================================

def polygons2grid(polygons, vX, vY, block_size=None):
    """
    Poligon(lar)ı boolean ızgara maskesine dönüştürür.

    MATLAB imzası:
        lGrid = polygons2grid(polygons, vX, vY, sParams)

    Parametreler
    ----------
    polygons   : list[dict] — GeoJSON benzeri geometri listesi
                 {'type': 'Polygon', 'coordinates': [...]}
                 veya shapely geometri listesi
                 veya shapefile yolu (str)
    vX         : np.ndarray — artan X (boylam)
    vY         : np.ndarray — azalan Y (enlem)
    block_size : int veya None

    Döndürür
    -------
    np.ndarray, dtype=bool, shape (len(vY), len(vX))
    """
    if vX[0] > vX[-1]:
        raise ValueError("vX artan sırada olmalıdır.")
    if len(vY) > 1 and vY[0] < vY[-1]:
        raise ValueError("vY azalan sırada olmalıdır.")

    H, W = len(vY), len(vX)

    # Shapefile yolu verilmişse yükle
    if isinstance(polygons, (str, Path)):
        polygons = _load_shapefile(polygons)

    if not polygons:
        return np.zeros((H, W), dtype=bool)

    bs = block_size or max(H, W) + 1

    def _make_block(s_input):
        ix = s_input['count_x']
        iy = s_input['count_y']
        idx_x = s_input['index_x']
        idx_y = s_input['index_y']
        blk_x = vX[idx_x[ix]:idx_x[ix + 1]]
        blk_y = vY[idx_y[iy]:idx_y[iy + 1]]
        return _rasterize_polygons(polygons, blk_x, blk_y)

    return block_process((H, W), bs, 'bool', _make_block)


def _load_shapefile(path):
    """Shapefile'dan poligon listesi yükler."""
    try:
        import fiona
        with fiona.open(path) as src:
            return [feature['geometry'] for feature in src]
    except ImportError:
        warnings.warn("fiona kurulu değil. Shapefile okunamıyor.")
        return []


def _rasterize_polygons(polygons, vX, vY):
    """Bir blok üzerinde poligonları rasterleştirir."""
    mX, mY = np.meshgrid(vX, vY)
    pts_flat = np.column_stack([mX.ravel(), mY.ravel()])
    mask = np.zeros(len(pts_flat), dtype=bool)

    for geom in polygons:
        try:
            if HAS_SHAPELY:
                from shapely.geometry import shape as shp_shape
                poly = shp_shape(geom) if isinstance(geom, dict) else geom
                from shapely.vectorized import contains as shp_contains
                mask |= shp_contains(poly, pts_flat[:, 0], pts_flat[:, 1])
            else:
                # Basit ray casting (shapely yoksa)
                coords = (geom.get('coordinates', [[]])[0]
                          if isinstance(geom, dict) else [])
                if coords:
                    mask |= _ray_cast(np.array(coords), pts_flat)
        except Exception as e:
            warnings.warn(f"Poligon rasterleştirme hatası: {e}")

    return mask.reshape(len(vY), len(vX))


def _ray_cast(poly_xy, pts):
    """Basit nokta-içi-poligon testi (ray casting)."""
    n = len(poly_xy)
    inside = np.zeros(len(pts), dtype=bool)
    x, y = pts[:, 0], pts[:, 1]
    j = n - 1
    for i in range(n):
        xi, yi = poly_xy[i]
        xj, yj = poly_xy[j]
        cond = ((yi > y) != (yj > y)) & \
               (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


# ===========================================================================
# SHIFT DEM — Nuth & Kääb DEM Kayma Düzeltmesi
# MATLAB karşılığı: shared/shiftDem.m
# ===========================================================================

def shift_dem(pts, dem_ref, mask, vX, vY, scale):
    """
    Nuth & Kääb (2011) yöntemiyle DEM kayma vektörünü hesaplar.

    MATLAB imzası:
        vShift = shiftDem(mPts, mDem, lM, vX, vY, vScale)

    Parametreler
    ----------
    pts    : np.ndarray, shape (3+, N) — [x, y, z, ...]
    dem_ref: np.ndarray, shape (H, W) — referans DEM
    mask   : np.ndarray, bool (H, W) — kararsız arazi maskesi
    vX     : np.ndarray — artan X koordinatları
    vY     : np.ndarray — azalan Y koordinatları
    scale  : np.ndarray, shape (2,) — [sx, sy] ölçek faktörleri

    Döndürür
    -------
    vShift : np.ndarray, shape (3,) — [dx, dy, dz]
    """
    from geo_optimize import points2grid

    # Hareketli DEM'i ızgaraya dönüştür
    dem_moving = points2grid(pts[:3], vX, vY, 'sparse')

    # Referans DEM gradyanları
    dx = np.mean(np.abs(np.diff(vX))) * scale[0]
    dy = np.mean(np.abs(np.diff(vY))) * scale[1]
    mP = np.gradient(dem_ref, dx, axis=1)   # ∂Z/∂X
    mQ = np.gradient(dem_ref, dy, axis=0)   # ∂Z/∂Y

    slope  = np.degrees(np.arctan(np.sqrt(mP**2 + mQ**2)))
    aspect = np.degrees(np.arctan2(mP, mQ)) + 180

    # Yükseklik farkı
    diff = dem_ref - dem_moving
    diff[mask] = np.nan

    # Aykırı değerleri kaldır
    q = np.nanquantile(diff, [0.02, 0.98])
    diff[(diff < q[0]) | (diff > q[1])] = np.nan

    # Geçerli piksel maskesi (5° < eğim < 75°)
    valid = (slope > 5) & (slope < 75) & np.isfinite(diff)

    if valid.sum() < 10:
        warnings.warn("shiftDem: yeterli geçerli piksel yok.")
        return np.zeros(3)

    slope_v  = slope[valid]
    aspect_v = aspect[valid]
    diff_v   = diff[valid]
    norm_diff = diff_v / np.tan(np.radians(slope_v))

    # Kosinüs eğrisi fit: f(x) = A*cos(B - x) + C
    def cosine_model(params, x_data):
        A, B, C = params
        return A * np.cos(np.radians(B - x_data)) + C

    def residuals(params):
        return cosine_model(params, aspect_v) - norm_diff

    try:
        result = least_squares(residuals, [0.0, 0.0, 0.0],
                               method='lm', max_nfev=500)
        A, B, C = result.x
    except Exception:
        return np.zeros(3)

    vShift = np.array([
        A * np.sin(np.radians(B)) / scale[0],
        A * np.cos(np.radians(B)) / scale[1],
        C * np.tan(np.radians(np.mean(slope_v))),
    ])

    return vShift


# ===========================================================================
# READ GEOTIFF REGION — GeoTIFF Bölge Okuma
# MATLAB karşılığı: shared/readGeotiffRegion.m
# ===========================================================================

def read_geotiff_region(win, path, buffer_pct=None):
    """
    GeoTIFF dosyasının belirli bir coğrafi bölgesini okur.

    MATLAB imzası:
        [mData, vX, vY] = readGeotiffRegion(mWin, strPath, dBuffPct)

    Parametreler
    ----------
    win        : np.ndarray, shape (2, 2) — [[lon1,lat1],[lon2,lat2]]
    path       : str — GeoTIFF dosya yolu
    buffer_pct : float veya None — pencere etrafına % tampon

    Döndürür
    -------
    data : np.ndarray
    vX   : np.ndarray — boylam vektörü
    vY   : np.ndarray — enlem vektörü (azalan)
    """
    if not HAS_RASTERIO:
        raise ImportError("rasterio gerekli: pip install rasterio")

    win = np.sort(win, axis=0).astype(float)

    if buffer_pct is not None:
        buf = np.diff(win, axis=0)[0] * buffer_pct / 100
        win[0] -= buf
        win[1] += buf

    with rasterio.open(path) as ds:
        bounds = ds.bounds
        res_x = (bounds.right  - bounds.left)   / ds.width
        res_y = (bounds.top    - bounds.bottom)  / ds.height

        # Tam koordinat vektörleri
        vX_full = np.linspace(
            bounds.left  + res_x / 2,
            bounds.right - res_x / 2, ds.width
        )
        vY_full = np.linspace(
            bounds.top    - res_y / 2,
            bounds.bottom + res_y / 2, ds.height
        )

        # Pencereye en yakın indeksler
        ix0 = int(np.argmin(np.abs(vX_full - win[0, 0])))
        ix1 = int(np.argmin(np.abs(vX_full - win[1, 0])))
        iy0 = int(np.argmin(np.abs(vY_full - win[1, 1])))
        iy1 = int(np.argmin(np.abs(vY_full - win[0, 1])))

        ix0, ix1 = sorted([ix0, ix1])
        iy0, iy1 = sorted([iy0, iy1])

        if ix1 <= ix0 or iy1 <= iy0:
            raise ValueError(
                "Boş bölge. Koordinat sistemlerini ve uzamsal örtüşmeyi kontrol edin."
            )

        vX = vX_full[ix0:ix1 + 1]
        vY = vY_full[iy0:iy1 + 1]

        window = rasterio.windows.Window(
            col_off=ix0, row_off=iy0,
            width=ix1 - ix0 + 1, height=iy1 - iy0 + 1
        )
        data = ds.read(1, window=window).astype(float)
        nodata = ds.nodata
        if nodata is not None:
            data[data == nodata] = np.nan

    return data, vX, vY


# ===========================================================================
# LL2PS / PS2LL — Polar Stereografik Koordinat Dönüşümleri
# MATLAB karşılığı: shared/ll2ps.m, shared/ps2ll.m
# ===========================================================================

# WGS84 sabitleri
_WGS84_A  = 6378137.0
_WGS84_F  = 1 / 298.257223563
_WGS84_B  = _WGS84_A * (1 - _WGS84_F)
_WGS84_E  = np.sqrt((_WGS84_A**2 - _WGS84_B**2) / _WGS84_A**2)


def ll2ps(pts, params):
    """
    WGS84 enlem/boylam → Polar Stereografik koordinat dönüşümü.

    MATLAB imzası:
        mPts = ll2ps(mPts, vParams)

    Parametreler
    ----------
    pts    : np.ndarray, shape (N, 2+) — [lon, lat, ...]
    params : array-like, shape (5,)
        [lon0, lat0, false_easting, false_northing, scale_factor]

    Döndürür
    -------
    np.ndarray, shape (N, 2+) — [easting, northing, ...]
    """
    pts = np.array(pts, dtype=float)
    lon0, lat0, FE, FN, k0 = params

    lats = pts[:, 1]
    if np.all(lats > 0):
        hemisphere = 'N'
    elif np.all(lats <= 0):
        hemisphere = 'S'
    else:
        raise ValueError("Giriş koordinatları hem pozitif hem negatif enlem içeremez.")

    lon = np.radians(pts[:, 0])
    lat = np.radians(np.abs(pts[:, 1]))
    lat0_r = np.radians(abs(lat0))
    lon0_r = np.radians(lon0)
    e = _WGS84_E
    a = _WGS84_A

    # t parametresi
    t = np.tan(np.pi / 4 - lat / 2) / \
        ((1 - e * np.sin(lat)) / (1 + e * np.sin(lat))) ** (e / 2)

    # rho parametresi
    if abs(lat0_r) == np.pi / 2:
        rho = 2 * a * k0 * t / np.sqrt(
            (1 + e) ** (1 + e) * (1 - e) ** (1 - e)
        )
    else:
        t0 = np.tan(np.pi / 4 - lat0_r / 2) / \
             ((1 - e * np.sin(lat0_r)) / (1 + e * np.sin(lat0_r))) ** (e / 2)
        m0 = np.cos(lat0_r) / np.sqrt(1 - e**2 * np.sin(lat0_r)**2)
        rho = a * m0 * t / t0

    if hemisphere == 'N':
        northing = FN - rho * np.cos(lon - lon0_r)
    else:
        northing = FN + rho * np.cos(lon - lon0_r)

    easting = FE + rho * np.sin(lon - lon0_r)

    result = pts.copy()
    result[:, 0] = easting
    result[:, 1] = northing
    return result


def ps2ll(pts, params):
    """
    Polar Stereografik → WGS84 enlem/boylam koordinat dönüşümü.

    MATLAB imzası:
        mPts = ps2ll(mPts, vParams)

    Parametreler
    ----------
    pts    : np.ndarray, shape (N, 2+) — [easting, northing, ...]
    params : array-like, shape (5,)
        [lon0, lat0, false_easting, false_northing, scale_factor]

    Döndürür
    -------
    np.ndarray, shape (N, 2+) — [lon, lat, ...]
    """
    pts = np.array(pts, dtype=float)
    lon0, lat0, FE, FN, k0 = params
    hemisphere = 'N' if lat0 >= 0 else 'S'

    lat0_r = np.radians(abs(lat0))
    lon0_r = np.radians(lon0)
    e = _WGS84_E
    a = _WGS84_A

    x = pts[:, 0] - FE
    y = pts[:, 1] - FN

    # Boylam
    if hemisphere == 'N':
        lon = lon0_r + np.arctan2(x, -y)
    else:
        lon = lon0_r + np.arctan2(x, y)

    rho = np.sqrt(x**2 + y**2)

    # t parametresi
    if abs(lat0_r) == np.pi / 2:
        t = rho * np.sqrt(
            (1 + e) ** (1 + e) * (1 - e) ** (1 - e)
        ) / (2 * a * k0)
    else:
        t0 = np.tan(np.pi / 4 - lat0_r / 2) / \
             ((1 - e * np.sin(lat0_r)) / (1 + e * np.sin(lat0_r))) ** (e / 2)
        m0 = np.cos(lat0_r) / np.sqrt(1 - e**2 * np.sin(lat0_r)**2)
        t = rho * t0 / (a * m0)

    # İzometrik enlem
    chi = np.pi / 2 - 2 * np.arctan(t)

    # Seri açılımı ile coğrafi enlem
    e2, e4, e6, e8 = e**2, e**4, e**6, e**8
    A = e2/2  + 5*e4/24   + e6/12    + 13*e8/360
    B = 7*e4/48 + 29*e6/240 + 811*e8/11520
    C = 7*e6/120 + 81*e8/1120
    D = 4279*e8/161280

    lat = (chi
           + A * np.sin(2*chi)
           + B * np.sin(4*chi)
           + C * np.sin(6*chi)
           + D * np.sin(8*chi))

    if hemisphere == 'N':
        lon = np.mod(lon + np.pi, 2 * np.pi) - np.pi
    else:
        lat = -lat
        lon = np.mod(lon + np.pi, 2 * np.pi) - np.pi

    result = pts.copy()
    result[:, 0] = np.degrees(lon)
    result[:, 1] = np.degrees(lat)
    return result


# ===========================================================================
# CHECK INPUT — GeoTIFF / Shapefile WGS84 Doğrulaması
# MATLAB karşılığı: shared/checkInput.m
# ===========================================================================

def check_input(file_path):
    """
    GeoTIFF veya shapefile'ın WGS84 kullandığını doğrular.

    MATLAB imzası:
        checkInput(strFileIn)

    Parametreler
    ----------
    file_path : str veya Path

    Yükseltir
    ---------
    ValueError : WGS84 uyumsuzluğu
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    if ext == '.tif':
        if not HAS_RASTERIO:
            raise ImportError("rasterio gerekli.")
        with rasterio.open(path) as ds:
            crs = ds.crs
            if crs is None:
                raise ValueError("Raster CRS bilgisi eksik.")
            epsg = crs.to_epsg()
            if epsg != 4326:
                raise ValueError(
                    f"Raster WGS84 (EPSG:4326) kullanmalıdır. "
                    f"Bulunan: EPSG:{epsg}"
                )

    elif ext == '.shp':
        prj_path = path.with_suffix('.prj')
        if not prj_path.exists():
            warnings.warn(f".prj dosyası bulunamadı: {prj_path}. WGS84 varsayılıyor.")
            return
        prj = prj_path.read_text()
        if 'WGS' not in prj.upper() and 'GCS_WGS' not in prj:
            raise ValueError(
                "Shapefile WGS 84 coğrafi koordinat sistemini kullanmalıdır."
            )

    else:
        raise ValueError(f"Geçersiz dosya türü: '{ext}'")


# ===========================================================================
# CHECK SHP PATH — Klasördeki Shapefileları Doğrula
# MATLAB karşılığı: shared/checkShpPath.m
# ===========================================================================

def check_shp_path(directory):
    """
    Bir klasördeki tüm shapefile'ları WGS84 açısından doğrular.

    MATLAB imzası:
        checkShpPath(strPath)
    """
    shp_files = get_files(directory, '.shp')
    shp_files = [f for f in shp_files if '.xml' not in str(f)]

    if not shp_files:
        raise ValueError(
            f"Belirtilen klasörde geçerli shapefile bulunamadı: {directory}"
        )

    for f in shp_files:
        check_input(f)


# ===========================================================================
# COLORMAP DEM — DEM Görselleştirme Renk Paleti
# MATLAB karşılığı: shared/colormapDEM.m
# ===========================================================================

def colormap_dem():
    """
    DEM görselleştirmesi için yeşil-sarı-kahve renk paleti döndürür.

    MATLAB imzası:
        mCmap = colormapDEM()

    Döndürür
    -------
    np.ndarray, shape (64, 3) — RGB değerleri [0, 1] aralığında
    """
    return np.array([
        [0.000, 0.400, 0.200], [0.018, 0.420, 0.205],
        [0.036, 0.439, 0.210], [0.054, 0.459, 0.215],
        [0.072, 0.478, 0.220], [0.090, 0.498, 0.225],
        [0.108, 0.517, 0.230], [0.126, 0.537, 0.235],
        [0.144, 0.556, 0.240], [0.162, 0.576, 0.245],
        [0.180, 0.595, 0.250], [0.198, 0.615, 0.255],
        [0.216, 0.634, 0.260], [0.234, 0.654, 0.265],
        [0.252, 0.673, 0.270], [0.270, 0.693, 0.275],
        [0.325, 0.722, 0.310], [0.361, 0.740, 0.341],
        [0.398, 0.757, 0.371], [0.434, 0.775, 0.402],
        [0.471, 0.792, 0.432], [0.507, 0.810, 0.463],
        [0.544, 0.827, 0.494], [0.580, 0.845, 0.524],
        [0.617, 0.862, 0.555], [0.653, 0.880, 0.585],
        [0.690, 0.898, 0.616], [0.726, 0.915, 0.647],
        [0.763, 0.933, 0.677], [0.799, 0.950, 0.708],
        [0.835, 0.968, 0.738], [0.872, 0.985, 0.769],
        [0.883, 0.985, 0.769], [0.872, 0.968, 0.738],
        [0.861, 0.950, 0.708], [0.851, 0.933, 0.677],
        [0.840, 0.915, 0.647], [0.829, 0.897, 0.616],
        [0.819, 0.880, 0.585], [0.808, 0.862, 0.555],
        [0.797, 0.844, 0.524], [0.786, 0.827, 0.494],
        [0.775, 0.809, 0.463], [0.765, 0.792, 0.432],
        [0.754, 0.774, 0.402], [0.743, 0.756, 0.371],
        [0.733, 0.739, 0.341], [0.722, 0.721, 0.310],
        [0.700, 0.684, 0.289], [0.679, 0.647, 0.269],
        [0.658, 0.610, 0.248], [0.636, 0.573, 0.227],
        [0.615, 0.536, 0.207], [0.593, 0.499, 0.186],
        [0.572, 0.462, 0.165], [0.550, 0.425, 0.145],
        [0.529, 0.388, 0.124], [0.507, 0.351, 0.103],
        [0.486, 0.315, 0.083], [0.464, 0.278, 0.062],
        [0.443, 0.241, 0.041], [0.421, 0.204, 0.021],
        [0.400, 0.167, 0.000],
    ])


# ===========================================================================
# BASIT TEST
# ===========================================================================

if __name__ == '__main__':
    print("shared_utils.py — Temel testler\n")
    rng = np.random.default_rng(42)

    # TEST 1: get_files
    print("TEST 1: get_files")
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / 'a.tif').touch()
        (Path(d) / 'b.shp').touch()
        (Path(d) / 'sub').mkdir()
        (Path(d) / 'sub' / 'c.tif').touch()
        tifs = get_files(d, '.tif')
        assert len(tifs) == 2, f"Beklenen 2, alınan {len(tifs)}"
    print("  TEST 1 BASARILI ✓\n")

    # TEST 2: make_spatial_ref_vecs
    print("TEST 2: make_spatial_ref_vecs")
    sR = {'Lonlim': [30.0, 31.0], 'Latlim': [39.0, 40.0], 'RasterSize': (10, 20)}
    vX, vY = make_spatial_ref_vecs(sR, 'full')
    assert len(vX) == 20 and len(vY) == 10
    assert vX[0] > 30.0 and vX[-1] < 31.0
    print(f"  vX: {vX[0]:.4f}…{vX[-1]:.4f}, vY: {vY[0]:.4f}…{vY[-1]:.4f}")
    print("  TEST 2 BASARILI ✓\n")

    # TEST 3: neighbors
    print("TEST 3: neighbors")
    mI = np.arange(1, 26, dtype=float).reshape(5, 5)
    N = neighbors(mI, interp=True)
    assert N.shape == (25, 8)
    print(f"  Çıktı boyutu: {N.shape}")
    print("  TEST 3 BASARILI ✓\n")

    # TEST 4: ll2ps / ps2ll gidiş-dönüş
    print("TEST 4: ll2ps / ps2ll gidiş-dönüş (Antarktika UPS)")
    params_s = [0.0, -90.0, 2000000.0, 2000000.0, 0.994]
    pts_ll = np.array([[-45.0, -75.0], [90.0, -80.0], [135.0, -70.0]])
    pts_ps = ll2ps(pts_ll.copy(), params_s)
    pts_back = ps2ll(pts_ps.copy(), params_s)
    err = np.max(np.abs(pts_back[:, :2] - pts_ll))
    print(f"  Gidiş-dönüş hata: {err:.2e} derece")
    assert err < 1e-6, f"Hata çok büyük: {err}"
    print("  TEST 4 BASARILI ✓\n")

    # TEST 5: block_process
    print("TEST 5: block_process")
    def const_fn(s):
        ix, iy = s['count_x'], s['count_y']
        idx_x, idx_y = s['index_x'], s['index_y']
        h = idx_y[iy+1] - idx_y[iy]
        w = idx_x[ix+1] - idx_x[ix]
        return np.ones((h, w))
    grid = block_process((20, 30), 10, 'float64', const_fn)
    assert grid.shape == (20, 30)
    assert np.all(grid == 1.0)
    print(f"  Izgara boyutu: {grid.shape}, tüm değerler 1.0 ✓")
    print("  TEST 5 BASARILI ✓\n")

    # TEST 6: colormap_dem
    print("TEST 6: colormap_dem")
    cmap = colormap_dem()
    assert cmap.shape == (63, 3)
    assert cmap.min() >= 0.0 and cmap.max() <= 1.0
    print(f"  Renk paleti boyutu: {cmap.shape}")
    print("  TEST 6 BASARILI ✓\n")

    print("Tüm temel testler geçti ✓")
