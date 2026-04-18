"""
geo_optimize.py

MATLAB kaynak dosyaları:
  - main/shared/points2grid.m
  - main/shared/normals.m
  - main/shared/transformUsingSolverVar.m
  - main/shared/ll2utm.m
  - main/shared/alignDem.m
  - main/shared/optimizeDem.m
  - main/3_georef/geoOptiTrans.m
  - main/3_georef/geoInitTrans.m
  - main/3_georef/geoOptimize.m
  - main/3_georef/geoSamplePoints.m

Açıklama:
  Üçgenlenen Hexagon nokta bulutunu referans DEM ile hizalar ve
  jeoreferanslama dönüşümünü optimize eder.

Bağımlılıklar:
  - numpy   (pip install numpy)
  - scipy   (pip install scipy)
  - pyproj  (pip install pyproj)
  - open3d  (pip install open3d)
  - rasterio(pip install rasterio)

Dahili bağımlılıklar:
  - utils.py (make_rot_mat)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
from utils import make_rot_mat

try:
    from pyproj import Proj, Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    warnings.warn("pyproj kurulu değil. Koordinat dönüşümleri çalışmayacak.")

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    warnings.warn("open3d kurulu değil. ICP hizalama çalışmayacak.")


# ===========================================================================
# LL2UTM — Coğrafi → UTM Koordinat Dönüşümü
# MATLAB karşılığı: shared/ll2utm.m
# ===========================================================================

def ll2utm(pts, zone=None, hemisphere=None):
    """
    WGS84 enlem/boylam koordinatlarını UTM'ye dönüştürür.

    MATLAB imzası:
        [mPts, iZ, strH] = ll2utm(mPts, iZ, strH)

    Parametreler
    ----------
    pts        : np.ndarray, shape (N, 2+) — [lon, lat, ...] sütunları
    zone       : int veya None  — UTM dilimi (None ise otomatik)
    hemisphere : str veya None  — 'N' veya 'S' (None ise otomatik)

    Döndürür
    -------
    pts_utm    : np.ndarray, shape (N, 2+) — [easting, northing, ...]
    zone       : int
    hemisphere : str
    """
    if not HAS_PYPROJ:
        raise ImportError("pyproj gerekli: pip install pyproj")

    pts = np.array(pts, dtype=float)
    valid = ~np.any(np.isnan(pts[:, :2]), axis=1)

    lon = pts[valid, 0]
    lat = pts[valid, 1]

    if zone is None:
        zone = int(np.floor(np.mean(lon) / 6) + 31)
    if hemisphere is None:
        hemisphere = 'N' if np.mean(lat) >= 0 else 'S'

    epsg = 32600 + zone if hemisphere == 'N' else 32700 + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}",
                                        always_xy=True)

    easting, northing = transformer.transform(lon, lat)

    pts_utm = pts.copy()
    pts_utm[valid, 0] = easting
    pts_utm[valid, 1] = northing

    return pts_utm, zone, hemisphere


def utm2ll(pts_utm, zone, hemisphere):
    """UTM → WGS84 koordinat dönüşümü."""
    if not HAS_PYPROJ:
        raise ImportError("pyproj gerekli: pip install pyproj")

    pts_utm = np.array(pts_utm, dtype=float)
    epsg = 32600 + zone if hemisphere == 'N' else 32700 + zone
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326",
                                        always_xy=True)

    valid = ~np.any(np.isnan(pts_utm[:, :2]), axis=1)
    lon, lat = transformer.transform(pts_utm[valid, 0], pts_utm[valid, 1])

    pts_ll = pts_utm.copy()
    pts_ll[valid, 0] = lon
    pts_ll[valid, 1] = lat

    return pts_ll


# ===========================================================================
# POINTS2GRID — Düzensiz Noktaları Izgaraya Dönüştür
# MATLAB karşılığı: shared/points2grid.m
# ===========================================================================

def points2grid(pts, vX, vY, method='sparse'):
    """
    Düzensiz aralıklı noktaları düzenli ızgaraya dönüştürür.

    MATLAB imzası:
        mGrid = points2grid(mPts, vX, vY, strType)

    Parametreler
    ----------
    pts    : np.ndarray, shape (3, N) veya (4, N)  — [x, y, z, ...]
    vX     : np.ndarray  — artan X koordinatları
    vY     : np.ndarray  — azalan Y koordinatları
    method : str         — 'sparse' veya 'interp'

    Döndürür
    -------
    np.ndarray, shape (len(vY), len(vX))
    """
    if vX[0] > vX[-1]:
        raise ValueError("vX artan sırada olmalıdır.")
    if len(vY) > 1 and vY[0] < vY[-1]:
        raise ValueError("vY azalan sırada olmalıdır.")

    dX = np.abs(np.diff(vX)).mean() if len(vX) > 1 else 1.0
    dY = np.abs(np.diff(vY)).mean() if len(vY) > 1 else 1.0

    if method == 'sparse':
        return _sparse_grid(pts, vX, vY, dX, dY)
    else:
        return _interp_grid(pts, vX, vY)


def _sparse_grid(pts, vX, vY, dX, dY):
    """Seyrek matris yöntemi ile ızgara oluşturur."""
    ny = len(vY)
    nx = len(vX)

    # Piksel indekslerine çevir
    ix = np.round((pts[0] - vX[0]) / dX).astype(int)
    iy = np.round((vY[0] - pts[1]) / dY).astype(int)   # y azalan

    # Sınır dışı noktaları kaldır
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix = ix[valid]; iy = iy[valid]; z = pts[2, valid]

    if len(z) == 0:
        return np.full((ny, nx), np.nan)

    # Ortalama değer (çakışan noktalar için)
    grid_sum   = np.zeros((ny, nx))
    grid_count = np.zeros((ny, nx))
    np.add.at(grid_sum,   (iy, ix), z)
    np.add.at(grid_count, (iy, ix), 1)

    grid = np.full((ny, nx), np.nan)
    mask = grid_count > 0
    grid[mask] = grid_sum[mask] / grid_count[mask]

    return grid


def _interp_grid(pts, vX, vY):
    """Dağınık veri interpolasyonu ile ızgara oluşturur."""
    from scipy.interpolate import griddata
    mX, mY = np.meshgrid(vX, vY)
    grid = griddata(
        pts[:2].T, pts[2],
        (mX, mY), method='linear'
    )
    return grid


# ===========================================================================
# NORMALS — Yüzey Normallerini Hesapla
# MATLAB karşılığı: shared/normals.m
# ===========================================================================

def compute_normals(mX, mY, mZ, interp=True):
    """
    Düzenli ızgara üzerindeki yüzey normallerini hesaplar.

    MATLAB imzası:
        mNormals = normals(mX, mY, mZ, lInterp)

    Parametreler
    ----------
    mX, mY, mZ : np.ndarray, shape (H, W)
    interp     : bool — NaN'ları komşularla doldur

    Döndürür
    -------
    normals : np.ndarray, shape (H*W, 3)
    """
    dX = mX[0, 1] - mX[0, 0]
    dY = mY[1, 0] - mY[0, 0]

    if interp:
        # NaN'ları kübik interpolasyon ile doldur
        from scipy.ndimage import generic_filter
        mZ_filled = mZ.copy()
        nan_mask = np.isnan(mZ_filled)
        if nan_mask.any():
            # Basit: NaN komşularının ortalaması
            def fill_nan(arr):
                c = arr[len(arr)//2]
                if np.isnan(c):
                    valid = arr[~np.isnan(arr)]
                    return valid.mean() if len(valid) > 0 else np.nan
                return c
            mZ_filled = generic_filter(mZ_filled, fill_nan, size=3)
    else:
        mZ_filled = mZ.copy()

    # Gradyanlar (merkezi fark)
    p = np.gradient(mZ_filled, dX, axis=1)   # ∂Z/∂X
    q = np.gradient(mZ_filled, dY, axis=0)   # ∂Z/∂Y

    # Normalize edilmiş normaller
    denom = np.sqrt(1 + p**2 + q**2)
    nx = -p / denom
    ny = -q / denom
    nz = np.ones_like(p) / denom

    normals = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)
    return normals


# ===========================================================================
# GEO SAMPLE POINTS
# MATLAB karşılığı: 3_georef/geoSamplePoints.m
# ===========================================================================

def geo_sample_points(pts, n_pts):
    """
    Nokta bulutundan rastgele alt küme seçer.

    MATLAB imzası:
        [mPts, vIdx] = geoSamplePoints(mPts, iNumPts)

    Parametreler
    ----------
    pts   : np.ndarray, shape (D, N) veya (N, D)
    n_pts : int

    Döndürür
    -------
    pts_sample : np.ndarray — aynı yönelimde
    idx        : np.ndarray — seçilen indeksler
    """
    transposed = False
    if pts.shape[0] > pts.shape[1]:
        pts = pts.T
        transposed = True

    n = pts.shape[1]
    idx = np.random.choice(n, size=min(n_pts, n), replace=False)
    pts_sample = pts[:, idx]

    if transposed:
        pts_sample = pts_sample.T

    return pts_sample, idx


# ===========================================================================
# TRANSFORM USING SOLVER VAR
# MATLAB karşılığı: shared/transformUsingSolverVar.m
# ===========================================================================

def transform_using_solver_var(pts, solver_info):
    """
    Solver değişkenlerini kullanarak nokta bulutunu dönüştürür.

    MATLAB imzası:
        mPts = transformUsingSolverVar(mPts, sInput)

    Parametreler
    ----------
    pts         : np.ndarray, shape (4, N) — homojen noktalar
    solver_info : dict
        'scale_bounds'     : np.ndarray (2, M) — ölçek sınırları
        'rotation_center'  : np.ndarray (3,)
        'variables_index'  : np.ndarray (M,)  — değişken konumları
        'variables'        : np.ndarray (M,)  — optimize edilmiş değerler
        'direction'        : str — 'forward' veya 'inverse'
        'poly_surf'        : dict veya None   — polinom yüzey düzeltmesi

    Döndürür
    -------
    np.ndarray, shape (4, N)
    """
    mBnd   = solver_info['scale_bounds']
    vC     = solver_info['rotation_center']
    vIdx   = solver_info['variables_index']
    vVar   = solver_info['variables'].copy()
    direction = solver_info.get('direction', 'forward')

    # Değişkenleri ölçekten geri çevir
    for i in range(len(vVar)):
        vVar[i] = mBnd[0, i] + vVar[i] * (mBnd[1, i] - mBnd[0, i])

    # Tam değişken dizisi: [rx,ry,rz, tx,ty,tz, sx,sy,sz, s_global]
    vVarO = np.array([0.,0.,0., 0.,0.,0., 1.,1.,1., 1.])
    for i, idx in enumerate(vIdx):
        vVarO[idx] = vVar[i]

    R = np.eye(4)
    R[:3, :3] = make_rot_mat(vVarO[0], vVarO[1], vVarO[2])
    t = vVarO[3:6]
    s = vVarO[6:9]
    sg = vVarO[9]

    pts = pts.copy().astype(float)

    if direction == 'forward':
        pts[0] -= vC[0]; pts[1] -= vC[1]; pts[2] -= vC[2]
        pts[0] *= s[0];  pts[1] *= s[1];  pts[2] *= s[2]
        pts[:3] *= sg
        pts = R @ pts
        pts[0] += vC[0]; pts[1] += vC[1]; pts[2] += vC[2]
        pts[0] += t[0];  pts[1] += t[1];  pts[2] += t[2]

        poly = solver_info.get('poly_surf')
        if isinstance(poly, np.ndarray) and poly.size > 0:
            pts[2] += _eval_poly_surf(poly, pts[:2])

    elif direction == 'inverse':
        poly = solver_info.get('poly_surf')
        if isinstance(poly, np.ndarray) and poly.size > 0:
            pts[2] -= _eval_poly_surf(poly, pts[:2])

        pts[0] -= t[0]; pts[1] -= t[1]; pts[2] -= t[2]
        pts[0] -= vC[0]; pts[1] -= vC[1]; pts[2] -= vC[2]
        pts = np.linalg.solve(R, pts)
        pts[:3] /= sg
        pts[0] /= s[0]; pts[1] /= s[1]; pts[2] /= s[2]
        pts[0] += vC[0]; pts[1] += vC[1]; pts[2] += vC[2]

    return pts


def _eval_poly_surf(poly_coeffs, pts_xy):
    """3. derece polinom yüzeyi değerlendirir."""
    x, y = pts_xy[0], pts_xy[1]
    terms = [x, y, x*y, x**2, y**2, x**3, x**2*y, x*y**2, y**3]
    result = np.zeros(x.shape)
    for i, t in enumerate(terms):
        if i < len(poly_coeffs):
            result += poly_coeffs[i] * t
    return result


# ===========================================================================
# ALIGN DEM — ICP ile DEM Hizalama
# MATLAB karşılığı: shared/alignDem.m
# ===========================================================================

def align_dem(pts, dem_ref, mask, vX, vY):
    """
    ICP algoritması ile hareketli nokta bulutunu referans DEM'e hizalar.

    MATLAB imzası:
        mT = alignDem(mPts, mDem, lM, vX, vY)

    Parametreler
    ----------
    pts     : np.ndarray, shape (4, N) — homojen UTM noktaları
    dem_ref : np.ndarray, shape (H, W) — referans DEM
    mask    : np.ndarray, shape (H, W) — kararsız arazi maskesi
    vX      : np.ndarray — UTM X koordinatları (artan)
    vY      : np.ndarray — UTM Y koordinatları (azalan)

    Döndürür
    -------
    T : np.ndarray, shape (4, 4) — homojen dönüşüm matrisi
    """
    if not HAS_OPEN3D:
        warnings.warn("open3d kurulu değil, ICP atlanıyor (birim matris).")
        return np.eye(4)

    mX, mY = np.meshgrid(vX, vY)

    # Hareketli nokta bulutu (ızgara)
    dem_moving = points2grid(pts[:3], vX, vY, 'sparse')
    valid_m = ~np.isnan(dem_moving)
    pts_moving = np.column_stack([
        mX[valid_m], mY[valid_m], dem_moving[valid_m]
    ])

    # Normallerini hesapla
    n_moving = compute_normals(mX, mY, dem_moving, interp=True)
    n_moving = n_moving[valid_m.ravel()]

    # Alt küme seç
    pts_moving, idx = geo_sample_points(pts_moving.T, 10000)
    pts_moving = pts_moving.T
    n_moving = n_moving[idx]          # örnekleme sonrası normalları da kırp

    # NaN içerenleri kaldır
    valid = np.all(np.isfinite(n_moving), axis=1)
    pts_moving = pts_moving[valid]
    n_moving   = n_moving[valid]

    # Referans nokta bulutu
    valid_r = ~np.isnan(dem_ref) & ~mask
    pts_ref = np.column_stack([
        mX[valid_r], mY[valid_r], dem_ref[valid_r]
    ])
    n_ref = compute_normals(mX, mY, dem_ref, interp=True)
    n_ref = n_ref[valid_r.ravel()]
    valid_ref = np.all(np.isfinite(n_ref), axis=1)   # ayrı değişken
    pts_ref   = pts_ref[valid_ref]
    n_ref     = n_ref[valid_ref]                      # referans normalları da filtrele

    # open3d ICP
    pcd_moving = o3d.geometry.PointCloud()
    pcd_moving.points  = o3d.utility.Vector3dVector(pts_moving)
    pcd_moving.normals = o3d.utility.Vector3dVector(n_moving)

    pcd_ref = o3d.geometry.PointCloud()
    pcd_ref.points  = o3d.utility.Vector3dVector(pts_ref)
    pcd_ref.normals = o3d.utility.Vector3dVector(n_ref)    # PointToPlane için zorunlu

    result = o3d.pipelines.registration.registration_icp(
        pcd_moving, pcd_ref,
        max_correspondence_distance=50.0,
        estimation_method=o3d.pipelines.registration.
            TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100, relative_rmse=0.05
        )
    )

    return np.array(result.transformation)


# ===========================================================================
# OPTIMIZE DEM — Downhill Simplex Optimizasyonu
# MATLAB karşılığı: shared/optimizeDem.m
# ===========================================================================

def optimize_dem(pts, dem_ref, mask, vX, vY, opt):
    """
    DEM yönelimini referans DEM'e göre downhill simplex ile optimize eder.

    Parametreler
    ----------
    pts     : np.ndarray, shape (4, N)
    dem_ref : np.ndarray
    mask    : np.ndarray, bool
    vX, vY  : np.ndarray
    opt     : dict
        'rotation'    : list [rx, ry, rz] sınırları (derece)
        'translation' : list [tx, ty, tz] sınırları (metre)
        'scale'       : list [sx, sy, sz] — NaN ise dahil edilmez
        'globalScale' : float
        'maxIterations': int
        'polySurf'    : bool

    Döndürür
    -------
    sOutput : dict — optimizasyon sonuçları
    """
    mX, mY = np.meshgrid(vX, vY)

    # Değişken sınırları
    v0 = np.array([0.,0.,0., 0.,0.,0., 1.,1.,1., 1.])
    bounds_delta = np.array([
        opt['rotation'][0],  opt['rotation'][1],  opt['rotation'][2],
        opt['translation'][0], opt['translation'][1], opt['translation'][2],
        opt['scale'][0] if not np.isnan(opt['scale'][0]) else 0,
        opt['scale'][1] if not np.isnan(opt['scale'][1]) else 0,
        opt['scale'][2] if not np.isnan(opt['scale'][2]) else 0,
        opt['globalScale'],
    ])
    mBnd = np.vstack([v0 - bounds_delta, v0 + bounds_delta])

    # Aktif değişkenler
    active = bounds_delta > 0
    vIdx  = np.where(active)[0]
    v_active = v0[active]
    mBnd_active = mBnd[:, active]

    vC = pts[:3].mean(axis=1)

    # [0,1] aralığına ölçekle
    def scale_v(v):
        rng = mBnd_active[1] - mBnd_active[0] + 1e-12
        return (v - mBnd_active[0]) / rng

    def descale_v(vs):
        rng = mBnd_active[1] - mBnd_active[0] + 1e-12
        return mBnd_active[0] + vs * rng

    # İzleme değişkenleri
    state = {
        'best_cost': np.inf, 'best_var': scale_v(v_active).copy(),
        'rmse': np.nan, 'best_rmse': np.nan,
        'rmse_history': [], 'iter': 0,
    }

    def cost_fn(vs_scaled):
        v = descale_v(vs_scaled)
        vVarO = v0.copy()
        vVarO[vIdx] = v

        solver_info = {
            'scale_bounds':    mBnd_active,
            'rotation_center': vC,
            'variables_index': vIdx,
            'variables':       vs_scaled,
            'direction':       'forward',
        }
        pts_t = transform_using_solver_var(pts.copy(), solver_info)
        dem_t = points2grid(pts_t[:3], vX, vY, 'sparse')

        diff = dem_ref - dem_t
        diff[mask] = np.nan
        q = np.nanquantile(diff, [0.05, 0.95])
        diff[(diff < q[0]) | (diff > q[1])] = np.nan

        valid = ~np.isnan(diff)
        if valid.sum() == 0:
            return 1e12

        cost = np.nansum(diff**2)
        rmse = np.sqrt(np.nansum(diff**2) / valid.sum())
        state['rmse'] = rmse
        state['rmse_history'].append(rmse)
        state['iter'] += 1

        if cost < state['best_cost']:
            state['best_cost'] = cost
            state['best_var']  = vs_scaled.copy()
            state['best_rmse'] = rmse

        # Erken durdurma: son 10 iterasyonda değişim < 0.1 m
        if len(state['rmse_history']) > 10:
            recent = state['rmse_history'][-10:]
            if max(recent) - min(recent) < 0.1:
                raise StopIteration("Yakınsama sağlandı")

        return cost

    v_init = scale_v(v_active)

    try:
        minimize(
            cost_fn, v_init,
            method='Nelder-Mead',
            options={
                'maxiter':   opt.get('maxIterations', 200),
                'xatol':     1e-6,
                'fatol':     1e-6,
                'disp':      False,
            }
        )
    except StopIteration:
        pass

    sOutput = {
        'scale_bounds':     mBnd_active,
        'rotation_center':  vC,
        'variables_index':  vIdx,
        'variables':        state['best_var'],
        'poly_surf':        None,
        'direction':        'forward',
        'verticalRMSE':     state['best_rmse'],
    }

    return sOutput


# ===========================================================================
# GEO OPTI TRANS
# MATLAB karşılığı: 3_georef/geoOptiTrans.m
# ===========================================================================

def geo_opti_trans(pts, ref_path, shp_path, zone, hemisphere, opt,
                   get_ref_dem_fn):
    """
    Nokta bulutunu referans DEM'e hizalar ve optimize eder.

    Parametreler
    ----------
    pts          : np.ndarray, shape (4, N)
    ref_path     : str  — referans DEM GeoTIFF yolu
    shp_path     : str veya None — kararsız arazi shapefile yolu
    zone, hemisphere : UTM bölge bilgisi
    opt          : dict — optimizasyon parametreleri
    get_ref_dem_fn : callable — (bounds, ref_path) → (dem, vX, vY)

    Döndürür
    -------
    mT      : np.ndarray (4,4) — hizalama dönüşüm matrisi
    sOutput : dict — optimizasyon çıktısı
    """
    # Sınırları ll'ye çevir
    pts_min = pts[:2].min(axis=1)
    pts_max = pts[:2].max(axis=1)
    bounds_utm = np.array([[pts_min[0], pts_min[1]],
                            [pts_max[0], pts_max[1]]])
    bounds_ll = utm2ll(bounds_utm, zone, hemisphere)[:, :2]
    mBnd = np.array([[bounds_ll[:, 0].min(), bounds_ll[:, 1].min()],
                     [bounds_ll[:, 0].max(), bounds_ll[:, 1].max()]])

    # Referans DEM'i hazırla
    dem_ref, vLon, vLat = get_ref_dem_fn(mBnd, ref_path)

    # WGS84 → UTM dönüşümü
    mLon, mLat = np.meshgrid(vLon, vLat)
    pts_ref_ll = np.column_stack([mLon.ravel(), mLat.ravel()])
    pts_ref_utm, _, _ = ll2utm(pts_ref_ll, zone, hemisphere)
    dX = dY = 90.0
    vX = np.arange(pts_ref_utm[:, 0].min(), pts_ref_utm[:, 0].max() + dX, dX)
    vY = np.arange(pts_ref_utm[:, 1].max(), pts_ref_utm[:, 1].min() - dY, -dY)

    dem_utm = points2grid(
        np.vstack([pts_ref_utm[:, :2].T, dem_ref.ravel()]),
        vX, vY, 'sparse'
    )

    # Kararsız arazi maskesi
    mask = np.zeros(dem_utm.shape, dtype=bool)

    # ICP hizalama
    mT = align_dem(pts, dem_utm, mask, vX, vY)
    pts_aligned = mT @ pts
    if pts_aligned.shape[0] == 3:
        pts_aligned = np.vstack([pts_aligned, np.ones((1, pts_aligned.shape[1]))])

    # Optimizasyon
    sOutput = optimize_dem(pts_aligned, dem_utm, mask, vX, vY, opt)

    return mT, sOutput


# ===========================================================================
# GEO INIT TRANS
# MATLAB karşılığı: 3_georef/geoInitTrans.m
# ===========================================================================

def geo_init_trans(win_obj, ref_path, shp_path, visualize, get_ref_dem_fn,
                   absor_fn, progress_cb=None):
    """
    Başlangıç jeoreferanslama dönüşümünü hesaplar.

    Parametreler
    ----------
    win_obj       : dict  — pencere nesnesi
    ref_path      : str   — referans DEM yolu
    shp_path      : str   — shapefile yolu
    visualize     : bool
    get_ref_dem_fn: callable
    absor_fn      : callable — estimate_transform_ransac veya absor
    progress_cb   : callable veya None
    """
    if progress_cb:
        progress_cb('başlangıç jeoreferanslama dönüşümleri hesaplanıyor...')

    sGeo   = win_obj['GeorefInfo']
    pts_t  = sGeo['Initial']['GroundControlPoints']['Triangulated']
    pts_w  = sGeo['Initial']['GroundControlPoints']['World']

    # Dünya noktalarını UTM'ye çevir
    pts_w_utm, zone, hemi = ll2utm(pts_w.T[:, :2], None, None)

    # Horn yöntemi ile başlangıç dönüşümü
    pts_w3 = np.vstack([pts_w_utm[:, :2].T,
                        pts_w.T[2:3] if pts_w.shape[0] > 2
                        else np.zeros((1, pts_w_utm.shape[0]))])
    valid = np.all(np.isfinite(pts_t), axis=0) & \
            np.all(np.isfinite(pts_w3), axis=0)
    M = absor_fn(pts_t[:3, valid], pts_w3[:3, valid])

    # Alt küme seç
    tri_pts = win_obj['TriangulatedPoints']
    pts_sample, _ = geo_sample_points(M @ tri_pts, 1_000_000)

    # İyileştirme optimizasyonu
    opt = {
        'rotation':    [45, 45, 45],
        'translation': [10e3, 10e3, 10e3],
        'scale':       [float('nan')] * 3,
        'globalScale': 2,
        'maxIterations': 200,
        'polySurf': False,
        'visualize': visualize,
    }
    mTa, sOutput = geo_opti_trans(
        pts_sample, ref_path, shp_path,
        zone, hemi, opt, get_ref_dem_fn
    )

    if sOutput['verticalRMSE'] > 50:
        raise RuntimeError(
            f"Dikey RMSE çok yüksek ({sOutput['verticalRMSE']:.1f} m). "
            "Farklı bir pencere deneyin."
        )

    # Sonuçları kaydet
    sT = {'trans': M, 'zone': zone, 'hemi': hemi}
    sGeo['Initial']['Triangulated2WorldTransform'] = sT
    sGeo['Initial']['AlignmentOutput']    = mTa
    sGeo['Initial']['OptimizationOutput'] = sOutput
    win_obj['GeorefInfo'] = sGeo

    return win_obj


# ===========================================================================
# BASIT TEST
# ===========================================================================

if __name__ == '__main__':
    print("geo_optimize.py — Temel testler\n")

    rng = np.random.default_rng(42)

    # -------------------------------------------------------------------
    # TEST 1: ll2utm gidiş-dönüş
    # -------------------------------------------------------------------
    print("TEST 1: ll2utm / utm2ll gidiş-dönüş")
    pts_ll = np.array([
        [30.5, 39.8],
        [31.2, 40.1],
        [29.8, 38.5],
    ])
    pts_utm, zone, hemi = ll2utm(pts_ll.copy(), None, None)
    pts_back = utm2ll(pts_utm.copy(), zone, hemi)

    err = np.max(np.abs(pts_back[:, :2] - pts_ll))
    print(f"  UTM dilimi     : {zone}{hemi}")
    print(f"  Gidiş-dönüş hata: {err:.6f} derece (~{err*111000:.2f} m)")
    print(f"  TEST 1 BASARILI ✓\n" if err < 1e-6
          else f"  TEST 1 BASARISIZ ✗\n")

    # -------------------------------------------------------------------
    # TEST 2: points2grid (sparse)
    # -------------------------------------------------------------------
    print("TEST 2: points2grid sparse")
    vX = np.arange(0.0, 100.0, 10.0)
    vY = np.arange(90.0, -1.0, -10.0)
    # Izgara merkezlerinde noktalar oluştur
    mX, mY = np.meshgrid(vX, vY)
    z = np.sin(mX/30) + np.cos(mY/30)
    pts = np.vstack([mX.ravel(), mY.ravel(), z.ravel()])

    grid = points2grid(pts, vX, vY, 'sparse')
    err = np.nanmax(np.abs(grid - z))
    print(f"  Izgara boyutu  : {grid.shape}")
    print(f"  Maksimum hata  : {err:.6f}")
    print(f"  TEST 2 BASARILI ✓\n" if err < 0.01
          else f"  TEST 2 BASARISIZ ✗\n")

    # -------------------------------------------------------------------
    # TEST 3: transform_using_solver_var ileri-geri
    # -------------------------------------------------------------------
    print("TEST 3: transform_using_solver_var ileri-geri")
    n = 20
    pts_orig = np.vstack([
        rng.random((3, n)) * 1000,
        np.ones((1, n))
    ])

    v0 = np.array([0.,0.,0., 0.,0.,0., 1.,1.,1., 1.])
    bounds_d = np.array([5.,5.,5., 100.,100.,100., 0.,0.,0., 0.])
    active = bounds_d > 0
    vIdx_t = np.where(active)[0]
    mBnd_t = np.vstack([v0[active] - bounds_d[active],
                        v0[active] + bounds_d[active]])
    vC_t = pts_orig[:3].mean(axis=1)

    # [0,1] ölçekli test değerleri
    v_test = np.array([0.6, 0.4, 0.55, 0.7, 0.3, 0.8])

    sinfo_fwd = {
        'scale_bounds':    mBnd_t,
        'rotation_center': vC_t,
        'variables_index': vIdx_t,
        'variables':       v_test,
        'direction':       'forward',
    }
    sinfo_inv = dict(sinfo_fwd)
    sinfo_inv['direction'] = 'inverse'

    pts_fwd = transform_using_solver_var(pts_orig.copy(), sinfo_fwd)
    pts_back = transform_using_solver_var(pts_fwd.copy(), sinfo_inv)

    err3 = np.max(np.abs(pts_back - pts_orig))
    print(f"  İleri-geri hata: {err3:.2e}")
    print(f"  TEST 3 BASARILI ✓\n" if err3 < 1e-6
          else f"  TEST 3 BASARISIZ ✗\n")

    # -------------------------------------------------------------------
    # TEST 4: geo_sample_points
    # -------------------------------------------------------------------
    print("TEST 4: geo_sample_points")
    pts_big = rng.random((3, 50000))
    sampled, idx = geo_sample_points(pts_big, 1000)
    print(f"  Giriş boyutu   : {pts_big.shape}")
    print(f"  Örnek boyutu   : {sampled.shape}")
    print(f"  İndeks boyutu  : {idx.shape}")
    ok = sampled.shape == (3, 1000) and idx.shape == (1000,)
    print(f"  TEST 4 BASARILI ✓\n" if ok else f"  TEST 4 BASARISIZ ✗\n")

    # -------------------------------------------------------------------
    # TEST 5: open3d ICP (varsa)
    # -------------------------------------------------------------------
    print("TEST 5: open3d ICP hazırlık")
    if HAS_OPEN3D:
        # Basit iki nokta bulutu
        pts_src = rng.random((100, 3)).astype(np.float64) * 100
        pts_dst = pts_src + np.array([5.0, 3.0, 1.0])

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pts_src)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts_dst)

        result = o3d.pipelines.registration.registration_icp(
            pcd1, pcd2, 20.0,
            estimation_method=o3d.pipelines.registration.
                TransformationEstimationPointToPoint(),
        )
        t_est = result.transformation[:3, 3]
        t_true = np.array([5.0, 3.0, 1.0])
        err5 = np.max(np.abs(t_est - t_true))
        print(f"  ICP öteleme tahmini: {np.round(t_est, 3)}")
        print(f"  Gerçek öteleme     : {t_true}")
        print(f"  Hata               : {err5:.4f} m")
        print(f"  TEST 5 BASARILI ✓" if err5 < 1.0
              else f"  TEST 5 BASARISIZ ✗")
    else:
        print("  open3d kurulu değil — TEST 5 atlandı")
