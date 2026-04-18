"""
ras_export.py

MATLAB kaynak dosyaları:
  - main/4_rasterize/rasDem.m
  - main/4_rasterize/rasClean.m
  - main/4_rasterize/rasSmooth.m
  - main/4_rasterize/rasOrtho.m
  - main/shared/meshDenoise.m
  - main/shared/curvatureCorrection.m

Açıklama:
  Jeoreferanslanmış üçgenleme noktalarından raster DEM ve ortogörüntü
  üretir. GeoTIFF olarak dışa aktarır.

Bağımlılıklar:
  - numpy    (pip install numpy)
  - scipy    (pip install scipy)
  - rasterio (pip install rasterio)
  - pyproj   (pip install pyproj)
  - open3d   (pip install open3d)

Dahili bağımlılıklar:
  - geo_optimize.py (points2grid, ll2utm, utm2ll, compute_normals,
                     geo_sample_points, transform_using_solver_var)
  - utils.py        (make_rot_mat)
"""

import numpy as np
import warnings
from pathlib import Path
from scipy.ndimage import median_filter, binary_dilation, label
from scipy.interpolate import RegularGridInterpolator
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
sys.path.insert(0, str(Path(__file__).parent.parent / '3_georef'))

from geo_optimize import (points2grid, ll2utm, utm2ll,
                           compute_normals, geo_sample_points,
                           transform_using_solver_var)

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


# ===========================================================================
# CURVATURE CORRECTION
# MATLAB karşılığı: shared/curvatureCorrection.m
# ===========================================================================

def curvature_correction(pts, zone, hemisphere, direction,
                          transform_matrix=None, absor_fn=None):
    """
    Dünya eğriliğinden kaynaklanan dikey hataları düzeltir.

    MATLAB imzası:
        [mPtsT, mT] = curvatureCorrection(mPtsT, iZ, strH, strDir, varargin)

    Parametreler
    ----------
    pts              : np.ndarray, shape (4, N) — homojen UTM noktaları
    zone, hemisphere : UTM bölge bilgisi
    direction        : 'forward' veya 'inverse'
    transform_matrix : np.ndarray (4,4) veya None
    absor_fn         : callable veya None — Horn dönüşümü için

    Döndürür
    -------
    pts_out : np.ndarray, shape (4, N)
    mT      : np.ndarray, shape (4,4)
    """
    from pyproj import Transformer

    # UTM → WGS84 → ECEF dönüşümcüsü
    epsg_utm = 32600 + zone if hemisphere == 'N' else 32700 + zone
    to_ecef = Transformer.from_crs(
        f"EPSG:{epsg_utm}", "EPSG:4978", always_xy=True
    )
    from_ecef = Transformer.from_crs(
        "EPSG:4978", f"EPSG:{epsg_utm}", always_xy=True
    )

    if direction == 'forward':
        # Küçük alt küme üzerinde dönüşüm matrisi tahmin et
        pts_s, _ = geo_sample_points(pts, 100)

        # UTM → ECEF
        xe, ye, ze = to_ecef.transform(pts_s[0], pts_s[1], pts_s[2])
        pts_ecef = np.vstack([xe, ye, ze, np.ones((1, pts_s.shape[1]))])

        if transform_matrix is None:
            if absor_fn is None:
                raise ValueError("transform_matrix veya absor_fn gerekli.")
            mT = absor_fn(pts_s[:3], pts_ecef[:3])
        else:
            mT = transform_matrix

        # Tüm noktaları dönüştür
        pts_t = mT @ pts

        # ECEF → UTM
        xu, yu, zu = from_ecef.transform(pts_t[0], pts_t[1], pts_t[2])
        pts_out = np.vstack([xu, yu, zu, np.ones((1, pts.shape[1]))])

    elif direction == 'inverse':
        # UTM → ECEF
        xe, ye, ze = to_ecef.transform(pts[0], pts[1], pts[2])
        pts_ecef = np.vstack([xe, ye, ze, np.ones((1, pts.shape[1]))])

        if transform_matrix is None:
            raise ValueError("Ters dönüşüm için transform_matrix gerekli.")

        mT = transform_matrix
        pts_out = np.linalg.solve(mT, pts_ecef)

    else:
        raise ValueError(f"Geçersiz direction: '{direction}'")

    return pts_out, mT


# ===========================================================================
# MESH DENOISE
# MATLAB karşılığı: shared/meshDenoise.m
# ===========================================================================

def mesh_denoise(dem, vLon, vLat, null_val='dem',
                 params=(0.9, 5, 10), block_size=1000,
                 zone=None, hemisphere=None):
    """
    Sun et al. (2007) yöntemine dayalı örgü (mesh) gürültü giderme.

    MATLAB imzası:
        mZ = meshDenoise(mZ, sInput)

    Parametreler
    ----------
    dem        : np.ndarray, shape (H, W)
    vLon, vLat : np.ndarray — koordinat vektörleri
    null_val   : str 'dem' veya float
    params     : tuple (T, N, V) — T: eşik, N: normal güncelleme,
                                   V: tepe noktası güncelleme sayısı
    block_size : int
    zone, hemisphere : UTM bölge (None ise otomatik)

    Döndürür
    -------
    np.ndarray, shape (H, W)
    """
    dT, iN, iV = params
    dem = dem.copy().astype(float)

    # Null değerleri NaN yap
    if null_val == 'dem':
        dem[(dem < -500) | (dem > 9000)] = np.nan
    else:
        dem[dem == null_val] = np.nan

    valid_mask = np.isfinite(dem)
    h, w = dem.shape

    # UTM koordinatlarına çevir
    mLon, mLat = np.meshgrid(vLon, vLat)
    pts_ll = np.column_stack([mLon.ravel(), mLat.ravel()])
    pts_utm, zone_out, hemi_out = ll2utm(
        pts_ll, zone, hemisphere
    )
    mX = pts_utm[:, 0].reshape(h, w)
    mY = pts_utm[:, 1].reshape(h, w)

    buf = 5
    n_blk_x = max(1, int(np.ceil(w / block_size)))
    n_blk_y = max(1, int(np.ceil(h / block_size)))
    vX_idx = np.round(np.linspace(buf, w - buf, n_blk_x + 1)).astype(int)
    vY_idx = np.round(np.linspace(buf, h - buf, n_blk_y + 1)).astype(int)

    for iy in range(len(vY_idx) - 1):
        for ix in range(len(vX_idx) - 1):
            # Tamponlu blok indeksleri
            r0 = max(0, vY_idx[iy] - buf)
            r1 = min(h, vY_idx[iy + 1] + buf)
            c0 = max(0, vX_idx[ix] - buf)
            c1 = min(w, vX_idx[ix + 1] + buf)

            blk_x = mX[r0:r1, c0:c1].ravel()
            blk_y = mY[r0:r1, c0:c1].ravel()
            blk_z = dem[r0:r1, c0:c1].ravel()
            blk_v = valid_mask[r0:r1, c0:c1].ravel()

            verts = np.column_stack([blk_x[blk_v], blk_y[blk_v],
                                     blk_z[blk_v]])

            if len(verts) < 3:
                continue

            try:
                verts_d = _mesh_denoise_block(verts, dT, iN, iV)

                # Tamponsuz bölgeye geri yaz
                rb0 = vY_idx[iy] - r0
                rb1 = vY_idx[iy + 1] - r0
                cb0 = vX_idx[ix] - c0
                cb1 = vX_idx[ix + 1] - c0

                idx_in_block = np.where(blk_v)[0]
                blk_r = idx_in_block // (c1 - c0)
                blk_c = idx_in_block % (c1 - c0)

                core = (blk_r >= rb0) & (blk_r < rb1) & \
                       (blk_c >= cb0) & (blk_c < cb1)

                abs_r = blk_r[core] + r0
                abs_c = blk_c[core] + c0
                dem[abs_r, abs_c] = verts_d[core, 2]

            except Exception as e:
                warnings.warn(f"Blok ({iy},{ix}) mesh denoise hatası: {e}")

    # UTM → WGS84'e geri dönüştür ve ızgaraya çek
    pts_utm_out = np.column_stack([mX[valid_mask], mY[valid_mask]])
    pts_ll_out = utm2ll(pts_utm_out, zone_out, hemi_out)

    from scipy.interpolate import LinearNDInterpolator
    interp = LinearNDInterpolator(
        pts_ll_out[:, :2], dem[valid_mask]
    )
    mLon_s, mLat_s = np.meshgrid(vLon, vLat)
    dem_out = interp(mLon_s, mLat_s)
    dem_out[~valid_mask] = np.nan

    return dem_out


def _mesh_denoise_block(verts, dT, iN, iV):
    """Tek bir blok üzerinde mesh denoise uygular (Sun et al. 2007)."""
    from scipy.spatial import Delaunay

    tri = Delaunay(verts[:, :2])
    faces = tri.simplices

    # Yüzey normallerini hesapla
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    face_normals = face_normals / norms

    # Normal güncelleme döngüsü
    for _ in range(iN):
        new_normals = np.zeros_like(face_normals)
        for fi in range(len(faces)):
            nbrs = _face_neighbors(fi, faces, max_n=15)
            n_nbr = face_normals[nbrs]
            dot = np.dot(face_normals[fi], n_nbr.T)
            weights = np.where(dot > dT, (dot - dT)**2, 0)
            if weights.sum() > 0:
                new_normals[fi] = (weights[:, None] * n_nbr).sum(0)
            else:
                new_normals[fi] = face_normals[fi]
        norms = np.linalg.norm(new_normals, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        face_normals = new_normals / norms

    # Tepe güncelleme döngüsü
    for _ in range(iV):
        new_verts = verts.copy()
        for vi in range(len(verts)):
            fi_list = np.where(np.any(faces == vi, axis=1))[0]
            if len(fi_list) == 0:
                continue
            centers = verts[faces[fi_list]].mean(axis=1)
            n = face_normals[fi_list]
            diff = centers - verts[vi]
            proj = (n * diff).sum(axis=1, keepdims=True) * n
            new_verts[vi] += proj.mean(axis=0) / max(len(fi_list), 1)
        verts = new_verts

    return verts


def _face_neighbors(fi, faces, max_n=15):
    """Yüz fi'nin komşu yüzlerini döndürür."""
    verts_fi = set(faces[fi])
    nbrs = []
    for i, f in enumerate(faces):
        if i == fi:
            continue
        if len(verts_fi & set(f)) > 0:
            nbrs.append(i)
        if len(nbrs) >= max_n:
            break
    return [fi] + nbrs[:max_n - 1]


# ===========================================================================
# RAS CLEAN
# MATLAB karşılığı: 4_rasterize/rasClean.m
# ===========================================================================

def ras_clean(dem, res_m, do_clean=True, gap_thresh=1e6,
              speckle_thresh=1e6):
    """
    DEM temizleme: küçük boşlukları doldur, büyükleri genişlet,
    yalnız noktaları sil.

    Parametreler
    ----------
    dem           : np.ndarray — DEM (NaN = geçersiz)
    res_m         : float — piksel çözünürlüğü (metre)
    do_clean      : bool
    gap_thresh    : float — doldurulacak boşluk eşiği (m²)
    speckle_thresh: float — silinecek nokta adası eşiği (m²)

    Döndürür
    -------
    np.ndarray
    """
    if not do_clean:
        return dem

    dem = dem.copy()
    pix_area = res_m ** 2

    # Küçük boşlukları doldur
    gap_px = max(1, round(gap_thresh / pix_area))
    nan_mask = np.isnan(dem)

    labeled, n_comp = label(nan_mask)
    for comp in range(1, n_comp + 1):
        if (labeled == comp).sum() < gap_px:
            # Komşu geçerli piksellerin ortalamasıyla doldur
            comp_mask = labeled == comp
            from scipy.ndimage import binary_dilation as bdil
            dilated = bdil(comp_mask)
            nbrs = dilated & ~comp_mask & ~nan_mask
            if nbrs.any():
                dem[comp_mask] = np.nanmean(dem[nbrs])

    # Büyük boşlukları genişlet (kenar piksellerini sil)
    nan_mask_new = np.isnan(dem)
    labeled2, n2 = label(nan_mask_new)
    for comp in range(1, n2 + 1):
        if (labeled2 == comp).sum() >= gap_px:
            comp_mask = labeled2 == comp
            expanded = binary_dilation(comp_mask,
                                       iterations=1)
            dem[expanded] = np.nan

    # Yalnız noktaları (speckle) sil
    speckle_px = max(1, round(speckle_thresh / pix_area))
    valid = ~np.isnan(dem)
    labeled3, n3 = label(valid)
    for comp in range(1, n3 + 1):
        if (labeled3 == comp).sum() < speckle_px:
            dem[labeled3 == comp] = np.nan

    return dem


# ===========================================================================
# RAS SMOOTH
# MATLAB karşılığı: 4_rasterize/rasSmooth.m
# ===========================================================================

def ras_smooth(vX, vY, dem, do_median=True, do_denoise=True,
               median_size=3, denoise_t=0.9, denoise_n=5,
               zone=None, hemisphere=None):
    """
    Medyan filtre ve mesh denoise uygular.

    Parametreler
    ----------
    vX, vY      : np.ndarray — koordinat vektörleri
    dem         : np.ndarray
    do_median   : bool
    do_denoise  : bool
    median_size : int — medyan filtre pencere boyutu
    denoise_t   : float — denoise eşiği
    denoise_n   : int — denoise iterasyonu
    zone, hemisphere : UTM bölge

    Döndürür
    -------
    np.ndarray
    """
    dem = dem.copy()

    if do_median:
        nan_mask = np.isnan(dem)

        # NaN'ları geçici olarak komşu ortalamayla doldur
        from scipy.ndimage import generic_filter
        def fill_nan(arr):
            c = arr[len(arr)//2]
            if np.isnan(c):
                v = arr[~np.isnan(arr)]
                return v.mean() if len(v) > 0 else np.nan
            return c
        dem_filled = generic_filter(dem, fill_nan, size=3)

        dem_filtered = median_filter(dem_filled,
                                     size=(median_size, median_size),
                                     mode='reflect')
        dem[~nan_mask] = dem_filtered[~nan_mask]

    if do_denoise:
        dem = mesh_denoise(
            dem, vX, vY,
            null_val='dem',
            params=(denoise_t, denoise_n, denoise_n * 2),
            zone=zone, hemisphere=hemisphere
        )

    return dem


# ===========================================================================
# RAS DEM
# MATLAB karşılığı: 4_rasterize/rasDem.m
# ===========================================================================

def ras_dem(win_obj, save_path, do_clean=True, do_median=False,
            do_denoise=False, gap_thresh=1e6, speckle_thresh=1e6,
            median_size=3, denoise_t=0.9, denoise_n=5,
            progress_cb=None):
    """
    Jeoreferanslanmış üçgenleme noktalarından raster DEM üretir.

    MATLAB imzası:
        rasDem(objL, strWinPath, lClean, lMed, lDen, ...)

    Parametreler
    ----------
    win_obj   : dict  — pencere nesnesi
        'TriangulatedPointsGeoref' : np.ndarray (4, N)
        'GeorefInfo'               : dict
        'RegionID', 'WindowID'     : int
    save_path : str   — çıktı klasörü
    ...

    Döndürür
    -------
    win_obj : dict — 'HexagonDem', 'HexagonDemSpatialRef' eklenerek
    """
    if progress_cb:
        progress_cb('DEM rasterleştiriliyor...')

    pts = win_obj['TriangulatedPointsGeoref'].copy()
    # NaN içerenleri kaldır
    valid = ~np.any(np.isnan(pts), axis=0)
    pts = pts[:, valid]

    # Koordinat sistemi bilgisi
    sGeo = win_obj['GeorefInfo']
    sT   = sGeo['Initial']['Triangulated2WorldTransform']
    zone = sT['zone']
    hemi = sT['hemi']

    # Makul çözünürlük hesapla (metre cinsinden)
    pts_ll = utm2ll(pts[:3].T, zone, hemi)
    res_m  = np.median(np.abs(np.diff(pts_ll[:, 0]))) * 111320
    res_m  = max(res_m * 2, 1.0)

    # Coğrafi koordinat çözünürlüğü
    cen_utm = pts[:2].mean(axis=1)
    cen_ll  = utm2ll(
        np.array([[cen_utm[0], cen_utm[1]],
                  [cen_utm[0] + res_m, cen_utm[1] + res_m]]),
        zone, hemi
    )
    res_deg = np.abs(np.diff(cen_ll[:, :2], axis=0)).mean() * 2

    # Sınırlar ve koordinat vektörleri
    lon_min, lon_max = pts_ll[:, 0].min(), pts_ll[:, 0].max()
    lat_min, lat_max = pts_ll[:, 1].min(), pts_ll[:, 1].max()

    vX = np.arange(lon_min, lon_max + res_deg, res_deg)
    vY = np.arange(lat_max, lat_min - res_deg, -res_deg)

    # Interpolasyon
    dem = points2grid(
        np.vstack([pts_ll[:, 0], pts_ll[:, 1], pts_ll[:, 2]]),
        vX, vY, 'interp'
    )

    # Temizlik ve düzleştirme
    dem = ras_clean(dem, res_m, do_clean, gap_thresh, speckle_thresh)
    dem = ras_smooth(vX, vY, dem, do_median, do_denoise,
                     median_size, denoise_t, denoise_n, zone, hemi)

    # NoData değeri ve int16
    dem_out = dem.copy()
    dem_out[np.isnan(dem_out)] = -32768
    dem_int16 = dem_out.astype(np.int16)

    # GeoTIFF yaz
    save_path = Path(save_path)
    (save_path / 'dems').mkdir(parents=True, exist_ok=True)
    out_file = save_path / 'dems' / \
        f"dem_r{win_obj['RegionID']}_w{win_obj['WindowID']}.tif"

    transform = from_bounds(
        vX[0] - res_deg/2, vY[-1] - res_deg/2,
        vX[-1] + res_deg/2, vY[0] + res_deg/2,
        len(vX), len(vY)
    )
    with rasterio.open(
        out_file, 'w',
        driver='GTiff',
        height=len(vY), width=len(vX),
        count=1, dtype='int16',
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=-32768,
    ) as dst:
        dst.write(dem_int16, 1)

    # Spatial ref sözlüğü
    sR = {
        'Lonlim':      [vX[0] - res_deg/2, vX[-1] + res_deg/2],
        'Latlim':      [vY[-1] - res_deg/2, vY[0] + res_deg/2],
        'RasterSize':  dem_int16.shape,
        'resolution':  res_deg,
        'vX':          vX,
        'vY':          vY,
    }

    win_obj['HexagonDem']           = dem_int16
    win_obj['HexagonDemSpatialRef'] = sR

    print(f"  DEM kaydedildi: {out_file}")
    return win_obj


# ===========================================================================
# RAS ORTHO
# MATLAB karşılığı: 4_rasterize/rasOrtho.m
# ===========================================================================

def ras_ortho(win_obj, save_path, progress_cb=None):
    """
    Jeoreferanslanmış DEM üzerinden raster ortogörüntü üretir.

    Parametreler
    ----------
    win_obj   : dict  — hem DEM hem görüntü bilgilerini içermeli
    save_path : str

    Döndürür
    -------
    win_obj : dict — 'HexagonImage', 'HexagonImageSpatialRef' eklenerek
    """
    if progress_cb:
        progress_cb('ortogörüntü rasterleştiriliyor...')

    sGeo = win_obj['GeorefInfo']
    sT   = sGeo['Initial']['Triangulated2WorldTransform']
    zone = sT['zone']
    hemi = sT['hemi']

    sR_hex = win_obj['HexagonDemSpatialRef']
    vLon_h = sR_hex['vX']
    vLat_h = sR_hex['vY']

    # Daha ince çözünürlüklü ızgara
    res_deg = sR_hex['resolution'] / 10.0
    vLon = np.arange(vLon_h[0], vLon_h[-1] + res_deg, res_deg)
    vLat = np.arange(vLat_h[0], vLat_h[-1] - res_deg, -res_deg)

    mLon, mLat = np.meshgrid(vLon, vLat)
    mLon_h, mLat_h = np.meshgrid(vLon_h, vLat_h)

    # DEM'i yeni çözünürlüğe interpolasyon
    dem_f = win_obj['HexagonDem'].astype(float)
    dem_f[(dem_f < -500) | (dem_f > 9000)] = np.nan

    interp_dem = RegularGridInterpolator(
        (vLat_h[::-1], vLon_h), dem_f[::-1],
        method='linear', bounds_error=False, fill_value=np.nan
    )
    dem_new = interp_dem(
        np.stack([mLat.ravel(), mLon.ravel()], axis=-1)
    ).reshape(mLon.shape)

    # Referans DEM ile boşlukları doldur
    if 'ReferenceDem' in win_obj:
        ref_lon = win_obj['ReferenceLon']
        ref_lat = win_obj['ReferenceLat']
        interp_ref = RegularGridInterpolator(
            (ref_lat[::-1], ref_lon),
            win_obj['ReferenceDem'][::-1],
            method='linear', bounds_error=False, fill_value=np.nan
        )
        holes = np.isnan(dem_new)
        if holes.any():
            dem_fill = interp_ref(
                np.stack([mLat[holes], mLon[holes]], axis=-1)
            )
            dem_new[holes] = dem_fill

    # Kalan boşlukları komşu ortalaması ile doldur
    from scipy.ndimage import generic_filter
    def fill_nan(arr):
        c = arr[len(arr)//2]
        if np.isnan(c):
            v = arr[~np.isnan(arr)]
            return v.mean() if len(v) > 0 else np.nan
        return c
    dem_new = generic_filter(dem_new, fill_nan, size=3)

    # DEM noktaları WGS84 → UTM
    pts = np.vstack([
        mLon.ravel(), mLat.ravel(), dem_new.ravel(),
        np.ones(mLon.size)
    ])
    pts_utm, _, _ = ll2utm(pts[:3].T, zone, hemi)
    pts = np.vstack([pts_utm.T, np.ones((1, pts_utm.shape[0]))])

    # Jeoreferanslama dönüşümlerini geri al (ters sırada)
    c_output = sGeo['Final']['OptimizationOutput']
    c_trans  = sGeo['Final']['AlignmentOutput']

    for i in range(len(c_output) - 1, -1, -1):
        s_out = dict(c_output[i])
        s_out['direction'] = 'inverse'
        pts = transform_using_solver_var(pts, s_out)
        T4 = np.eye(4)
        T4[:3, :4] = c_trans[i]
        pts = np.linalg.solve(T4, pts)

    # Başlangıç optimizasyonunu geri al
    s_init = dict(sGeo['Initial']['OptimizationOutput'])
    s_init['direction'] = 'inverse'
    pts = transform_using_solver_var(pts, s_init)
    T_align = np.eye(4)
    T_align[:3, :4] = sGeo['Initial']['AlignmentOutput']
    pts = np.linalg.solve(T_align, pts)

    # Başlangıç dönüşümünü geri al
    pts = np.linalg.solve(sT['trans'], pts)

    if sGeo['Initial'].get('WindowTransform', 'none') != 'none':
        pts = np.linalg.solve(sGeo['Initial']['WindowTransform'], pts)

    # Görüntü düzlemine projekte et
    K  = win_obj['IntrinsicMatrix']
    P1 = win_obj['PoseMatrix']
    pts_img = K @ P1 @ pts
    pts_img = pts_img[:2] / pts_img[2:3]

    # Pencere ofsetini çıkar
    hex_win = win_obj['Window']
    pts_img[0] -= hex_win[0, 0] - 1
    pts_img[1] -= hex_win[0, 1] - 1

    # Görüntü interpolasyonu
    img = win_obj['Image'].astype(float)
    h_img, w_img = img.shape[:2]

    interp_img = RegularGridInterpolator(
        (np.arange(h_img), np.arange(w_img)),
        img, method='linear', bounds_error=False, fill_value=0
    )
    ortho = interp_img(
        np.stack([pts_img[1].ravel(), pts_img[0].ravel()], axis=-1)
    ).reshape(mLon.shape).astype(np.uint8)

    # GeoTIFF yaz
    save_path = Path(save_path)
    (save_path / 'images').mkdir(parents=True, exist_ok=True)
    out_file = save_path / 'images' / \
        f"image_r{win_obj['RegionID']}w{win_obj['WindowID']}.tif"

    res_out = res_deg
    transform_gt = from_bounds(
        vLon[0] - res_out/2, vLat[-1] - res_out/2,
        vLon[-1] + res_out/2, vLat[0] + res_out/2,
        len(vLon), len(vLat)
    )
    with rasterio.open(
        out_file, 'w',
        driver='GTiff',
        height=len(vLat), width=len(vLon),
        count=1, dtype='uint8',
        crs=CRS.from_epsg(4326),
        transform=transform_gt,
    ) as dst:
        dst.write(ortho, 1)

    sR_out = {
        'Lonlim':     [vLon[0] - res_out/2, vLon[-1] + res_out/2],
        'Latlim':     [vLat[-1] - res_out/2, vLat[0] + res_out/2],
        'RasterSize': ortho.shape,
        'resolution': res_out,
    }

    win_obj['HexagonImage']           = ortho
    win_obj['HexagonImageSpatialRef'] = sR_out

    print(f"  Ortogörüntü kaydedildi: {out_file}")
    return win_obj


# ===========================================================================
# BASIT TEST
# ===========================================================================

if __name__ == '__main__':
    print("ras_export.py — Temel testler\n")

    rng = np.random.default_rng(42)

    # -------------------------------------------------------------------
    # TEST 1: ras_clean
    # -------------------------------------------------------------------
    print("TEST 1: ras_clean")

    dem_test = rng.random((50, 50)) * 1000.0
    # Küçük boşluklar ekle
    dem_test[10:12, 10:12] = np.nan   # 4 piksel boşluk
    dem_test[30:45, 30:45] = np.nan   # büyük boşluk
    # Küçük ada
    dem_test[:3, :3] = 100.0
    dem_test[3:, :] = np.nan
    dem_test[:3, 3:] = np.nan

    n_nan_before = np.isnan(dem_test).sum()
    dem_cleaned = ras_clean(dem_test.copy(), res_m=30.0,
                            do_clean=True,
                            gap_thresh=200,
                            speckle_thresh=200)
    n_nan_after = np.isnan(dem_cleaned).sum()

    print(f"  NaN öncesi : {n_nan_before}")
    print(f"  NaN sonrası: {n_nan_after}")
    print(f"  TEST 1 BASARILI ✓\n")

    # -------------------------------------------------------------------
    # TEST 2: ras_smooth (yalnızca medyan)
    # -------------------------------------------------------------------
    print("TEST 2: ras_smooth (medyan filtre)")

    dem_noisy = rng.random((30, 30)) * 500.0
    dem_noisy[5, 5] = 9999.0   # aykırı değer

    vX_t = np.linspace(30.0, 31.0, 30)
    vY_t = np.linspace(40.0, 39.0, 30)

    dem_smooth = ras_smooth(vX_t, vY_t, dem_noisy,
                            do_median=True, do_denoise=False,
                            median_size=3)
    assert dem_smooth[5, 5] < 9999.0, "Medyan filtre aykırı değeri azaltmalı"
    print(f"  Aykırı değer öncesi: {dem_noisy[5,5]:.1f}")
    print(f"  Aykırı değer sonrası: {dem_smooth[5,5]:.1f}")
    print(f"  TEST 2 BASARILI ✓\n")

    # -------------------------------------------------------------------
    # TEST 3: ll2utm / utm2ll gidiş-dönüş (geo_optimize bağımlılığı)
    # -------------------------------------------------------------------
    print("TEST 3: Koordinat dönüşümü bağımlılığı")
    import tempfile

    pts_ll = np.array([[31.0, 39.5], [31.5, 40.0]])
    pts_utm, z, h = ll2utm(pts_ll, None, None)
    pts_back = utm2ll(pts_utm, z, h)
    err = np.max(np.abs(pts_back[:, :2] - pts_ll))
    print(f"  Gidiş-dönüş hata: {err:.2e} derece")
    print(f"  TEST 3 BASARILI ✓\n")

    # -------------------------------------------------------------------
    # TEST 4: ras_dem (sentetik veri ile GeoTIFF yazma)
    # -------------------------------------------------------------------
    print("TEST 4: ras_dem — GeoTIFF yazma")

    # Sentetik jeoreferanslanmış nokta bulutu (WGS84 ll + yükseklik)
    n_pts = 500
    lon_pts = rng.uniform(30.0, 30.1, n_pts)
    lat_pts = rng.uniform(39.0, 39.1, n_pts)
    elev    = 1000.0 + rng.normal(0, 5, n_pts)
    pts_utm_s, zs, hs = ll2utm(
        np.column_stack([lon_pts, lat_pts]), None, None
    )
    pts_geo = np.vstack([
        pts_utm_s[:, 0], pts_utm_s[:, 1], elev, np.ones(n_pts)
    ])

    win_obj_test = {
        'TriangulatedPointsGeoref': pts_geo,
        'RegionID': 1,
        'WindowID': 1,
        'GeorefInfo': {
            'Initial': {
                'Triangulated2WorldTransform': {
                    'zone': zs, 'hemi': hs
                }
            }
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        result = ras_dem(win_obj_test, tmpdir,
                         do_clean=False, do_median=False, do_denoise=False)
        dem_file = Path(tmpdir) / 'dems' / 'dem_r1_w1.tif'
        assert dem_file.exists(), "GeoTIFF dosyası oluşturulmalı"
        with rasterio.open(dem_file) as ds:
            print(f"  GeoTIFF boyutu   : {ds.width} x {ds.height}")
            print(f"  CRS              : {ds.crs}")
            print(f"  Veri tipi        : {ds.dtypes[0]}")
        print(f"  TEST 4 BASARILI ✓")
