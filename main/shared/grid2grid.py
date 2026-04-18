"""
grid2grid.py

MATLAB kaynak dosyaları:
  - main/shared/grid2grid.m
  - main/shared/blockProcess.m
  - main/shared/makeSpatialRefVecs.m

Açıklama:
  Düzenli aralıklı bir ızgarayı (GeoTIFF veya matris) yeni bir koordinat
  sistemine yeniden örnekler (resample). İsteğe bağlı koordinat dönüşümü
  ve blok-tabanlı işleme desteği içerir.

Bağımlılıklar:
  - numpy      (pip install numpy)
  - rasterio   (pip install rasterio)
  - scipy      (pip install scipy)
"""

import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path


# ===========================================================================
# MAKE SPATIAL REF VECS
# MATLAB karşılığı: shared/makeSpatialRefVecs.m
# ===========================================================================

def make_spatial_ref_vecs(spatial_ref, vec_type='full'):
    """
    Uzamsal referans bilgisinden X ve Y koordinat vektörleri üretir.

    MATLAB imzası:
        [vX, vY] = makeSpatialRefVecs(sR, strType)

    Parametreler
    ----------
    spatial_ref : dict
        Uzamsal referans bilgisi. Beklenen anahtarlar:
        Coğrafi koordinat sistemi için:
            'DeltaLon', 'DeltaLat', 'Lonlim' (2,), 'Latlim' (2,)
        Yansıtmalı koordinat sistemi için:
            'DeltaX', 'DeltaY', 'XLimWorld' (2,), 'YLimWorld' (2,)
        rasterio.DatasetReader nesnesi de kabul edilir.
    vec_type : str
        'full'   — tam vektörler
        'limits' — yalnızca ilk ve son eleman

    Döndürür
    -------
    vX : np.ndarray  — yatay koordinatlar (batıdan doğuya)
    vY : np.ndarray  — dikey koordinatlar (kuzeyden güneye, azalan)
    """

    # rasterio nesnesi verilmişse dict'e çevir
    if hasattr(spatial_ref, 'transform'):
        ds = spatial_ref
        t = ds.transform
        dx = t.a
        dy = t.e                            # negatif (kuzeyden güneye)
        x_min = t.c + dx / 2               # piksel merkezi
        x_max = x_min + dx * (ds.width - 1)
        y_max = t.f + dy / 2               # piksel merkezi (kuzey)
        y_min = y_max + dy * (ds.height - 1)
        sR = {
            'DeltaX': dx, 'DeltaY': dy,
            'XLimWorld': (t.c, t.c + dx * ds.width),
            'YLimWorld': (t.f + dy * ds.height, t.f),
        }
    else:
        sR = spatial_ref

    # Coğrafi (Lon/Lat) sistem
    if 'DeltaLon' in sR:
        dx = sR['DeltaLon']
        dy = sR['DeltaLat']          # genellikle negatif
        x_start = sR['Lonlim'][0] + dx / 2
        x_end   = sR['Lonlim'][1] - dx / 2
        y_start = sR['Latlim'][1] + dy / 2   # kuzey sınırı
        y_end   = sR['Latlim'][0] - dy / 2   # güney sınırı

    # Yansıtmalı (X/Y) sistem
    elif 'DeltaX' in sR:
        dx = sR['DeltaX']
        dy = sR['DeltaY']
        x_start = sR['XLimWorld'][0] + dx / 2
        x_end   = sR['XLimWorld'][1] - dx / 2
        y_start = sR['YLimWorld'][1] + dy / 2
        y_end   = sR['YLimWorld'][0] - dy / 2

    else:
        raise ValueError("spatial_ref içinde tanınan koordinat anahtarı bulunamadı.")

    if vec_type == 'full':
        vX = np.arange(x_start, x_end + dx * 0.5, dx)
        vY = np.arange(y_start, y_end + dy * 0.5, dy)
    elif vec_type == 'limits':
        vX = np.array([x_start, x_end])
        vY = np.array([y_start, y_end])
    else:
        raise ValueError(f"Geçersiz vec_type: '{vec_type}'. 'full' veya 'limits' olmalı.")

    return vX, vY


# ===========================================================================
# BLOCK PROCESS
# MATLAB karşılığı: shared/blockProcess.m
# ===========================================================================

def block_process(grid_shape, block_size, dtype, func, mat_info=None):
    """
    Büyük ızgaraları bloklar halinde işler.

    MATLAB imzası:
        mGrid = blockProcess(vSz, dBlkSz, strClass, hFun, [sMatInfo])

    Parametreler
    ----------
    grid_shape : tuple (rows, cols)
        İşlenecek ızgaranın toplam boyutu.
    block_size : int
        Her bloğun piksel cinsinden boyutu.
    dtype : str veya np.dtype
        Çıktı dizisinin veri tipi ('double' → float64, 'logical' → bool, vb.)
    func : callable
        Her blok için çağrılacak fonksiyon.
        Girdi: sözlük {'index_x', 'index_y', 'count_x', 'count_y'}
        Çıktı: np.ndarray (blok boyutunda) veya None (atla)
    mat_info : dict veya None
        Belirtilirse sonuç dosyaya yazılır (şu an bellek içi destekleniyor).

    Döndürür
    -------
    np.ndarray, shape grid_shape
    """
    # MATLAB sınıf adlarını numpy dtype'a çevir
    dtype_map = {
        'double':  np.float64,
        'single':  np.float32,
        'logical': bool,
        'uint8':   np.uint8,
        'uint16':  np.uint16,
        'int16':   np.int16,
        'int32':   np.int32,
    }
    np_dtype = dtype_map.get(dtype, dtype)

    rows, cols = grid_shape

    # Blok sınır indeksleri
    num_blk_x = max(1, int(np.ceil(cols / block_size)))
    num_blk_y = max(1, int(np.ceil(rows / block_size)))
    idx_x = np.round(np.linspace(0, cols, num_blk_x + 1)).astype(int)
    idx_y = np.round(np.linspace(0, rows, num_blk_y + 1)).astype(int)

    # Çıktı ızgarasını başlat
    if np_dtype == bool:
        grid = np.zeros(grid_shape, dtype=bool)
    elif np.issubdtype(np_dtype, np.integer):
        grid = np.zeros(grid_shape, dtype=np_dtype)
    else:
        grid = np.full(grid_shape, np.nan, dtype=np_dtype)

    # Her bloğu işle
    for iy in range(len(idx_y) - 1):
        for ix in range(len(idx_x) - 1):

            block_input = {
                'index_x': idx_x,
                'index_y': idx_y,
                'count_x': ix,
                'count_y': iy,
            }

            try:
                block = func(block_input)
            except Exception as e:
                import warnings
                warnings.warn(f"Blok ({iy},{ix}) işlenirken hata: {e}")
                continue

            if block is None:
                continue

            # Beklenen blok boyutu
            row_slice = slice(idx_y[iy], idx_y[iy + 1])
            col_slice = slice(idx_x[ix], idx_x[ix + 1])
            expected_shape = (
                idx_y[iy + 1] - idx_y[iy],
                idx_x[ix + 1] - idx_x[ix],
            )

            if block.shape != expected_shape:
                import warnings
                warnings.warn(
                    f"Blok boyutu uyuşmuyor: beklenen {expected_shape}, "
                    f"alınan {block.shape}. Blok atlanıyor."
                )
                continue

            grid[row_slice, col_slice] = block.astype(np_dtype)

    return grid


# ===========================================================================
# GRID2GRID — ANA FONKSİYON
# MATLAB karşılığı: shared/grid2grid.m
# ===========================================================================

def grid2grid(source, vX, vY, params=None):
    """
    Düzenli bir kaynak ızgarayı yeni koordinatlara yeniden örnekler.

    MATLAB imzası:
        mGrid = grid2grid(sourceGridInfo, vX, vY, sParams)

    Parametreler
    ----------
    source : str veya dict veya np.ndarray
        - str       : GeoTIFF dosya yolu
        - dict      : {'data': np.ndarray, 'spatial_ref': dict} şeklinde ızgara bilgisi
        - np.ndarray: doğrudan veri matrisi (spatial_ref params içinde olmalı)
    vX : np.ndarray
        Hedef X (lon veya sütun) koordinatları — artan sırada.
    vY : np.ndarray
        Hedef Y (lat veya satır) koordinatları — azalan sırada.
    params : dict veya None
        İsteğe bağlı parametreler:
            'block_size'  : int   — blok boyutu (varsayılan 500)
            'null_val'    : float veya list — geçersiz piksel değerleri (varsayılan NaN)
            'tiff_layer'  : int   — GeoTIFF band numarası, 1-tabanlı (varsayılan 1)
            'interp'      : str   — interpolasyon yöntemi: 'linear' veya 'nearest'
            'transform'   : np.ndarray (4,4) — homojen koordinat dönüşüm matrisi

    Döndürür
    -------
    np.ndarray, shape (len(vY), len(vX))
        Yeniden örneklenmiş ızgara.
    """
    if params is None:
        params = {}

    block_size  = params.get('block_size', 500)
    null_val    = params.get('null_val', np.nan)
    tiff_layer  = params.get('tiff_layer', 1)       # 1-tabanlı (MATLAB uyumlu)
    interp_method = params.get('interp', 'linear')
    transform   = params.get('transform', None)

    # Koordinat yönü kontrolü
    if len(vX) > 1 and vX[0] > vX[-1]:
        raise ValueError("vX artan sırada olmalıdır.")
    if len(vY) > 1 and vY[0] < vY[-1]:
        raise ValueError("vY azalan sırada olmalıdır (kuzeyden güneye).")

    grid_shape = (len(vY), len(vX))

    def make_block(block_input):
        return _make_block(
            block_input, source, vX, vY,
            buffer=100, null_val=null_val,
            tiff_layer=tiff_layer, transform=transform,
            interp_method=interp_method,
        )

    return block_process(grid_shape, block_size, 'double', make_block)


# ===========================================================================
# _MAKE_BLOCK — Blok işleme iç fonksiyonu
# MATLAB karşılığı: grid2grid.m > makeBlock()
# ===========================================================================

def _make_block(block_input, source, vX, vY, buffer,
                null_val, tiff_layer, transform, interp_method):
    """
    Tek bir blok için hedef ızgarayı oluşturur.
    """
    idx_x = block_input['index_x']
    idx_y = block_input['index_y']
    ix    = block_input['count_x']
    iy    = block_input['count_y']

    # Blok koordinatları
    blk_x = vX[idx_x[ix]:idx_x[ix + 1]]
    blk_y = vY[idx_y[iy]:idx_y[iy + 1]]
    mX, mY = np.meshgrid(blk_x, blk_y)

    # Koordinat dönüşümü varsa uygula
    if transform is not None:
        if isinstance(transform, np.ndarray) and transform.shape == (4, 4):
            pts = np.vstack([mX.ravel(), mY.ravel(),
                             np.ones(mX.size), np.ones(mX.size)])
            pts_t = np.linalg.solve(transform, pts)
            mX = pts_t[0].reshape(mX.shape)
            mY = pts_t[1].reshape(mY.shape)
        else:
            raise ValueError("transform 4x4 numpy dizisi olmalıdır.")

    # Kaynak verisini oku
    src_data, src_x, src_y = _read_source(
        source, mX, mY, buffer, tiff_layer
    )

    if src_data is None:
        return None

    # Geçersiz pikselleri NaN yap
    src_data = _mask_nulls(src_data, null_val)

    # Hedef koordinatlarda yeniden örnekle
    block = _resample(src_data, src_x, src_y, mX, mY, interp_method)

    return block


# ===========================================================================
# YARDIMCI FONKSİYONLAR
# ===========================================================================

def _read_source(source, mX, mY, buffer, tiff_layer):
    """
    Kaynak veriden ilgili bölgeyi okur.
    GeoTIFF (rasterio) veya dict/ndarray destekler.
    """
    x_min, x_max = mX.min(), mX.max()
    y_min, y_max = mY.min(), mY.max()

    if isinstance(source, (str, Path)):
        with rasterio.open(source) as ds:
            t = ds.transform
            dx = abs(t.a)
            dy = abs(t.e)

            # Piksel indekslerine çevir (buffer ekle)
            col_min = max(0, int((x_min - t.c) / t.a) - buffer)
            col_max = min(ds.width  - 1, int((x_max - t.c) / t.a) + buffer)
            row_min = max(0, int((t.f - y_max) / abs(t.e)) - buffer)
            row_max = min(ds.height - 1, int((t.f - y_min) / abs(t.e)) + buffer)

            if col_max - col_min < 2 or row_max - row_min < 2:
                return None, None, None

            window = rasterio.windows.Window(
                col_min, row_min,
                col_max - col_min + 1,
                row_max - row_min + 1,
            )
            data = ds.read(tiff_layer, window=window).astype(np.float64)

            # Kaynak koordinat vektörleri
            src_x = t.c + (col_min + np.arange(data.shape[1]) + 0.5) * t.a
            src_y = t.f + (row_min + np.arange(data.shape[0]) + 0.5) * t.e

    elif isinstance(source, dict):
        data = source['data'].astype(np.float64)
        sR   = source['spatial_ref']
        src_x, src_y = make_spatial_ref_vecs(sR, 'full')

        # Kesişim bölgesini bul
        ix0 = max(0, np.searchsorted(src_x, x_min) - buffer)
        ix1 = min(len(src_x) - 1, np.searchsorted(src_x, x_max) + buffer)
        iy0 = max(0, np.searchsorted(-src_y, -y_max) - buffer)
        iy1 = min(len(src_y) - 1, np.searchsorted(-src_y, -y_min) + buffer)

        if ix1 - ix0 < 2 or iy1 - iy0 < 2:
            return None, None, None

        data  = data[iy0:iy1+1, ix0:ix1+1]
        src_x = src_x[ix0:ix1+1]
        src_y = src_y[iy0:iy1+1]

    elif isinstance(source, np.ndarray):
        raise ValueError(
            "np.ndarray kaynağı için 'spatial_ref' anahtarını içeren "
            "bir dict kullanın: {'data': arr, 'spatial_ref': sR}"
        )
    else:
        raise ValueError(f"Geçersiz kaynak tipi: {type(source)}")

    return data, src_x, src_y


def _mask_nulls(data, null_val):
    """Geçersiz piksel değerlerini NaN yapar."""
    if null_val is None or (isinstance(null_val, float) and np.isnan(null_val)):
        return data
    if isinstance(null_val, str):
        if null_val == 'dem':
            data[(data < -500) | (data > 9000)] = np.nan
        else:
            raise ValueError(f"Geçersiz null_val string: '{null_val}'")
    else:
        for v in np.atleast_1d(null_val):
            data[data == v] = np.nan
    return data


def _resample(data, src_x, src_y, mX, mY, method):
    """
    RegularGridInterpolator ile yeniden örnekler.
    src_y azalan sıradaysa tersine çevrilir (interpolator artan ister).
    """
    # scipy interpolator artan eksen ister
    if src_y[0] > src_y[-1]:
        data  = data[::-1, :]
        src_y = src_y[::-1]

    interp_func = RegularGridInterpolator(
        (src_y, src_x), data,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )
    pts = np.stack([mY.ravel(), mX.ravel()], axis=-1)
    result = interp_func(pts).reshape(mX.shape)
    return result


# ===========================================================================
# BASIT TEST
# ===========================================================================

if __name__ == '__main__':
    print("grid2grid.py — Temel test")

    # Kaynak ızgara: 10x10 derece, 0.1 derece çözünürlük
    src_x = np.arange(30.0, 40.0, 0.1)
    src_y = np.arange(40.0, 30.0, -0.1)   # azalan (kuzeyden güneye)
    mSX, mSY = np.meshgrid(src_x, src_y)
    src_data = np.sin(mSX / 5) * np.cos(mSY / 5)  # sentetik veri

    source = {
        'data': src_data,
        'spatial_ref': {
            'DeltaLon':  0.1,
            'DeltaLat': -0.1,
            'Lonlim': (30.0, 40.0),
            'Latlim': (30.0, 40.0),
        }
    }

    # Hedef ızgara: daha ince çözünürlük
    vX_out = np.arange(32.0, 38.0, 0.05)
    vY_out = np.arange(38.0, 32.0, -0.05)

    result = grid2grid(source, vX_out, vY_out)

    nan_pct = np.isnan(result).mean() * 100
    print(f"  Kaynak ızgara boyutu : {src_data.shape}")
    print(f"  Hedef ızgara boyutu  : {result.shape}")
    print(f"  NaN oranı            : %{nan_pct:.1f}")
    print(f"  Değer aralığı        : [{np.nanmin(result):.3f}, {np.nanmax(result):.3f}]")
    print("\n  Test BASARILI ✓" if nan_pct < 5 else "\n  Test BASARISIZ ✗")
