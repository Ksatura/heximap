"""
integration_test_stitch.py

HEXIMAP Python portuna entegrasyon testi — 1. Aşama: Stitch

Bu script gerçek Hexagon GeoTIFF dosyaları üzerinde stiStitch.py'yi
çalıştırır ve çıktıyı doğrular.

Kullanım:
    python3 integration_test_stitch.py

Çıktı:
    DZB1206-500009L008001.npz  (birleştirilmiş görüntü)
    DZB1206-500009L009001_9.npz
"""

import numpy as np
import rasterio
import sys
import os
from pathlib import Path

# Modül yollarını ekle
BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / 'main' / 'shared'))
sys.path.insert(0, str(BASE / 'main' / '1_stitch'))

from sti_stitch import sti_stitch

# ===========================================================================
# GÖRÜNTÜ ÇİFTLERİ VE KÖŞELERİ
# ===========================================================================

# EarthExplorer'dan alınan köşe koordinatları (piksel → indeks dönüşümü için)
# Hexagon tarama yönü: sol=kuzey, sağ=güney, üst=doğu, alt=batı
# parse_corner_gcps sırası: [NW_lat, NW_lon, NE_lat, NE_lon, SE_lat, SE_lon, SW_lat, SW_lon]

IMAGE_PAIRS = [
    {
        'name':   'L008001',
        'file_a': 'DZB1206-500009L008001_a.tif',
        'file_b': 'DZB1206-500009L008001_b.tif',
        'gcps': [
            39.425, 26.505,   # NW lat, lon
            39.277, 27.862,   # NE lat, lon
            37.180, 27.471,   # SE lat, lon
            37.323, 26.152,   # SW lat, lon
        ]
    },
    {
        'name':   'L009001',
        'file_a': 'DZB1206-500009L009001_9_a.tif',
        'file_b': 'DZB1206-500009L009001_9_b.tif',
        'gcps': [
            38.795, 26.360,   # NW lat, lon
            38.648, 27.704,   # NE lat, lon
            36.552, 27.321,   # SE lat, lon
            36.694, 26.013,   # SW lat, lon
        ]
    },
]


def find_tif_dir():
    """TIF dosyalarının bulunduğu klasörü otomatik olarak bulur."""
    search_dirs = [
        Path.home() / 'Desktop' / 'Heximap_Tests',
        Path.home() / 'Masaüstü' / 'Heximap_Tests',
        Path.home() / 'İndirilenler',
        Path.home() / 'Downloads',
        Path.home() / 'Masaüstü',
        Path.home() / 'Desktop',
        Path.home() / 'heximap_data',
        Path.home(),
    ]
    for d in search_dirs:
        if d.exists():
            tifs = list(d.glob('DZB*.tif'))
            if tifs:
                print(f"  TIF dosyaları bulundu: {d}")
                return d
    return None


def get_image_corners(filepath):
    """
    GeoTIFF dosyasından köşe piksel koordinatlarını hesaplar.
    Hexagon görüntüsünün siyah kenar boşluklarını dikkate alarak
    yaklaşık köşe konumlarını döndürür.

    Döndürür
    -------
    corners : np.ndarray, shape (2, 2)
        [[x_min, y_min], [x_max, y_max]] piksel koordinatları
        (yaklaşık — gerçek köşe tespiti stiGetCorners ile yapılır)
    """
    with rasterio.open(filepath) as ds:
        h, w = ds.height, ds.width

    # Kenar tamponları (Hexagon görüntülerinde ~%5 siyah kenar var)
    margin_x = round(w * 0.03)
    margin_y = round(h * 0.03)

    corners = np.array([
        [margin_x + 1,     margin_y + 1],
        [w - margin_x,  h - margin_y],
    ])
    return corners


def run_stitch_test(pair_info, tif_dir, output_dir):
    """Tek bir görüntü çifti için stitch testini çalıştırır."""
    name    = pair_info['name']
    file_a  = pair_info['file_a']
    file_b  = pair_info['file_b']

    print(f"\n{'='*60}")
    print(f"Görüntü çifti: {name}")
    print(f"{'='*60}")

    # Dosyaların varlığını kontrol et
    path_a = tif_dir / file_a
    path_b = tif_dir / file_b

    if not path_a.exists():
        print(f"  ⚠️  Dosya bulunamadı: {path_a}")
        return False
    if not path_b.exists():
        print(f"  ⚠️  Dosya bulunamadı: {path_b}")
        return False

    # Görüntü bilgilerini göster
    for f, label in [(path_a, 'Sol (a)'), (path_b, 'Sağ (b)')]:
        with rasterio.open(f) as ds:
            print(f"  {label}: {ds.width} x {ds.height} piksel, "
                  f"{ds.dtypes[0]}, CRS={ds.crs}")

    # Köşe koordinatlarını hesapla
    corners_a = get_image_corners(path_a)
    corners_b = get_image_corners(path_b)

    print(f"\n  Sol yarı köşeleri  : {corners_a.tolist()}")
    print(f"  Sağ yarı köşeleri  : {corners_b.tolist()}")
    print(f"\n  ORB eşleştirmesi başlıyor...")

    def progress(step, total, msg):
        print(f"  [{step}/{total}] {msg}")

    try:
        out_path = sti_stitch(
            path=str(tif_dir),
            file_l=file_a,
            file_r=file_b,
            corners_l=corners_a,
            corners_r=corners_b,
            progress_cb=progress,
        )

        # Sonucu doğrula
        data = np.load(out_path)
        img = data['Image']
        T   = data['Transform']

        print(f"\n  ✓ Birleştirme başarılı!")
        print(f"  Çıktı dosyası    : {out_path}")
        print(f"  Görüntü boyutu   : {img.shape[1]} x {img.shape[0]} piksel")
        print(f"  Dönüşüm matrisi  :\n{np.round(T, 4)}")
        print(f"  Sıfır olmayan px : {np.count_nonzero(img)} / {img.size} "
              f"(%{100*np.count_nonzero(img)/img.size:.1f})")

        return True

    except Exception as e:
        import traceback
        print(f"\n  ✗ HATA: {e}")
        traceback.print_exc()
        return False


def main():
    print("HEXIMAP Entegrasyon Testi — Aşama 1: Stitch")
    print("=" * 60)

    # TIF klasörünü bul
    tif_dir = find_tif_dir()
    if tif_dir is None:
        print("\n⚠️  TIF dosyaları bulunamadı.")
        print("Lütfen dosyaların tam yolunu aşağıya girin:")
        tif_dir = Path(input("Klasör yolu: ").strip())
        if not tif_dir.exists():
            print("Klasör bulunamadı, çıkılıyor.")
            sys.exit(1)

    output_dir = tif_dir

    # Testleri çalıştır
    results = []
    for pair in IMAGE_PAIRS:
        ok = run_stitch_test(pair, tif_dir, output_dir)
        results.append((pair['name'], ok))

    # Özet
    print(f"\n{'='*60}")
    print("ÖZET")
    print(f"{'='*60}")
    for name, ok in results:
        status = "✓ BAŞARILI" if ok else "✗ BAŞARISIZ"
        print(f"  {name}: {status}")

    all_ok = all(ok for _, ok in results)
    if all_ok:
        print(f"\nTüm stitch testleri geçti!")
        print(f"Çıktı .npz dosyaları {tif_dir} klasöründe.")
        print(f"\nSıradaki adım: extract (DEM çıkarma) aşaması")
    else:
        print(f"\nBazı testler başarısız. Hata mesajlarını inceleyiniz.")


if __name__ == '__main__':
    main()
