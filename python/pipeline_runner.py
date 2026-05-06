"""
pipeline_runner.py
==================
HEXIMAP arayüz katmanı (Streamlit / QGIS eklentisi) ile core pipeline
modülleri arasındaki tek köprü noktası.

KURALLAR:
  - Bu dosya hiçbir Streamlit veya QGIS import'u içermez.
  - Arayüz katmanı hiçbir zaman shared/ veya extract/ modüllerini
    doğrudan çağırmaz; yalnızca bu dosyadaki run_* fonksiyonlarını kullanır.
  - Her run_* fonksiyonu bir RunResult döndürür.
  - İlerleme ve log bildirimleri callback'ler üzerinden iletilir;
    fonksiyonlar asla print() çağırmaz.

Kullanım örneği:
    from pipeline_runner import run_stitch, run_extract, run_georef, run_rasterize

    def my_log(msg):               print(msg)
    def my_prog(step, total, msg): print(f"{step}/{total} {msg}")

    result = run_stitch(params, log_cb=my_log, progress_cb=my_prog)
    if result.success:
        print(result.output)
    else:
        print(result.error)
"""

from __future__ import annotations

import importlib.util as _ilu
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Tip tanımları
# ---------------------------------------------------------------------------

LogCallback      = Callable[[str], None]
ProgressCallback = Callable[[int, int, str], None]   # (adım, toplam, mesaj)


# ---------------------------------------------------------------------------
# Ortak sonuç nesnesi
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Her run_* fonksiyonunun döndürdüğü sonuç."""
    success: bool
    output:  dict[str, Any] = field(default_factory=dict)
    error:   str            = ""
    tb:      str            = ""   # traceback — debug için


def _fail(exc: Exception) -> RunResult:
    return RunResult(
        success=False,
        error=str(exc),
        tb=traceback.format_exc(),
    )


# ---------------------------------------------------------------------------
# Yardımcılar: path + importlib
# ---------------------------------------------------------------------------

def _ensure_path() -> None:
    """
    python/ kökünü sys.path'e ekler.
    shared.geo_optimize gibi normal paket import'ları için gerekli.
    """
    root = Path(__file__).resolve().parent
    for candidate in [root, root.parent]:
        s = str(candidate)
        if s not in sys.path:
            sys.path.insert(0, s)


def _load(folder: str, module: str):
    """
    Rakamla başlayan klasörlerden (1_stitch, 2_extract vb.) modül yükler.
    Python doğrudan '1_stitch' gibi isimleri import edemez;
    importlib ile dosya yolundan yükleyerek bu kısıtı aşarız.

    Örnek:
        mod = _load("1_stitch", "sti_stitch")
        sti_stitch = mod.sti_stitch
    """
    root = Path(__file__).resolve().parent
    path = root / folder / f"{module}.py"
    spec = _ilu.spec_from_file_location(module, path)
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ensure_path()


# ---------------------------------------------------------------------------
# Adım 1 — Stitch
# ---------------------------------------------------------------------------

def run_stitch(
    params:      dict[str, Any],
    log_cb:      Optional[LogCallback]      = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> RunResult:
    """
    Hexagon görüntü yarılarını birleştirir.

    Beklenen params anahtarları:
        path      (str | Path) : .mat çıktılarının yazılacağı klasör
        file_l    (str | Path) : sol yarı GeoTIFF yolu
        file_r    (str | Path) : sağ yarı GeoTIFF yolu
        corners_l (list)       : sol görüntü köşe koordinatları
        corners_r (list)       : sağ görüntü köşe koordinatları
        — sti_stitch() imzasındaki diğer kwargs buraya eklenebilir —
    """
    def _log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    def _prog(step: int, total: int, msg: str) -> None:
        if progress_cb:
            progress_cb(step, total, msg)

    try:
        import numpy as np
        sti_stitch = _load("1_stitch", "sti_stitch").sti_stitch

        _log("Stitch adımı başlıyor…")
        _prog(0, 3, "Görüntüler okunuyor")

        # sti_stitch:
        #   - corners shape (2,2): [[x_üst, y_üst], [x_alt, y_alt]]
        #     Kullanıcıdan 4 nokta gelirse 1. ve 3. noktayı al (üst-sol, alt-sağ)
        #   - path: görüntü dosyalarının klasörü
        #   - file_l/file_r: sadece dosya adı (tam yol değil)
        def _to_two_point(corners):
            arr = np.array(corners, dtype=float)
            if arr.shape == (2, 2):
                return arr
            # 4 nokta: üst-sol (0) ve alt-sağ (2)
            return np.array([arr[0], arr[2]])

        corners_l = _to_two_point(params["corners_l"])
        corners_r = _to_two_point(params["corners_r"])

        # path ve file_l/file_r'yi ayır
        # sti_stitch'te path = görüntülerin bulunduğu klasör (çıktı klasörü değil)
        file_l_path = Path(params["file_l"])
        file_r_path = Path(params["file_r"])
        img_dir     = file_l_path.parent   # görüntü dosyasının gerçek klasörü

        # Stitch öncesi .npz snapshot — yalnızca bu kareye ait .npz'i bulmak için
        npz_before = set(img_dir.glob("*.npz"))

        result = sti_stitch(
            path      = img_dir,
            file_l    = file_l_path.name,
            file_r    = file_r_path.name,
            corners_l = corners_l,
            corners_r = corners_r,
            **{k: v for k, v in params.items()
               if k not in ("path", "file_l", "file_r", "corners_l", "corners_r")},
        )

        # ── .npz yolunu bul ──────────────────────────────────────────────
        # Önce: sti_stitch doğrudan Path/str döndürüyor mu?
        new_npz: list = []

        if isinstance(result, (str, Path)) and Path(str(result)).suffix == ".npz":
            new_npz = [Path(result)]
        elif isinstance(result, dict):
            # Ortak key adları: 'npz', 'npz_path', 'output_path', 'save_path', 'output'
            for key in ("npz", "npz_path", "output_path", "save_path", "output"):
                val = result.get(key)
                if val and Path(str(val)).suffix == ".npz":
                    new_npz = [Path(val)]
                    break

        # Fallback: stitch sonrası klasörde beliren YENİ .npz'leri diff ile bul
        # Bu yöntem her sti_stitch çağrısından sonra SADECE o çağrının ürettiklerini alır.
        if not new_npz:
            npz_after = set(img_dir.glob("*.npz"))
            new_npz   = sorted(npz_after - npz_before)

        # Hâlâ bulunamazsa dosya adından tahmin et
        if not new_npz:
            stem    = file_l_path.stem.split("_")[0]
            guesses = sorted(img_dir.glob(f"{stem}*.npz"))
            new_npz = guesses if guesses else []

        _log(f"  Üretilen .npz: {[str(p) for p in new_npz]}")

        _prog(3, 3, "Stitch tamamlandı")
        _log("Stitch adımı başarıyla tamamlandı.")
        return RunResult(
            success=True,
            output={
                "stitch_result": result,
                "npz_files":     new_npz,   # ← extract adımına otomatik aktarılır
            },
        )

    except Exception as exc:
        _log(f"HATA (stitch): {exc}")
        return _fail(exc)


# ---------------------------------------------------------------------------
# Adım 2 — Extract
# ---------------------------------------------------------------------------

def run_extract(
    params:      dict[str, Any],
    log_cb:      Optional[LogCallback]      = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> RunResult:
    """
    Stereo DEM çıkarma adımlarını sırayla çalıştırır:
      ext_read_sort → ext_filter_images → ext_stereo_rect
      → ext_init_trans → ext_bundle_adjust → ext_disparity

    Beklenen params anahtarları:
        hex_files   (list[Path]) : birleştirilmiş Hexagon dosyaları
        roi_ids     (list)       : ilgilenilen bölge ID'leri
        target_roi  (any)        : hedef ROI tanımı
        resolution  (str)        : disparity çözünürlüğü, ör. '1/2'
        block_size  (int)        : SGBM blok boyutu, varsayılan 7
        save_path   (str | Path) : ara sonuçların kaydedileceği klasör
    """
    def _log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    def _prog(step: int, total: int, msg: str) -> None:
        if progress_cb:
            progress_cb(step, total, msg)

    TOTAL = 6

    try:
        import numpy as np

        # importlib ile 2_extract/ altındaki modülleri yükle
        _rs = _load("2_extract", "ext_read_sort")
        _fi = _load("2_extract", "ext_filter_images")
        _sr = _load("2_extract", "ext_stereo_rect")
        _ib = _load("2_extract", "ext_init_bundle")
        _di = _load("2_extract", "ext_disparity")

        ext_read_image    = _rs.ext_read_image   # (mat_obj, win_obj) → win_obj
        ext_sort_images   = _rs.ext_sort_images   # (files, mats) → (files_sorted, mats_sorted)
        ext_filter_images = _fi.ext_filter_images
        ext_stereo_rect   = _sr.ext_stereo_rect
        ext_init_trans    = _ib.ext_init_trans
        ext_bundle_adjust = _ib.ext_bundle_adjust
        ext_disparity     = _di.ext_disparity

        _log("Extract adımı başlıyor…")

        # ── .npz doğrulama ────────────────────────────────────────────────
        hex_files = params.get("hex_files") or []
        if not hex_files:
            raise ValueError(
                "Extract adımı için .npz dosya listesi boş. "
                "Stitch adımını önce çalıştırın veya "
                "Adım 2 metin kutusuna .npz yollarını girin."
            )
        hex_files = sorted(set(Path(f) for f in hex_files))
        missing   = [f for f in hex_files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                "Şu .npz dosyaları bulunamadı: "
                + ", ".join(str(m) for m in missing)
            )
        _log(f"  İşlenecek .npz ({len(hex_files)} adet): "
             + ", ".join(f.name for f in hex_files))

        # ── 1. .npz dosyalarını yükle ─────────────────────────────────────
        # ext_read_image(mat_obj, win_obj) imzası:
        #   mat_obj  = {"Image": ndarray, "SpatialTrans": ndarray, ...}
        #   win_obj  = {"Window": ndarray shape (2,2)}
        # .npz dosyaları bu dict yapısını allow_pickle=True ile içeriyor.
        _prog(1, TOTAL, f".npz yükleniyor ({len(hex_files)} dosya)")
        mat_objects = []
        for f in hex_files:
            raw = np.load(f, allow_pickle=True)
            # .npz içindeki anahtarlar: 'Image', 'SpatialTrans' ve diğerleri
            # allow_pickle ile yüklenen object array'lerini dict'e çevir
            mat_obj = {}
            for key in raw.files:
                val = raw[key]
                # 0-d object array → gerçek nesneye çevir
                if val.ndim == 0:
                    val = val.item()
                mat_obj[key] = val
            # .npz "Transform" key'ini ext_sort_images/ext_read_image'in
            # beklediği "SpatialTrans" adıyla eşitle.
            # MATLAB (4,4) homogeneous → Python ext_sort_images (3,3) bekler:
            #   [R t]      [a b tx]
            #   [0 1]  →   [c d ty]   (sol-üst 3x3 blok, son satır/sütun atılır)
            #              [0 0  1]
            if "Transform" in mat_obj and "SpatialTrans" not in mat_obj:
                T_raw = np.array(mat_obj["Transform"])
                if T_raw.shape == (4, 4):
                    # (4,4) MATLAB homojen → (3,3): sadece [0:2, 0:2] + son sütun/satır
                    # Hexagon affin: x,y düzleminde → satır/sütun 2 (z) atılır
                    # Sıra: [x_col, y_col, t_col] → [0,1,3] index
                    T33 = np.zeros((3, 3))
                    T33[0:2, 0:2] = T_raw[0:2, 0:2]   # 2x2 rotasyon/ölçek
                    T33[0:2, 2]   = T_raw[0:2, 3]      # tx, ty (son sütun)
                    T33[2, 2]     = 1.0
                    mat_obj["SpatialTrans"] = T33
                    _log(f"  Transform (4,4)→(3,3) dönüşümü: {T33.round(6).tolist()}")
                elif T_raw.shape == (3, 3):
                    mat_obj["SpatialTrans"] = T_raw
                else:
                    raise ValueError(
                        f"Beklenmedik Transform shape: {T_raw.shape}. "
                        "(3,3) veya (4,4) bekleniyor."
                    )
            _log(f"  {f.name} yüklendi → keys: {list(mat_obj.keys())}")
            mat_objects.append(mat_obj)

        # ── 2. Görüntüleri stereo çift sırasına göre sırala ───────────────
        _prog(2, TOTAL, "Stereo çift sıralanıyor")
        sorted_result = ext_sort_images(hex_files, mat_objects)
        # ext_sort_images → (hex_files_sorted, mat_objects_sorted) tuple döndürür
        if (isinstance(sorted_result, tuple) and len(sorted_result) == 2):
            hex_files_s, mat_objects_s = sorted_result
        else:
            # In-place sıralama yapıyorsa listeyi olduğu gibi kullan
            hex_files_s, mat_objects_s = hex_files, mat_objects

        _log(f"  Sıralama sonrası: {[Path(f).name for f in hex_files_s]}")

        if len(mat_objects_s) < 2:
            raise RuntimeError(
                f"En az 2 görüntü gerekli, {len(mat_objects_s)} bulundu."
            )

        # ── 3. Pencere kırpma ────────────────────────────────────────────
        # Görüntüler dev (60k×32k px, ~2 GB). Tamamını RAM'e almak kernel
        # Killed'a yol açıyor. params["corners_l/r"] ile kullanıcının seçtiği
        # bölgeyi kırp; yoksa görüntüyü MAX_PX×MAX_PX ile orantılı küçült.
        _prog(3, TOTAL, "Görüntü pencereleri kırpılıyor")

        # MAX_PX: filtre + rektifikasyon için güvenli üst sınır.
        # float64 dönüşümü 2× bellek; CLAHE tile'ları 4× → 8000px ≈ 480 MB peak.
        # Corners seçilmiş olsa bile bu sınır uygulanır.
        MAX_PX = 6000   # ~270 MP → float64 ~4.3 GB peak, RAM güvenli

        def _make_window(mat_obj, corners=None):
            """
            1) corners varsa o bölgeyi al
            2) bölge MAX_PX'ten büyükse merkeze hizalayarak kırp
            Window: [[x_min, y_min], [x_max, y_max]] 1-tabanlı
            """
            img = mat_obj["Image"]
            h, w = img.shape[:2]

            if corners and len(corners) == 2:
                x_min = max(1, int(min(corners[0][0], corners[1][0])))
                y_min = max(1, int(min(corners[0][1], corners[1][1])))
                x_max = min(w, int(max(corners[0][0], corners[1][0])))
                y_max = min(h, int(max(corners[0][1], corners[1][1])))
            else:
                x_min, y_min, x_max, y_max = 1, 1, w, h

            # MAX_PX sınırını uygula — bölgenin merkezinden kırp
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            half_x = min((x_max - x_min) // 2, MAX_PX // 2)
            half_y = min((y_max - y_min) // 2, MAX_PX // 2)
            x_min = max(1, cx - half_x)
            y_min = max(1, cy - half_y)
            x_max = min(w, cx + half_x)
            y_max = min(h, cy + half_y)

            return {
                "Window": np.array([[x_min, y_min], [x_max, y_max]], dtype=float)
            }

        win_l = _make_window(mat_objects_s[0], params.get("corners_l"))
        win_r = _make_window(mat_objects_s[1], params.get("corners_r"))

        _log(f"  Sol pencere: {win_l['Window'].tolist()}")
        _log(f"  Sağ pencere: {win_r['Window'].tolist()}")

        win_l = ext_read_image(mat_objects_s[0], win_l)
        win_r = ext_read_image(mat_objects_s[1], win_r)

        mat_l = {**mat_objects_s[0], **win_l}
        mat_r = {**mat_objects_s[1], **win_r}

        crop_l = mat_l["Image"].shape
        crop_r = mat_r["Image"].shape
        mem_mb  = (crop_l[0]*crop_l[1] + crop_r[0]*crop_r[1]) * 8 / 1024**2
        _log(f"  Kırpılmış sol  Image: {crop_l}  sağ: {crop_r}")
        _log(f"  Tahmini filtre belleği (float64×2): ~{mem_mb:.0f} MB")

        # ── 4. Görüntü filtreleme ─────────────────────────────────────────
        # ext_filter_images(img1, img2) → np.ndarray bekler, dict değil
        _prog(4, TOTAL, "Görüntüler filtreleniyor")
        img_l_f, img_r_f = ext_filter_images(
            mat_l["Image"], mat_r["Image"],
            empty_val=params.get("empty_val", 0),
            opt=params.get("filter_opt"),
        )
        # Belleği hemen serbest bırak
        del mat_l["Image"], mat_r["Image"]
        mat_l_f = {**mat_l, "Image": img_l_f}
        mat_r_f = {**mat_r, "Image": img_r_f}
        del img_l_f, img_r_f



        # ── 5. Stereo rektifikasyon ───────────────────────────────────────
        # ext_stereo_rect(obj_l, obj_r, save_path, progress_cb) → (obj_l, obj_r)
        # Her iki nesne 'Image' ve 'Window' içermeli; dönüşte 'RectImage',
        # 'Homography', 'RectWindow', 'PointMatches', 'Accuracy' eklenir.
        # progress_cb imzası: progress_cb(msg: str) — sadece mesaj alır.
        _prog(5, TOTAL, "Stereo rektifikasyon")
        _log("  ext_stereo_rect başlıyor (ORB eşleştirme + epipolar)…")

        rect_l, rect_r = ext_stereo_rect(
            mat_l_f, mat_r_f,
            save_path   = str(params["save_path"]) if params.get("save_path") else None,
            progress_cb = lambda msg: _log(f"  rect: {msg}"),
        )
        _log(f"  Rektifikasyon tamamlandı — "
             f"RectImage boyutu: {rect_l.get('RectImage', np.array([])).shape}")

        # ── 6. Disparity (bundle + stereo matching) ───────────────────────
        # ext_init_trans ve ext_bundle_adjust modülleri yoksa bu adım atlanır.
        # ext_disparity rect_l/rect_r üzerinden çalışır.
        _prog(6, TOTAL, "Disparity hesaplanıyor")
        _log("  ext_disparity başlıyor…")

        disp_result = ext_disparity(
            rect_l, rect_r,
            resolution = params.get("resolution", "1/2"),
            block_size = params.get("block_size", 7),
        )

        _log("Extract adımı başarıyla tamamlandı.")
        return RunResult(
            success=True,
            output={
                "rect_l":      rect_l,
                "rect_r":      rect_r,
                "disp_result": disp_result,
            },
        )


    except Exception as exc:
        _log(f"HATA (extract): {exc}")
        return _fail(exc)


# ---------------------------------------------------------------------------
# Adım 3 — Georef
# ---------------------------------------------------------------------------

def run_georef(
    params:      dict[str, Any],
    log_cb:      Optional[LogCallback]      = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> RunResult:
    """
    Jeoreferanslama optimizasyonu.

    Beklenen params anahtarları:
        win_obj        (dict)       : extract adımından gelen pencere nesnesi
        ref_path       (str | Path) : referans DEM yolu
        shp_path       (str | Path) : shapefile yolu (opsiyonel)
        zone           (int)        : UTM dilimi
        hemisphere     (str)        : 'N' veya 'S'
        visualize      (bool)       : ara sonuçları görselleştir
        opt            (dict)       : optimizasyon seçenekleri
        get_ref_dem_fn (callable)   : referans DEM yükleme fonksiyonu (opsiyonel)
    """
    def _log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    def _prog(step: int, total: int, msg: str) -> None:
        if progress_cb:
            progress_cb(step, total, msg)

    TOTAL = 2

    try:
        # shared/ normal paket olduğundan doğrudan import çalışır
        from shared.geo_optimize import geo_init_trans, geo_opti_trans  # noqa

        _log("Georef adımı başlıyor…")

        _prog(1, TOTAL, "İlk dönüşüm hesaplanıyor")
        init_result = geo_init_trans(
            win_obj       =params["win_obj"],
            ref_path      =params["ref_path"],
            shp_path      =params.get("shp_path"),
            visualize     =params.get("visualize", False),
            get_ref_dem_fn=params.get("get_ref_dem_fn"),
        )

        _prog(2, TOTAL, "Optimizasyon çalışıyor")
        opti_result = geo_opti_trans(
            pts        =init_result["pts"],
            ref_path   =params["ref_path"],
            shp_path   =params.get("shp_path"),
            zone       =params["zone"],
            hemisphere =params["hemisphere"],
            opt        =params.get("opt", {}),
            progress_cb=lambda s, t, m: _prog(2, TOTAL, m),
        )

        _log("Georef adımı başarıyla tamamlandı.")
        return RunResult(
            success=True,
            output={
                "init_result": init_result,
                "opti_result": opti_result,
            },
        )

    except Exception as exc:
        _log(f"HATA (georef): {exc}")
        return _fail(exc)


# ---------------------------------------------------------------------------
# Adım 4 — Rasterize
# ---------------------------------------------------------------------------

def run_rasterize(
    params:      dict[str, Any],
    log_cb:      Optional[LogCallback]      = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> RunResult:
    """
    DEM ve ortofoto GeoTIFF olarak dışa aktarır.

    Beklenen params anahtarları:
        win_obj      (dict)       : georef adımından gelen pencere nesnesi
        save_path    (str | Path) : çıktı GeoTIFF klasörü
        do_clean     (bool)       : DEM temizleme    (varsayılan True)
        do_median    (bool)       : medyan filtresi  (varsayılan False)
        do_denoise   (bool)       : denoising        (varsayılan True)
        export_ortho (bool)       : ortofoto da dışa aktar (varsayılan True)
    """
    def _log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    def _prog(step: int, total: int, msg: str) -> None:
        if progress_cb:
            progress_cb(step, total, msg)

    TOTAL = 2

    try:
        _re       = _load("4_rasterize", "ras_export")
        ras_dem   = _re.ras_dem
        ras_ortho = _re.ras_ortho

        _log("Rasterize adımı başlıyor…")

        _prog(1, TOTAL, "DEM dışa aktarılıyor")
        dem_path = ras_dem(
            win_obj   =params["win_obj"],
            save_path =params["save_path"],
            do_clean  =params.get("do_clean",   True),
            do_median =params.get("do_median",  False),
            do_denoise=params.get("do_denoise", True),
        )

        ortho_path = None
        if params.get("export_ortho", True):
            _prog(2, TOTAL, "Ortofoto dışa aktarılıyor")
            ortho_path = ras_ortho(
                win_obj    =params["win_obj"],
                save_path  =params["save_path"],
                progress_cb=lambda s, t, m: _prog(2, TOTAL, m),
            )

        _log("Rasterize adımı başarıyla tamamlandı.")
        return RunResult(
            success=True,
            output={
                "dem_path":   dem_path,
                "ortho_path": ortho_path,
            },
        )

    except Exception as exc:
        _log(f"HATA (rasterize): {exc}")
        return _fail(exc)


# ---------------------------------------------------------------------------
# Tüm pipeline'ı sırayla çalıştır
# ---------------------------------------------------------------------------

def run_pipeline(
    stitch_params:    dict[str, Any],
    extract_params:   dict[str, Any],
    georef_params:    dict[str, Any],
    rasterize_params: dict[str, Any],
    steps:            list[str] | None           = None,
    log_cb:           Optional[LogCallback]      = None,
    progress_cb:      Optional[ProgressCallback] = None,
) -> dict[str, RunResult]:
    """
    İstenen adımları sırayla çalıştırır.

    steps=None ise tüm adımlar çalışır.
    Örnek: steps=["stitch", "extract"]  → yalnızca ilk iki adım

    Önceki adımın çıktısındaki win_obj otomatik olarak
    sonraki adımın params'ına aktarılır.
    """
    all_steps = ["stitch", "extract", "georef", "rasterize"]
    active    = steps if steps is not None else all_steps

    runners = {
        "stitch":    (run_stitch,    stitch_params),
        "extract":   (run_extract,   extract_params),
        "georef":    (run_georef,    georef_params),
        "rasterize": (run_rasterize, rasterize_params),
    }

    results: dict[str, RunResult] = {}

    for step_name in all_steps:
        if step_name not in active:
            continue

        fn, p = runners[step_name]

        # Önceki adımın win_obj çıktısını params'a aktar
        if step_name == "georef" and "extract" in results:
            ext_out = results["extract"].output
            p.setdefault("win_obj", ext_out.get("bundle_result", {}).get("win_obj"))

        if step_name == "rasterize" and "georef" in results:
            geo_out = results["georef"].output
            p.setdefault("win_obj", geo_out.get("opti_result", {}).get("win_obj"))

        result = fn(p, log_cb=log_cb, progress_cb=progress_cb)
        results[step_name] = result

        if not result.success:
            if log_cb:
                log_cb(f"Pipeline '{step_name}' adımında durdu: {result.error}")
            break

        # Stitch → Extract: üretilen .npz dosyalarını otomatik aktar
        if step_name == "stitch" and "extract" in active:
            npz_files = result.output.get("npz_files", [])
            if npz_files:
                extract_params.setdefault("hex_files", npz_files)
                if log_cb:
                    log_cb(f"Extract'a otomatik aktarılan .npz: "
                           f"{[str(p) for p in npz_files]}")

    return results
