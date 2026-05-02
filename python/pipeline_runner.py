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

        result = sti_stitch(
            path      = img_dir,
            file_l    = file_l_path.name,
            file_r    = file_r_path.name,
            corners_l = corners_l,
            corners_r = corners_r,
            **{k: v for k, v in params.items()
               if k not in ("path", "file_l", "file_r", "corners_l", "corners_r")},
        )

        _prog(3, 3, "Stitch tamamlandı")
        _log("Stitch adımı başarıyla tamamlandı.")
        return RunResult(success=True, output={"stitch_result": result})

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
        # importlib ile 2_extract/ altındaki modülleri yükle
        _rs = _load("2_extract", "ext_read_sort")
        _fi = _load("2_extract", "ext_filter_images")
        _sr = _load("2_extract", "ext_stereo_rect")
        _ib = _load("2_extract", "ext_init_bundle")
        _di = _load("2_extract", "ext_disparity")
        _ew = _load("2_extract", "ext_windows")

        ext_read_image    = _rs.ext_read_image
        ext_sort_images   = _rs.ext_sort_images
        ext_filter_images = _fi.ext_filter_images
        ext_stereo_rect   = _sr.ext_stereo_rect
        ext_init_trans    = _ib.ext_init_trans
        ext_bundle_adjust = _ib.ext_bundle_adjust
        ext_disparity     = _di.ext_disparity
        ext_get_roi       = _ew.ext_get_roi

        _log("Extract adımı başlıyor…")

        # 1. ROI seçimi
        _prog(1, TOTAL, "ROI seçiliyor")
        files_l, files_r = ext_get_roi(
            params["hex_files"],
            params["hex_files"],
            params.get("roi_ids", []),
            params.get("target_roi"),
        )

        # 2. Görüntüleri oku ve sırala
        _prog(2, TOTAL, "Görüntüler okunuyor ve sıralanıyor")
        mat_objects = [ext_read_image(f, win_obj=None) for f in files_l + files_r]
        mat_objects = ext_sort_images(files_l + files_r, mat_objects)

        # 3. Görüntü filtreleme
        _prog(3, TOTAL, "Görüntüler filtreleniyor")
        mat_l, mat_r = mat_objects[0], mat_objects[1]
        mat_l_f, mat_r_f = ext_filter_images(
            mat_l, mat_r,
            empty_val=params.get("empty_val", 0),
            opt=params.get("filter_opt"),
        )

        # 4. Stereo rektifikasyon
        _prog(4, TOTAL, "Stereo rektifikasyon")
        rect_result = ext_stereo_rect(
            mat_l_f, mat_r_f,
            save_path  =params.get("save_path"),
            progress_cb=lambda s, t, m: _prog(4, TOTAL, m),
        )

        # 5. İlk dönüşüm tahmini + bundle adjustment
        _prog(5, TOTAL, "Bundle adjustment")
        init_result = ext_init_trans(
            rect_result["mat_l"],
            rect_result["mat_r"],
            progress_cb=lambda s, t, m: _prog(5, TOTAL, m),
        )
        bundle_result = ext_bundle_adjust(
            rect_result["windows_l"],
            rect_result["windows_r"],
            hex_path   =params.get("save_path"),
            progress_cb=lambda s, t, m: _prog(5, TOTAL, m),
        )

        # 6. Disparity haritası
        _prog(6, TOTAL, "Disparity haritası hesaplanıyor")
        disp_result = ext_disparity(
            bundle_result["obj_l"],
            bundle_result["obj_r"],
            resolution=params.get("resolution", "1/2"),
            block_size=params.get("block_size", 7),
        )

        _log("Extract adımı başarıyla tamamlandı.")
        return RunResult(
            success=True,
            output={
                "rect_result":   rect_result,
                "init_result":   init_result,
                "bundle_result": bundle_result,
                "disp_result":   disp_result,
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

    return results
