"""
app.py — HEXIMAP Streamlit Arayüzü  v6
========================================
Çalıştırma:
    cd ~/heximap/python
    streamlit run app.py

Bağımlılıklar:
    pip install streamlit streamlit-drawable-canvas Pillow rasterio matplotlib

DEĞİŞİKLİKLER (v5 → v6):
  - HATA DÜZELTMESİ: Zenity thread içinde st.session_state'e erişim
    ScriptRunContext olmadığı için KeyError veriyordu. zenity_running ve
    zenity_results artık modül-düzeyinde sıradan Python dict'leri;
    session_state ile her rerun'da senkronize ediliyor.
  - HATA DÜZELTMESİ: RANSAC başarısız olduğunda pipeline sessizce asılı
    kalıyordu. run_stitch sonucu kontrol edilip açık hata loglanıyor.
  - Logo desteği: python/logo/logo.png varsa header'da görüntülenir.
  - Canvas koordinat düzeltmesi (v5'ten): st_canvas "point" modu "path"
    tipi SVG nesneleri döndürür; left/top/radius yerine path komutları
    ayrıştırılıyor. display_scale ve tiff_scale ayrı adımlarda uygulanıyor.
  - Zenity polling (v5'ten): _browse_button'da st.rerun() yok; sayfa
    sonundaki 1 s'lik polling bloğu sonucu yakalar.
  - canvas_ver_l/r: "Önizleme Yükle" her basışta canvas key'ini sıfırlar,
    stale drawing sorunu ortadan kalkar.
"""

from __future__ import annotations

import warnings
try:
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except ImportError:
    pass

import base64
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════
# Thread-safe Zenity state
# ───────────────────────────────────────────────────────────────────────────
# st.session_state thread'den erişilemez (ScriptRunContext yok → KeyError).
# Bu iki dict modül düzeyinde yaşar; tüm Zenity thread'leri bunlara yazar.
# Her Streamlit rerun başında _sync_zenity_to_session() ile session_state'e
# kopyalanır; widget'lar session_state'ten okur.
# ═══════════════════════════════════════════════════════════════════════════
_ZENITY_RUNNING: dict[str, bool] = {}  # session_key → True/False
_ZENITY_RESULTS: dict[str, str]  = {}  # session_key → seçilen yol


# ── Sayfa yapılandırması ───────────────────────────────────────────────────
st.set_page_config(
    page_title="HEXIMAP",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logo ───────────────────────────────────────────────────────────────────
_LOGO_PATH = Path(__file__).resolve().parent / "logo" / "logo.png"

def _logo_html() -> str:
    """Logo varsa base64 <img>, yoksa metin başlık döndürür."""
    if _LOGO_PATH.exists():
        data = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
        return (
            f'<img src="data:image/png;base64,{data}" '
            f'style="height:52px;width:auto;object-fit:contain;" alt="HEXIMAP">'
        )
    return '<h1 class="hx-title">⬡ HEXIMAP</h1>'


# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.hx-header {
    background: linear-gradient(135deg,#0f1923 0%,#1a2e40 100%);
    border-bottom: 2px solid #2dd4bf;
    padding: 1.4rem 2rem; margin: -1rem -1rem 2rem -1rem;
    display:flex; align-items:center; gap:1.2rem;
}
.hx-title {
    font-family:'IBM Plex Mono',monospace; font-size:1.8rem;
    font-weight:600; color:#2dd4bf; letter-spacing:0.08em; margin:0;
}
.hx-subtitle { font-size:0.85rem; color:#94a3b8; margin:0; }
.hx-step {
    border:1px solid #1e3448; border-left:4px solid #2dd4bf;
    border-radius:6px; padding:1.2rem 1.4rem;
    margin-bottom:1.6rem; background:#0d1f2d;
}
.hx-step-title {
    font-family:'IBM Plex Mono',monospace; font-size:0.95rem;
    font-weight:600; color:#2dd4bf; margin-bottom:0.8rem;
    text-transform:uppercase; letter-spacing:0.06em;
}
.hx-step-done  { border-left-color:#22c55e !important; }
.hx-step-error { border-left-color:#ef4444 !important; }
.hx-step-run   { border-left-color:#f59e0b !important; }
.hx-pair-box {
    border:1px solid #1e3448; border-radius:6px;
    padding:1rem; margin-bottom:1rem; background:#0a1a28;
}
.hx-pair-title {
    font-family:'IBM Plex Mono',monospace; font-size:0.82rem;
    color:#64748b; margin-bottom:0.6rem;
}
.hx-log {
    font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
    background:#060e14; color:#94a3b8; border:1px solid #1e3448;
    border-radius:4px; padding:0.8rem 1rem; max-height:220px;
    overflow-y:auto; white-space:pre-wrap;
}
section[data-testid="stSidebar"] { background:#0a1520; border-right:1px solid #1e3448; }
section[data-testid="stSidebar"] label { font-size:0.82rem !important; color:#94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hx-header">
    {_logo_html()}
    <div>
        <p class="hx-subtitle">Hexagon KH-9 Imagery · DEM Extraction Pipeline</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Session state başlatma ─────────────────────────────────────────────────
_PAIR_TEMPLATE: dict[str, Any] = {
    "file_l": "", "file_r": "",
    "corners_l": [],        # [[x_üst,y_üst],[x_alt,y_alt]] — orijinal piksel
    "corners_r": [],
    "preview_l": None, "scale_l": 1.0,
    "preview_r": None, "scale_r": 1.0,
    # canvas_ver: "Önizleme Yükle" her basışta artar → canvas widget sıfırlanır
    "canvas_ver_l": 0,
    "canvas_ver_r": 0,
}

_defaults: dict[str, Any] = {
    "pairs":            [dict(_PAIR_TEMPLATE)],
    "log_lines":        [],
    "step_states":      {s: "idle" for s in ["stitch", "extract", "georef", "rasterize"]},
    "pipeline_running": False,
    "last_results":     {},
    # Zenity için session_state mirror — asıl kaynak _ZENITY_* dict'leri
    "zenity_results":   {},
    "zenity_running":   {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _sync_zenity_to_session() -> None:
    """
    Modül-düzeyi _ZENITY_* dict'lerini session_state'e kopyalar.
    Her rerun başında çağrılır.
    Thread'ler _ZENITY_* dict'lerine yazar → bu fonksiyon ana thread'e taşır.
    """
    # Yeni sonuçları session'a aktar ve modül dict'ten sil
    for k, v in list(_ZENITY_RESULTS.items()):
        st.session_state["zenity_results"][k] = v
        del _ZENITY_RESULTS[k]
    # Çalışma durumunu yansıt
    for k, v in _ZENITY_RUNNING.items():
        st.session_state["zenity_running"][k] = v

_sync_zenity_to_session()


# ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────
_STATE_ICON = {"idle": "○", "running": "◌", "done": "✓", "error": "✗"}
_STATE_CSS  = {
    "idle":    "",
    "running": "hx-step-run",
    "done":    "hx-step-done",
    "error":   "hx-step-error",
}

def _step_header(key: str, label: str) -> None:
    state = st.session_state["step_states"][key]
    st.markdown(
        f'<div class="hx-step-title {_STATE_CSS[state]}">'
        f'{_STATE_ICON[state]} {label}</div>',
        unsafe_allow_html=True,
    )

def _append_log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    st.session_state["log_lines"].append(f"[{ts}] {msg}")

def _log_widget(placeholder=None) -> None:
    html = (
        '<div class="hx-log">'
        + "\n".join(st.session_state["log_lines"][-80:])
        + "</div>"
    )
    if placeholder:
        placeholder.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


# ── Zenity dosya seçici ────────────────────────────────────────────────────
def _zenity_browse(session_key: str, start_dir: str = "",
                   folder_only: bool = False) -> None:
    """
    Zenity'i ayrı bir thread'de başlatır.

    DÜZELTİLDİ (v6): Thread içinde st.session_state KULLANILMIYOR.
    _ZENITY_RUNNING ve _ZENITY_RESULTS modül-düzeyi dict'lerine yazılır.
    Bir sonraki rerun'da _sync_zenity_to_session() bunları session_state'e taşır.
    """
    if _ZENITY_RUNNING.get(session_key):
        return  # zaten çalışıyor

    _ZENITY_RUNNING[session_key] = True

    def _run():
        env = os.environ.copy()
        env.setdefault("DISPLAY", ":0")
        try:
            if folder_only:
                cmd = ["zenity", "--file-selection", "--directory",
                       "--title=Klasör Seç"]
            else:
                cmd = [
                    "zenity", "--file-selection",
                    "--title=Dosya Seç",
                    "--file-filter=GeoTIFF | *.tif *.tiff *.TIF *.TIFF",
                    "--file-filter=Shapefile | *.shp",
                    "--file-filter=Tüm dosyalar | *",
                ]
            if start_dir and Path(start_dir).exists():
                cmd += [f"--filename={start_dir}/"]
            cmd += ["--modal"]
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=300, env=env)
            path = r.stdout.strip()
            if path:
                _ZENITY_RESULTS[session_key] = path   # ← thread-safe
        except Exception:
            pass
        finally:
            _ZENITY_RUNNING[session_key] = False       # ← thread-safe

    threading.Thread(target=_run, daemon=True).start()


def _browse_button(session_key: str, btn_key: str,
                   start_dir: str = "", folder_only: bool = False) -> None:
    """
    Gözat butonu — Zenity thread başlatır, st.rerun() ÇAĞIRMAZ.
    Polling bloğu (sayfa sonu) sonucu yakalar ve rerun'u tetikler.
    """
    is_running = _ZENITY_RUNNING.get(session_key, False)
    label = "⏳" if is_running else "📂"
    if st.button(label, key=btn_key, help="Dosya seç", disabled=is_running):
        _zenity_browse(session_key, start_dir=start_dir, folder_only=folder_only)


# ── GeoTIFF önizleme ───────────────────────────────────────────────────────
def _geotiff_preview(tiff_path: str, max_px: int = 800):
    """
    GeoTIFF'ten hafıza dostu PIL önizleme.
    rasterio out_shape ile direkt küçük boyutta okur (OOM yok).
    Döndürür: (PIL.Image, tiff_scale)
        tiff_scale = önizleme_piksel / orijinal_piksel
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
        import numpy as np
        from PIL import Image

        with rasterio.open(tiff_path) as src:
            orig_w, orig_h = src.width, src.height
            scale  = min(max_px / orig_w, max_px / orig_h, 1.0)
            out_w  = max(1, int(orig_w * scale))
            out_h  = max(1, int(orig_h * scale))
            if src.count >= 3:
                data = src.read([1, 2, 3],
                                out_shape=(3, out_h, out_w),
                                resampling=Resampling.average)
                data = np.moveaxis(data, 0, -1)
            else:
                data = src.read(1, out_shape=(out_h, out_w),
                                resampling=Resampling.average)
                data = np.stack([data] * 3, axis=-1)

        data = np.nan_to_num(data.astype(float), nan=0.0)
        lo, hi = data.min(), data.max()
        if hi > lo:
            data = ((data - lo) / (hi - lo) * 255).astype("uint8")
        else:
            data = np.zeros_like(data, dtype="uint8")
        return Image.fromarray(data, "RGB"), scale

    except Exception as exc:
        st.error(f"Önizleme hatası: {exc}")
        return None, 1.0


# ── Canvas yardımcısı ──────────────────────────────────────────────────────
def _parse_canvas_points(objects: list, display_scale: float) -> list:
    """
    st_canvas "point" modundan gelen nesneleri önizleme piksel uzayına çevirir.

    st_canvas point modu → tip: "path", veri: [["M", x, y], ["L", x, y], ...]
    Her "path" nesnesi bir noktaya karşılık gelir; ilk "M" komutu koordinatı verir.
    Eski "circle" tipi de fallback olarak desteklenir.
    """
    pts = []
    for obj in objects:
        obj_type = obj.get("type", "")

        if obj_type == "path":
            for cmd in obj.get("path", []):
                if isinstance(cmd, list) and len(cmd) >= 3 and cmd[0] in ("M", "L"):
                    try:
                        pts.append([float(cmd[1]) / display_scale,
                                    float(cmd[2]) / display_scale])
                        break  # her path nesnesi = 1 nokta
                    except (ValueError, TypeError):
                        continue

        elif obj_type in ("circle", "point"):
            try:
                r  = float(obj.get("radius", 0))
                cx = float(obj.get("left", 0)) + r
                cy = float(obj.get("top",  0)) + r
                pts.append([cx / display_scale, cy / display_scale])
            except (ValueError, TypeError):
                continue

    return pts  # önizleme piksel uzayında


def _canvas_corner_picker(img, tiff_scale: float, corners: list,
                           key: str) -> list:
    """
    streamlit-drawable-canvas ile köşe seçimi.

    Koordinat zinciri:
        canvas koordinatı
            ÷ display_scale  →  önizleme pikseli
            ÷ tiff_scale     →  orijinal GeoTIFF pikseli

    Döndürür: güncellenmiş corners listesi [[x_orig, y_orig], ...]
    """
    try:
        try:
            from streamlit_drawable_canvas import st_canvas
        except ImportError:
            from streamlit_drawable_canvas_fix import st_canvas
        from PIL import Image as PILImage

        iw, ih = img.size                        # önizleme boyutu
        canvas_w      = min(iw, 720)
        display_scale = canvas_w / iw           # canvas → önizleme
        canvas_h_adj  = int(ih * display_scale)
        display_img   = img.resize((canvas_w, canvas_h_adj), PILImage.LANCZOS)

        n = len(corners)
        if n == 0:
            hint = "1. tıklama = üst-sol köşe"
        elif n == 1:
            hint = "2. tıklama = alt-sağ köşe"
        else:
            hint = "✓ 2 köşe seçildi — Sıfırla ile yeniden seç"

        st.caption(f"**Nokta aracı aktif** — görüntüye tıklayın.  {hint}")

        result = st_canvas(
            fill_color           = "#2dd4bf",
            stroke_width         = 2,
            stroke_color         = "#0f1923",
            background_image     = display_img,
            update_streamlit     = True,
            height               = canvas_h_adj,
            width                = canvas_w,
            drawing_mode         = "point",
            point_display_radius = 8,
            key                  = key,
            # initial_drawing verilmiyor: canvas_ver key ile zaten sıfırlanıyor
        )

        if (result is None
                or result.json_data is None
                or "objects" not in result.json_data):
            return corners

        objects = result.json_data["objects"]
        if not objects:
            return corners

        # Önizleme koordinatları → orijinal GeoTIFF pikselleri
        preview_pts = _parse_canvas_points(objects, display_scale)
        if not preview_pts:
            return corners

        new_corners = [
            [int(px / tiff_scale), int(py / tiff_scale)]
            for px, py in preview_pts
        ][:2]

        # json karşılaştırması: list == güvenilmez
        if json.dumps(new_corners) != json.dumps(corners):
            return new_corners
        return corners

    except ImportError:
        st.error("`streamlit-drawable-canvas` kurulu değil.  "
                 "`pip install streamlit-drawable-canvas`")
        return corners


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📁 Çalışma Klasörü")
    work_dir  = st.text_input("Proje klasörü",
                               value=str(Path.home() / "heximap_project"))
    work_path = Path(work_dir)

    st.divider()
    st.markdown("### ⚙️ Pipeline Adımları")
    run_stitch_cb    = st.checkbox("1 · Stitch",    value=True)
    run_extract_cb   = st.checkbox("2 · Extract",   value=True)
    run_georef_cb    = st.checkbox("3 · Georef",    value=True)
    run_rasterize_cb = st.checkbox("4 · Rasterize", value=True)

    st.divider()
    st.markdown("### 🛰️ Jeoreferans")
    utm_zone   = st.number_input("UTM Dilimi", 1, 60, 38)
    hemisphere = st.selectbox("Yarımküre", ["N", "S"])

    st.divider()
    st.markdown("### 🔧 Extract Parametreleri")
    resolution = st.selectbox("Disparity çözünürlüğü", ["1/2", "1/4", "full"])
    block_size = st.slider("SGBM blok boyutu", 3, 15, 7, step=2)

    st.divider()
    st.markdown("### 🗺️ Rasterize Parametreleri")
    do_clean     = st.checkbox("DEM temizleme",       value=True)
    do_median    = st.checkbox("Medyan filtresi",      value=False)
    do_denoise   = st.checkbox("Denoising",            value=True)
    export_ortho = st.checkbox("Ortofoto dışa aktar", value=True)

    st.divider()
    st.markdown("### 🔎 Stitch Kalite Eşiği")
    ransac_min_inlier = st.slider(
        "Min RANSAC inlier oranı (%)", 5, 80, 20,
        help=(
            "Bu oranın altındaki eşleşmeler hatalı kabul edilir "
            "ve stitch durdurulur. "
            "Düşük kaliteli görüntüler için 5–15 arası deneyin."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# ADIM 1 — STITCH
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hx-step">', unsafe_allow_html=True)
_step_header("stitch", "ADIM 1 · Stitch — Görüntü Yarılarını Birleştir")

st.caption(
    "Her Hexagon karesi için bir **sol (a) + sağ (b)** çifti ekleyin. "
    "'Önizleme Yükle' sonrası görüntüye tıklayarak **2 köşe** seçin: "
    "1. tıklama üst-sol · 2. tıklama alt-sağ."
)

btn_col, _ = st.columns([1, 7])
with btn_col:
    if st.button("➕ Çift Ekle", use_container_width=True):
        st.session_state["pairs"].append(dict(_PAIR_TEMPLATE))
        st.rerun()

pairs = st.session_state["pairs"]

for pi, pair in enumerate(pairs):
    # Geriye dönük uyumluluk: eski session'larda canvas_ver olmayabilir
    pair.setdefault("canvas_ver_l", 0)
    pair.setdefault("canvas_ver_r", 0)

    st.markdown('<div class="hx-pair-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="hx-pair-title">KARE {pi + 1}</div>',
                unsafe_allow_html=True)

    del_col, _ = st.columns([1, 11])
    with del_col:
        if len(pairs) > 1 and st.button("🗑", key=f"del_{pi}"):
            pairs.pop(pi)
            st.rerun()

    # ── Dosya yolları ──────────────────────────────────────────────────────
    path_l_col, path_r_col = st.columns(2)

    with path_l_col:
        st.markdown("**Sol yarı (a)**")
        zkey_l = f"file_l_{pi}"
        # Zenity sonucu geldiyse uygula
        if zkey_l in st.session_state["zenity_results"]:
            pair["file_l"]       = st.session_state["zenity_results"].pop(zkey_l)
            pair["preview_l"]    = None
            pair["corners_l"]    = []
            pair["canvas_ver_l"] += 1

        inp_c, btn_c = st.columns([5, 1])
        with inp_c:
            new_l = st.text_input(
                "Sol GeoTIFF", key=f"finp_l_{pi}",
                value=pair["file_l"],
                placeholder="/veri/DZB_001a.tif",
                label_visibility="collapsed",
            )
        with btn_c:
            _browse_button(
                zkey_l, f"fbtn_l_{pi}",
                start_dir=(str(Path(pair["file_l"]).parent)
                           if pair["file_l"] else str(Path.home())),
            )

        if new_l != pair["file_l"]:
            pair["file_l"]       = new_l
            pair["preview_l"]    = None
            pair["corners_l"]    = []
            pair["canvas_ver_l"] += 1

        if st.button("🔍 Önizleme Yükle", key=f"prev_l_{pi}"):
            if pair["file_l"] and Path(pair["file_l"]).exists():
                with st.spinner("Önizleme üretiliyor…"):
                    img, sc = _geotiff_preview(pair["file_l"])
                pair["preview_l"]    = img
                pair["scale_l"]      = sc
                pair["corners_l"]    = []
                pair["canvas_ver_l"] += 1
            else:
                st.error("Dosya bulunamadı.")

    with path_r_col:
        st.markdown("**Sağ yarı (b)**")
        zkey_r = f"file_r_{pi}"
        if zkey_r in st.session_state["zenity_results"]:
            pair["file_r"]       = st.session_state["zenity_results"].pop(zkey_r)
            pair["preview_r"]    = None
            pair["corners_r"]    = []
            pair["canvas_ver_r"] += 1

        inp_c, btn_c = st.columns([5, 1])
        with inp_c:
            new_r = st.text_input(
                "Sağ GeoTIFF", key=f"finp_r_{pi}",
                value=pair["file_r"],
                placeholder="/veri/DZB_001b.tif",
                label_visibility="collapsed",
            )
        with btn_c:
            _browse_button(
                zkey_r, f"fbtn_r_{pi}",
                start_dir=(str(Path(pair["file_r"]).parent)
                           if pair["file_r"] else str(Path.home())),
            )

        if new_r != pair["file_r"]:
            pair["file_r"]       = new_r
            pair["preview_r"]    = None
            pair["corners_r"]    = []
            pair["canvas_ver_r"] += 1

        if st.button("🔍 Önizleme Yükle", key=f"prev_r_{pi}"):
            if pair["file_r"] and Path(pair["file_r"]).exists():
                with st.spinner("Önizleme üretiliyor…"):
                    img, sc = _geotiff_preview(pair["file_r"])
                pair["preview_r"]    = img
                pair["scale_r"]      = sc
                pair["corners_r"]    = []
                pair["canvas_ver_r"] += 1
            else:
                st.error("Dosya bulunamadı.")

    # ── Köşe seçimi ────────────────────────────────────────────────────────
    st.markdown("**📍 Köşe Seçimi**")
    tab_canvas, tab_json = st.tabs(["🖱️ Önizleme üzerinden seç", "⌨️ Manuel JSON"])

    with tab_canvas:
        c_col_l, c_col_r = st.columns(2)

        with c_col_l:
            st.caption("Sol görüntü")
            if pair["preview_l"] is not None:
                # canvas_ver_l değişince key değişir → canvas temizlenir
                canvas_key_l = f"canvas_l_{pi}_v{pair['canvas_ver_l']}"
                new_cl = _canvas_corner_picker(
                    pair["preview_l"], pair["scale_l"],
                    pair["corners_l"], key=canvas_key_l,
                )
                if json.dumps(new_cl) != json.dumps(pair["corners_l"]):
                    pair["corners_l"] = new_cl
                    st.rerun()

                rst_l, info_l = st.columns([1, 3])
                with rst_l:
                    if st.button("🔄 Sıfırla", key=f"rst_l_{pi}"):
                        pair["corners_l"]    = []
                        pair["canvas_ver_l"] += 1
                        st.rerun()
                with info_l:
                    n = len(pair["corners_l"])
                    if n < 2:
                        st.info(f"{n}/2 köşe seçildi")
                    else:
                        st.success("✓ Tamamlandı")
                        st.caption(f"Sol: {pair['corners_l']}")
            else:
                st.info("Önce 'Önizleme Yükle' butonuna basın.")

        with c_col_r:
            st.caption("Sağ görüntü")
            if pair["preview_r"] is not None:
                canvas_key_r = f"canvas_r_{pi}_v{pair['canvas_ver_r']}"
                new_cr = _canvas_corner_picker(
                    pair["preview_r"], pair["scale_r"],
                    pair["corners_r"], key=canvas_key_r,
                )
                if json.dumps(new_cr) != json.dumps(pair["corners_r"]):
                    pair["corners_r"] = new_cr
                    st.rerun()

                rst_r, info_r = st.columns([1, 3])
                with rst_r:
                    if st.button("🔄 Sıfırla", key=f"rst_r_{pi}"):
                        pair["corners_r"]    = []
                        pair["canvas_ver_r"] += 1
                        st.rerun()
                with info_r:
                    n = len(pair["corners_r"])
                    if n < 2:
                        st.info(f"{n}/2 köşe seçildi")
                    else:
                        st.success("✓ Tamamlandı")
                        st.caption(f"Sağ: {pair['corners_r']}")
            else:
                st.info("Önce 'Önizleme Yükle' butonuna basın.")

    with tab_json:
        st.caption(
            "Format: `[[x_üst, y_üst], [x_alt, y_alt]]` — orijinal GeoTIFF pikseli"
        )
        jl_col, jr_col = st.columns(2)
        with jl_col:
            raw_l = st.text_area(
                "corners_l", key=f"jl_{pi}", height=80,
                value=json.dumps(pair["corners_l"]) if pair["corners_l"] else "",
            )
            if raw_l:
                try:
                    parsed = json.loads(raw_l)
                    if json.dumps(parsed) != json.dumps(pair["corners_l"]):
                        pair["corners_l"] = parsed
                except json.JSONDecodeError:
                    st.error("Geçersiz JSON")
        with jr_col:
            raw_r = st.text_area(
                "corners_r", key=f"jr_{pi}", height=80,
                value=json.dumps(pair["corners_r"]) if pair["corners_r"] else "",
            )
            if raw_r:
                try:
                    parsed = json.loads(raw_r)
                    if json.dumps(parsed) != json.dumps(pair["corners_r"]):
                        pair["corners_r"] = parsed
                except json.JSONDecodeError:
                    st.error("Geçersiz JSON")

    # ── Kare özet durumu ───────────────────────────────────────────────────
    nl, nr = len(pair["corners_l"]), len(pair["corners_r"])
    if nl == 2 and nr == 2 and pair["file_l"] and pair["file_r"]:
        st.success(f"✓ Kare {pi + 1} hazır")
    else:
        missing = []
        if not pair["file_l"]: missing.append("sol dosya")
        if not pair["file_r"]: missing.append("sağ dosya")
        if nl < 2:            missing.append(f"sol köşeler ({nl}/2)")
        if nr < 2:            missing.append(f"sağ köşeler ({nr}/2)")
        st.warning("Eksik: " + ", ".join(missing))

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ADIM 2 — EXTRACT
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hx-step">', unsafe_allow_html=True)
_step_header("extract", "ADIM 2 · Extract — Stereo DEM Çıkarma")

hex_files_raw = st.text_area(
    "Birleştirilmiş Hexagon dosya yolları (her satıra bir yol)",
    height=80, placeholder="/veri/hex_001.npz\n/veri/hex_002.npz",
    key="hex_files_raw",
)
ex_c1, ex_c2 = st.columns(2)
with ex_c1:
    roi_ids_raw = st.text_input(
        "ROI ID'leri (virgülle, boş=tümü)",
        placeholder="1, 2, 3", key="roi_ids_raw",
    )
with ex_c2:
    st.info(f"Çözünürlük: **{resolution}** · Blok: **{block_size}** (sidebar)")
st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ADIM 3 — GEOREF
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hx-step">', unsafe_allow_html=True)
_step_header("georef", "ADIM 3 · Georef — Jeoreferanslama Optimizasyonu")

# Zenity sonuçlarını uygula
if "ref_path_z" in st.session_state["zenity_results"]:
    st.session_state["ref_path"] = st.session_state["zenity_results"].pop("ref_path_z")
if "shp_path_z" in st.session_state["zenity_results"]:
    st.session_state["shp_path"] = st.session_state["zenity_results"].pop("shp_path_z")

g1, g2 = st.columns(2)
with g1:
    ri, rb = st.columns([5, 1])
    with ri:
        ref_path = st.text_input(
            "Referans DEM yolu",
            placeholder="/veri/ref_dem.tif", key="ref_path",
        )
    with rb:
        _browse_button("ref_path_z", "fbtn_ref")
with g2:
    si, sb = st.columns([5, 1])
    with si:
        shp_path = st.text_input(
            "Shapefile (opsiyonel)",
            placeholder="/veri/maske.shp", key="shp_path",
        )
    with sb:
        _browse_button("shp_path_z", "fbtn_shp")

st.caption(f"UTM Dilimi: **{utm_zone}{hemisphere}** (sidebar)")
st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ADIM 4 — RASTERIZE
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hx-step">', unsafe_allow_html=True)
_step_header("rasterize", "ADIM 4 · Rasterize — GeoTIFF Dışa Aktarma")

if "save_path_z" in st.session_state["zenity_results"]:
    st.session_state["save_path"] = st.session_state["zenity_results"].pop("save_path_z")

rs_i, rs_b = st.columns([8, 1])
with rs_i:
    save_path = st.text_input(
        "Çıktı klasörü",
        value=str(work_path / "output"), key="save_path",
    )
with rs_b:
    _browse_button("save_path_z", "fbtn_save", folder_only=True)

st.caption(
    f"temizleme={'açık' if do_clean else 'kapalı'} · "
    f"medyan={'açık' if do_median else 'kapalı'} · "
    f"denoising={'açık' if do_denoise else 'kapalı'} · "
    f"ortofoto={'evet' if export_ortho else 'hayır'}"
)
st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE ÇALIŞTIRMA
# ═══════════════════════════════════════════════════════════════════════════
st.divider()

run_col, status_col = st.columns([1, 3])
with run_col:
    run_btn = st.button(
        "▶  Pipeline'ı Çalıştır", type="primary",
        disabled=st.session_state["pipeline_running"],
        use_container_width=True,
    )
with status_col:
    states = st.session_state["step_states"]
    st.markdown("  →  ".join(
        f"{_STATE_ICON[states[s]]} {s}"
        for s in ["stitch", "extract", "georef", "rasterize"]
    ))

st.markdown("#### 📋 Log")
log_placeholder      = st.empty()
progress_placeholder = st.empty()
output_placeholder   = st.empty()

if run_btn and not st.session_state["pipeline_running"]:

    # ── Doğrulama ──────────────────────────────────────────────────────────
    errors = []
    if run_stitch_cb:
        for i, p in enumerate(pairs):
            if not p["file_l"]:
                errors.append(f"Kare {i+1}: sol dosya yolu eksik.")
            if not p["file_r"]:
                errors.append(f"Kare {i+1}: sağ dosya yolu eksik.")
            if len(p["corners_l"]) != 2:
                errors.append(
                    f"Kare {i+1}: sol için 2 köşe gerekli ({len(p['corners_l'])}/2).")
            if len(p["corners_r"]) != 2:
                errors.append(
                    f"Kare {i+1}: sağ için 2 köşe gerekli ({len(p['corners_r'])}/2).")
    if run_georef_cb and not st.session_state.get("ref_path"):
        errors.append("Referans DEM yolu girilmedi.")
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    # ── Parametre hazırlama ────────────────────────────────────────────────
    hex_files = [
        Path(p.strip())
        for p in (st.session_state.get("hex_files_raw") or "").splitlines()
        if p.strip()
    ]
    roi_ids = [
        int(x.strip())
        for x in (st.session_state.get("roi_ids_raw") or "").split(",")
        if x.strip().isdigit()
    ]

    stitch_pairs = [
        {
            "file_l":    p["file_l"],
            "file_r":    p["file_r"],
            "corners_l": p["corners_l"],
            "corners_r": p["corners_r"],
        }
        for p in pairs
    ]

    # corners_l/r: stitch'te seçilen köşeler (tüm karelerden birincisini al;
    # çift kare durumunda her npz aynı anda bir stereo çift oluşturur)
    _first_pair = pairs[0] if pairs else {}
    extract_params = {
        "hex_files":  hex_files,
        "roi_ids":    roi_ids,
        "resolution": resolution,
        "block_size": block_size,
        "save_path":  work_path / "extract_out",
        # Kullanıcının seçtiği köşe bölgeleri — RAM tasarrufu için kırpma sınırı
        "corners_l":  _first_pair.get("corners_l") or None,
        "corners_r":  _first_pair.get("corners_r") or None,
    }
    georef_params = {
        "ref_path":   st.session_state.get("ref_path", ""),
        "shp_path":   st.session_state.get("shp_path") or None,
        "zone":       utm_zone,
        "hemisphere": hemisphere,
        "opt":        {},
    }
    rasterize_params = {
        "save_path":    st.session_state.get("save_path", str(work_path / "output")),
        "do_clean":     do_clean,
        "do_median":    do_median,
        "do_denoise":   do_denoise,
        "export_ortho": export_ortho,
    }

    active_steps = [
        s for s, flag in [
            ("stitch",    run_stitch_cb),
            ("extract",   run_extract_cb),
            ("georef",    run_georef_cb),
            ("rasterize", run_rasterize_cb),
        ] if flag
    ]

    st.session_state["log_lines"]        = []
    st.session_state["step_states"]      = {
        s: "idle" for s in ["stitch", "extract", "georef", "rasterize"]
    }
    st.session_state["pipeline_running"] = True

    log_q  = Queue()
    prog_q = Queue()

    # Sidebar'daki RANSAC eşiğini thread'e kapatmak için yerel değişkene al
    _ransac_threshold = ransac_min_inlier

    def _run_thread():
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from pipeline_runner import (
                run_stitch, run_extract, run_georef, run_rasterize,
            )

            def log_cb(msg):      log_q.put(("log", msg))
            def prog_cb(s, t, m): log_q.put(("log", m)); prog_q.put((s, t, m))

            results: dict[str, Any] = {}
            all_npz: list = []   # stitch adımında biriken .npz yolları

            # ── Stitch ────────────────────────────────────────────────────
            if "stitch" in active_steps:
                log_q.put(("log", "Stitch adımı başlıyor…"))

                for idx, sp in enumerate(stitch_pairs):
                    log_q.put(("log",
                        f"  Kare {idx+1}/{len(stitch_pairs)} işleniyor…"))
                    try:
                        r = run_stitch(sp, log_cb=log_cb, progress_cb=prog_cb)
                    except Exception as exc:
                        import traceback
                        log_q.put(("log",
                            f"HATA (stitch kare {idx+1}): {exc}\n"
                            + traceback.format_exc()))
                        log_q.put(("done", results))
                        return

                    if not r.success:
                        log_q.put(("log",
                            f"HATA (stitch kare {idx+1}): {r.error}\n"
                            "İpucu: Köşe koordinatlarını kontrol edin."))
                        log_q.put(("done", results))
                        return

                    # RANSAC inlier oranı kontrolü (output'ta varsa)
                    if hasattr(r, "output") and isinstance(r.output, dict):
                        inlier_pct = r.output.get("inlier_pct")
                        if inlier_pct is not None:
                            log_q.put(("log",
                                f"  Kare {idx+1} RANSAC inlier: "
                                f"%{inlier_pct:.1f}"))
                            if inlier_pct < _ransac_threshold:
                                log_q.put(("log",
                                    f"HATA (stitch kare {idx+1}): "
                                    f"RANSAC inlier oranı çok düşük "
                                    f"(%{inlier_pct:.1f} < %{_ransac_threshold}). "
                                    "Stitch durduruldu.\n"
                                    "İpucu: Sidebar'dan eşiği düşürün veya "
                                    "köşe koordinatlarını düzeltin."))
                                log_q.put(("done", results))
                                return
                    # .npz yollarını biriktir (for döngüsü içi)
                    if hasattr(r, "output") and isinstance(r.output, dict):
                        all_npz.extend(r.output.get("npz_files", []))

                results["stitch"] = True
                log_q.put(("log",
                    f"Stitch adımı başarıyla tamamlandı. "
                    f"({len(stitch_pairs)} kare)"))

                # Stitch .npz çıktılarını extract_params'a yaz
                # Kullanıcı manuel yol girdiyse ona öncelik ver
                if all_npz and not extract_params.get("hex_files"):
                    extract_params["hex_files"] = all_npz
                    log_q.put(("log",
                        f"Extract'a otomatik aktarılan .npz "
                        f"({len(all_npz)} adet): "
                        + ", ".join(str(p) for p in all_npz)))
                elif not all_npz and not extract_params.get("hex_files"):
                    log_q.put(("log",
                        "UYARI: Stitch .npz çıktısı yakalanamadı. "
                        "Adım 2 metin kutusuna .npz yollarını manuel girin."))


            # ── Extract ───────────────────────────────────────────────────
            if "extract" in active_steps:
                log_q.put(("log", "Extract adımı başlıyor…"))
                r = run_extract(extract_params,
                                log_cb=log_cb, progress_cb=prog_cb)
                results["extract"] = r
                if not r.success:
                    log_q.put(("log", f"HATA (extract): {r.error}"))
                    log_q.put(("done", results))
                    return
                log_q.put(("log", "Extract adımı başarıyla tamamlandı."))

            # ── Georef ────────────────────────────────────────────────────
            if "georef" in active_steps:
                log_q.put(("log", "Georef adımı başlıyor…"))
                if "extract" in results and hasattr(results["extract"], "output"):
                    georef_params.setdefault(
                        "win_obj",
                        results["extract"].output.get(
                            "bundle_result", {}).get("win_obj"),
                    )
                r = run_georef(georef_params,
                               log_cb=log_cb, progress_cb=prog_cb)
                results["georef"] = r
                if not r.success:
                    log_q.put(("log", f"HATA (georef): {r.error}"))
                    log_q.put(("done", results))
                    return
                log_q.put(("log", "Georef adımı başarıyla tamamlandı."))

            # ── Rasterize ─────────────────────────────────────────────────
            if "rasterize" in active_steps:
                log_q.put(("log", "Rasterize adımı başlıyor…"))
                if "georef" in results and hasattr(results["georef"], "output"):
                    rasterize_params.setdefault(
                        "win_obj",
                        results["georef"].output.get(
                            "opti_result", {}).get("win_obj"),
                    )
                r = run_rasterize(rasterize_params,
                                  log_cb=log_cb, progress_cb=prog_cb)
                results["rasterize"] = r
                if not r.success:
                    log_q.put(("log", f"HATA (rasterize): {r.error}"))
                    log_q.put(("done", results))
                    return
                log_q.put(("log", "Rasterize adımı başarıyla tamamlandı."))

            log_q.put(("done", results))

        except Exception as exc:
            import traceback
            log_q.put(("error", f"{exc}\n{traceback.format_exc()}"))

    threading.Thread(target=_run_thread, daemon=True).start()

    # ── Pipeline polling döngüsü ───────────────────────────────────────────
    step_weights = {"stitch": 15, "extract": 55, "georef": 20, "rasterize": 10}
    total_weight = sum(step_weights[s] for s in active_steps)
    done_weight  = 0
    progress_bar = progress_placeholder.progress(0, text="Başlıyor…")
    final_results: dict[str, Any] = {}

    while True:
        updated    = False
        drain_done = False

        while True:
            try:
                msg = log_q.get_nowait()
            except Empty:
                break
            kind = msg[0]
            if kind == "log":
                _append_log(msg[1])
                updated = True
                txt = msg[1].lower()
                for s in active_steps:
                    if f"{s} adımı başlıyor"  in txt:
                        st.session_state["step_states"][s] = "running"
                    if f"{s} adımı başarıyla" in txt:
                        st.session_state["step_states"][s] = "done"
                        done_weight += step_weights.get(s, 0)
                    if f"hata ({s}" in txt:
                        st.session_state["step_states"][s] = "error"
            elif kind == "done":
                final_results = msg[1]
                drain_done    = True
                break
            elif kind == "error":
                _append_log(f"KRİTİK HATA: {msg[1]}")
                drain_done = True
                break

        if updated:
            _log_widget(log_placeholder)

        try:
            while True:
                _, _, pmsg = prog_q.get_nowait()
                pct = int(done_weight / total_weight * 100) if total_weight else 0
                progress_bar.progress(min(pct, 99), text=pmsg)
        except Empty:
            pass

        if drain_done:
            break
        time.sleep(0.3)

    st.session_state["last_results"]     = final_results
    st.session_state["pipeline_running"] = False
    progress_bar.progress(100, text="Tamamlandı")

    # ── Sonuç gösterimi ────────────────────────────────────────────────────
    ras = final_results.get("rasterize")
    if ras and hasattr(ras, "success") and ras.success:
        st.success("✅ Pipeline başarıyla tamamlandı!")
        dem_path = ras.output.get("dem_path")
        if dem_path and Path(str(dem_path)).exists():
            with output_placeholder.container():
                st.markdown("#### 🗺️ DEM Önizleme")
                try:
                    import rasterio
                    import matplotlib.pyplot as plt
                    import numpy as np
                    with rasterio.open(str(dem_path)) as src:
                        dem = src.read(1)
                        nd  = src.nodata
                    if nd is not None:
                        dem = np.where(dem == nd, np.nan, dem)
                    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#060e14")
                    ax.set_facecolor("#060e14")
                    im = ax.imshow(dem, cmap="terrain", aspect="auto")
                    plt.colorbar(im, ax=ax, label="Yükseklik (m)", shrink=0.8)
                    ax.set_title("DEM Çıktısı", color="#2dd4bf", fontsize=12)
                    ax.tick_params(colors="#94a3b8")
                    for sp in ax.spines.values():
                        sp.set_edgecolor("#1e3448")
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.info(f"DEM kaydedildi: `{dem_path}`\nÖnizleme hatası: {e}")
    else:
        st.error("❌ Pipeline hata ile sonuçlandı. Log'u inceleyin.")

    _log_widget(log_placeholder)
    st.rerun()

if not st.session_state["pipeline_running"]:
    _log_widget(log_placeholder)


# ═══════════════════════════════════════════════════════════════════════════
# Zenity polling — sayfa sonu
# ───────────────────────────────────────────────────────────────────────────
# Herhangi bir Zenity thread'i aktifse VEYA henüz session'a taşınmamış
# bir sonuç varsa, 1 saniye bekleyip rerun yapılır.
# Bu sayede _browse_button'da st.rerun() olmadan dosya yolu güncellenir.
# ═══════════════════════════════════════════════════════════════════════════
_any_zenity_running = any(_ZENITY_RUNNING.values())
_has_zenity_result  = bool(_ZENITY_RESULTS)

if _any_zenity_running or _has_zenity_result:
    time.sleep(1)
    st.rerun()


# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;'
    'color:#334155;text-align:center;">'
    'HEXIMAP · Python Port · '
    '<a href="https://github.com/Ksatura/heximap" style="color:#2dd4bf;'
    'text-decoration:none;">github.com/Ksatura/heximap</a></p>',
    unsafe_allow_html=True,
)
