"""
app.py — HEXIMAP Streamlit Arayüzü  v4
========================================
Çalıştırma:
    cd ~/heximap/python
    streamlit run app.py

Bağımlılıklar:
    pip install streamlit streamlit-drawable-canvas Pillow rasterio matplotlib
"""

from __future__ import annotations

import warnings
try:
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
except ImportError:
    pass

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

# ── Sayfa yapılandırması ───────────────────────────────────────────────────
st.set_page_config(
    page_title="HEXIMAP",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.hx-header {
    background: linear-gradient(135deg,#0f1923 0%,#1a2e40 100%);
    border-bottom: 2px solid #2dd4bf;
    padding: 1.4rem 2rem; margin: -1rem -1rem 2rem -1rem;
    display:flex; align-items:baseline; gap:1rem;
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

st.markdown("""
<div class="hx-header">
    <h1 class="hx-title">⬡ HEXIMAP</h1>
    <p class="hx-subtitle">Hexagon KH-9 Imagery · DEM Extraction Pipeline</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
_PAIR_TEMPLATE: dict[str, Any] = {
    "file_l": "", "file_r": "",
    "corners_l": [],  # [[x_üst,y_üst],[x_alt,y_alt]]
    "corners_r": [],
    "preview_l": None, "scale_l": 1.0,
    "preview_r": None, "scale_r": 1.0,
}

_defaults: dict[str, Any] = {
    "pairs":             [dict(_PAIR_TEMPLATE)],
    "log_lines":         [],
    "step_states":       {s:"idle" for s in ["stitch","extract","georef","rasterize"]},
    "pipeline_running":  False,
    "last_results":      {},
    # Zenity thread sonuçları: {"key": yol}
    "zenity_results":    {},
    "zenity_running":    {},   # {"key": True/False}
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Yardımcılar ────────────────────────────────────────────────────────────
_STATE_ICON = {"idle":"○","running":"◌","done":"✓","error":"✗"}
_STATE_CSS  = {"idle":"","running":"hx-step-run","done":"hx-step-done","error":"hx-step-error"}

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
    html = ('<div class="hx-log">' +
            "\n".join(st.session_state["log_lines"][-80:]) +
            "</div>")
    if placeholder:
        placeholder.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


def _zenity_browse(session_key: str, start_dir: str = "",
                   folder_only: bool = False) -> None:
    """
    Zenity'i ayrı bir thread'de başlatır.
    Streamlit donmaz; sonuç st.session_state['zenity_results'][session_key]'e yazılır.
    Kullanım:
        _zenity_browse("file_l_0")
        # Sonraki rerun'da:
        result = st.session_state["zenity_results"].pop("file_l_0", None)
    """
    if st.session_state["zenity_running"].get(session_key):
        return  # zaten çalışıyor

    st.session_state["zenity_running"][session_key] = True

    def _run():
        env = os.environ.copy()
        env.setdefault("DISPLAY", ":0")
        try:
            if folder_only:
                cmd = ["zenity", "--file-selection", "--directory",
                       "--title=Klasör Seç"]
            else:
                cmd = ["zenity", "--file-selection",
                       "--title=Dosya Seç",
                       "--file-filter=GeoTIFF | *.tif *.tiff *.TIF *.TIFF",
                       "--file-filter=Shapefile | *.shp",
                       "--file-filter=Tüm dosyalar | *"]
            if start_dir and Path(start_dir).exists():
                cmd += [f"--filename={start_dir}/"]
            # Pencereyi öne getir
            cmd += ["--modal"]
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=300, env=env)
            path = r.stdout.strip()
            if path:
                st.session_state["zenity_results"][session_key] = path
        except Exception:
            pass
        finally:
            st.session_state["zenity_running"][session_key] = False

    threading.Thread(target=_run, daemon=True).start()


def _geotiff_preview(tiff_path: str, max_px: int = 800):
    """GeoTIFF'ten hafıza dostu PIL önizleme — rasterio out_shape ile."""
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
                data = src.read([1,2,3], out_shape=(3,out_h,out_w),
                                resampling=Resampling.average)
                data = np.moveaxis(data, 0, -1)
            else:
                data = src.read(1, out_shape=(out_h,out_w),
                                resampling=Resampling.average)
                data = np.stack([data]*3, axis=-1)

        data = np.nan_to_num(data.astype(float), nan=0.0)
        lo, hi = data.min(), data.max()
        data = ((data-lo)/(hi-lo)*255).astype("uint8") if hi>lo \
               else np.zeros_like(data, dtype="uint8")
        return Image.fromarray(data, "RGB"), scale
    except Exception:
        return None, 1.0


def _canvas_corner_picker(img, scale: float, corners: list,
                           key: str, canvas_h: int = 480) -> list:
    """
    streamlit-drawable-canvas ile köşe seçimi.
    Kullanıcı görüntü üzerinde nokta çizer (drawing_mode="point").
    Döndürür: güncellenmiş corners listesi [[x_orig,y_orig], ...]
    """
    try:
        try:
            from streamlit_drawable_canvas import st_canvas
        except ImportError:
            from streamlit_drawable_canvas_fix import st_canvas
        from PIL import Image as PILImage
        import numpy as np

        iw, ih = img.size
        # Canvas genişliği sütun genişliğine uysun
        canvas_w = min(iw, 720)
        display_scale = canvas_w / iw
        canvas_h_adj  = int(ih * display_scale)

        # Önizlemeyi canvas boyutuna getir
        display_img = img.resize((canvas_w, canvas_h_adj), PILImage.LANCZOS)

        # Mevcut köşeleri başlangıç çizimi olarak göster
        initial_drawing = {"version": "4.4.0", "objects": []}
        for c in corners:
            cx = c[0] * scale * display_scale
            cy = c[1] * scale * display_scale
            initial_drawing["objects"].append({
                "type": "circle",
                "left": cx - 6, "top": cy - 6,
                "radius": 6,
                "fill": "#2dd4bf", "stroke": "#0f1923", "strokeWidth": 2,
                "selectable": False,
            })

        st.caption(
            "**Nokta aracı aktif** — görüntüye tıklayarak köşe seçin. "
            "Yanlış seçimde 'Sıfırla' butonuna basın. "
            "1. tıklama = üst-sol · 2. tıklama = alt-sağ"
        )

        result = st_canvas(
            fill_color   = "#2dd4bf",
            stroke_width = 2,
            stroke_color = "#0f1923",
            background_image = display_img,
            update_streamlit = True,
            height       = canvas_h_adj,
            width        = canvas_w,
            drawing_mode = "point",
            point_display_radius = 7,
            key          = key,
            initial_drawing = initial_drawing if corners else None,
        )

        # Canvas'tan yeni noktaları oku
        new_corners = list(corners)  # mevcut köşelerle başla
        if (result is not None and
                result.json_data is not None and
                "objects" in result.json_data):
            objects = result.json_data["objects"]
            # Yalnızca nokta/circle tipindeki nesneleri al
            pts = []
            for obj in objects:
                if obj.get("type") in ("circle", "point"):
                    # canvas koordinatı → orijinal GeoTIFF koordinatı
                    cx = obj.get("left", 0) + obj.get("radius", 0)
                    cy = obj.get("top",  0) + obj.get("radius", 0)
                    orig_x = int(cx / display_scale / scale)
                    orig_y = int(cy / display_scale / scale)
                    pts.append([orig_x, orig_y])

            # Maksimum 2 köşe
            if pts:
                new_corners = pts[:2]

        return new_corners

    except ImportError:
        st.error("`streamlit-drawable-canvas` kurulu değil. "
                 "`pip install streamlit-drawable-canvas`")
        return corners


# ── Gözat butonu helper ────────────────────────────────────────────────────
def _browse_button(session_key: str, btn_key: str,
                   start_dir: str = "", folder_only: bool = False) -> None:
    """
    Gözat butonu + zenity thread başlatıcı.
    Sonuç bir sonraki rerun'da session_state['zenity_results'] içinde olur.
    """
    is_running = st.session_state["zenity_running"].get(session_key, False)
    label = "⏳" if is_running else "📂"
    if st.button(label, key=btn_key, help="Dosya seç",
                 disabled=is_running):
        _zenity_browse(session_key, start_dir=start_dir,
                       folder_only=folder_only)
        time.sleep(0.3)   # thread'in başlaması için kısa bekle
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📁 Çalışma Klasörü")
    work_dir  = st.text_input("Proje klasörü",
                               value=str(Path.home()/"heximap_project"))
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
    hemisphere = st.selectbox("Yarımküre", ["N","S"])

    st.divider()
    st.markdown("### 🔧 Extract Parametreleri")
    resolution = st.selectbox("Disparity çözünürlüğü", ["1/2","1/4","full"])
    block_size = st.slider("SGBM blok boyutu", 3, 15, 7, step=2)

    st.divider()
    st.markdown("### 🗺️ Rasterize Parametreleri")
    do_clean     = st.checkbox("DEM temizleme",       value=True)
    do_median    = st.checkbox("Medyan filtresi",      value=False)
    do_denoise   = st.checkbox("Denoising",            value=True)
    export_ortho = st.checkbox("Ortofoto dışa aktar", value=True)

# Zenity sonuçlarını kontrol et ve gerekirse rerun
_has_zenity = bool(st.session_state["zenity_results"])

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

btn_col, _ = st.columns([1,7])
with btn_col:
    if st.button("➕ Çift Ekle", use_container_width=True):
        st.session_state["pairs"].append(dict(_PAIR_TEMPLATE))
        st.rerun()

pairs = st.session_state["pairs"]

for pi, pair in enumerate(pairs):
    st.markdown('<div class="hx-pair-box">', unsafe_allow_html=True)
    st.markdown(f'<div class="hx-pair-title">KARE {pi+1}</div>',
                unsafe_allow_html=True)

    del_col, _ = st.columns([1,11])
    with del_col:
        if len(pairs) > 1 and st.button("🗑", key=f"del_{pi}"):
            pairs.pop(pi); st.rerun()

    # ── Dosya yolları ──────────────────────────────────────────────────────
    path_l_col, path_r_col = st.columns(2)

    with path_l_col:
        st.markdown("**Sol yarı (a)**")
        # Zenity sonucu varsa al
        zkey_l = f"file_l_{pi}"
        if zkey_l in st.session_state["zenity_results"]:
            pair["file_l"]    = st.session_state["zenity_results"].pop(zkey_l)
            pair["preview_l"] = None
            pair["corners_l"] = []

        inp_c, btn_c = st.columns([5,1])
        with inp_c:
            new_l = st.text_input("Sol GeoTIFF", key=f"finp_l_{pi}",
                                   value=pair["file_l"],
                                   placeholder="/veri/DZB_001a.tif",
                                   label_visibility="collapsed")
        with btn_c:
            _browse_button(zkey_l, f"fbtn_l_{pi}",
                           start_dir=str(Path(pair["file_l"]).parent)
                           if pair["file_l"] else str(Path.home()))
        if new_l != pair["file_l"]:
            pair["file_l"] = new_l
            pair["preview_l"] = None
            pair["corners_l"] = []

        if st.button("🔍 Önizleme Yükle", key=f"prev_l_{pi}"):
            if pair["file_l"] and Path(pair["file_l"]).exists():
                with st.spinner("Önizleme üretiliyor…"):
                    img, sc = _geotiff_preview(pair["file_l"])
                pair["preview_l"] = img
                pair["scale_l"]   = sc
                pair["corners_l"] = []
            else:
                st.error("Dosya bulunamadı.")

    with path_r_col:
        st.markdown("**Sağ yarı (b)**")
        zkey_r = f"file_r_{pi}"
        if zkey_r in st.session_state["zenity_results"]:
            pair["file_r"]    = st.session_state["zenity_results"].pop(zkey_r)
            pair["preview_r"] = None
            pair["corners_r"] = []

        inp_c, btn_c = st.columns([5,1])
        with inp_c:
            new_r = st.text_input("Sağ GeoTIFF", key=f"finp_r_{pi}",
                                   value=pair["file_r"],
                                   placeholder="/veri/DZB_001b.tif",
                                   label_visibility="collapsed")
        with btn_c:
            _browse_button(zkey_r, f"fbtn_r_{pi}",
                           start_dir=str(Path(pair["file_r"]).parent)
                           if pair["file_r"] else str(Path.home()))
        if new_r != pair["file_r"]:
            pair["file_r"] = new_r
            pair["preview_r"] = None
            pair["corners_r"] = []

        if st.button("🔍 Önizleme Yükle", key=f"prev_r_{pi}"):
            if pair["file_r"] and Path(pair["file_r"]).exists():
                with st.spinner("Önizleme üretiliyor…"):
                    img, sc = _geotiff_preview(pair["file_r"])
                pair["preview_r"] = img
                pair["scale_r"]   = sc
                pair["corners_r"] = []
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
                new_cl = _canvas_corner_picker(
                    pair["preview_l"], pair["scale_l"],
                    pair["corners_l"], key=f"canvas_l_{pi}"
                )
                if new_cl != pair["corners_l"]:
                    pair["corners_l"] = new_cl
                if st.button("Sıfırla", key=f"rst_l_{pi}"):
                    pair["corners_l"] = []
                    st.rerun()
                n = len(pair["corners_l"])
                st.info(f"{n}/2 köşe seçildi") if n < 2 else st.success("✓ Tamamlandı")
            else:
                st.info("Önce 'Önizleme Yükle' butonuna basın.")

        with c_col_r:
            st.caption("Sağ görüntü")
            if pair["preview_r"] is not None:
                new_cr = _canvas_corner_picker(
                    pair["preview_r"], pair["scale_r"],
                    pair["corners_r"], key=f"canvas_r_{pi}"
                )
                if new_cr != pair["corners_r"]:
                    pair["corners_r"] = new_cr
                if st.button("Sıfırla", key=f"rst_r_{pi}"):
                    pair["corners_r"] = []
                    st.rerun()
                n = len(pair["corners_r"])
                st.info(f"{n}/2 köşe seçildi") if n < 2 else st.success("✓ Tamamlandı")
            else:
                st.info("Önce 'Önizleme Yükle' butonuna basın.")

    with tab_json:
        st.caption("Format: `[[x_üst, y_üst], [x_alt, y_alt]]` — orijinal GeoTIFF pikseli")
        jl_col, jr_col = st.columns(2)
        with jl_col:
            raw_l = st.text_area("corners_l", key=f"jl_{pi}", height=80,
                                  value=json.dumps(pair["corners_l"])
                                  if pair["corners_l"] else "")
            if raw_l:
                try:
                    pair["corners_l"] = json.loads(raw_l)
                except json.JSONDecodeError:
                    st.error("Geçersiz JSON")
        with jr_col:
            raw_r = st.text_area("corners_r", key=f"jr_{pi}", height=80,
                                  value=json.dumps(pair["corners_r"])
                                  if pair["corners_r"] else "")
            if raw_r:
                try:
                    pair["corners_r"] = json.loads(raw_r)
                except json.JSONDecodeError:
                    st.error("Geçersiz JSON")

    # Özet
    nl, nr = len(pair["corners_l"]), len(pair["corners_r"])
    if nl == 2 and nr == 2 and pair["file_l"] and pair["file_r"]:
        st.success(f"✓ Kare {pi+1} hazır")
    else:
        missing = []
        if not pair["file_l"]: missing.append("sol dosya")
        if not pair["file_r"]: missing.append("sağ dosya")
        if nl < 2: missing.append(f"sol köşeler ({nl}/2)")
        if nr < 2: missing.append(f"sağ köşeler ({nr}/2)")
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
    roi_ids_raw = st.text_input("ROI ID'leri (virgülle, boş=tümü)",
                                 placeholder="1, 2, 3", key="roi_ids_raw")
with ex_c2:
    st.info(f"Çözünürlük: **{resolution}** · Blok: **{block_size}** (sidebar)")
st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# ADIM 3 — GEOREF
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hx-step">', unsafe_allow_html=True)
_step_header("georef", "ADIM 3 · Georef — Jeoreferanslama Optimizasyonu")

# Zenity sonuçları
if "ref_path_z" in st.session_state["zenity_results"]:
    st.session_state["ref_path"] = st.session_state["zenity_results"].pop("ref_path_z")
if "shp_path_z" in st.session_state["zenity_results"]:
    st.session_state["shp_path"] = st.session_state["zenity_results"].pop("shp_path_z")

g1, g2 = st.columns(2)
with g1:
    ri, rb = st.columns([5,1])
    with ri:
        ref_path = st.text_input("Referans DEM yolu",
                                  placeholder="/veri/ref_dem.tif", key="ref_path")
    with rb:
        _browse_button("ref_path_z", "fbtn_ref")

with g2:
    si, sb = st.columns([5,1])
    with si:
        shp_path = st.text_input("Shapefile (opsiyonel)",
                                  placeholder="/veri/maske.shp", key="shp_path")
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

rs_i, rs_b = st.columns([8,1])
with rs_i:
    save_path = st.text_input("Çıktı klasörü",
                               value=str(work_path/"output"), key="save_path")
with rs_b:
    _browse_button("save_path_z", "fbtn_save", folder_only=True)

st.caption(
    f"temizleme={'açık' if do_clean else 'kapalı'} · "
    f"medyan={'açık' if do_median else 'kapalı'} · "
    f"denoising={'açık' if do_denoise else 'kapalı'} · "
    f"ortofoto={'evet' if export_ortho else 'hayır'}"
)
st.markdown('</div>', unsafe_allow_html=True)

# Zenity thread'leri bittiyse rerun (sonuçları göster)
if _has_zenity:
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE ÇALIŞTIRMA
# ═══════════════════════════════════════════════════════════════════════════
st.divider()

run_col, status_col = st.columns([1,3])
with run_col:
    run_btn = st.button("▶  Pipeline'ı Çalıştır", type="primary",
                        disabled=st.session_state["pipeline_running"],
                        use_container_width=True)
with status_col:
    states = st.session_state["step_states"]
    st.markdown("  →  ".join(
        f"{_STATE_ICON[states[s]]} {s}"
        for s in ["stitch","extract","georef","rasterize"]
    ))

st.markdown("#### 📋 Log")
log_placeholder      = st.empty()
progress_placeholder = st.empty()
output_placeholder   = st.empty()

if run_btn and not st.session_state["pipeline_running"]:

    # Doğrulama
    errors = []
    if run_stitch_cb:
        for i, p in enumerate(pairs):
            if not p["file_l"]: errors.append(f"Kare {i+1}: sol dosya yolu eksik.")
            if not p["file_r"]: errors.append(f"Kare {i+1}: sağ dosya yolu eksik.")
            if len(p["corners_l"]) != 2:
                errors.append(f"Kare {i+1}: sol için 2 köşe gerekli ({len(p['corners_l'])}/2).")
            if len(p["corners_r"]) != 2:
                errors.append(f"Kare {i+1}: sağ için 2 köşe gerekli ({len(p['corners_r'])}/2).")
    if run_georef_cb and not st.session_state.get("ref_path"):
        errors.append("Referans DEM yolu girilmedi.")
    if errors:
        for e in errors: st.error(e)
        st.stop()

    hex_files = [Path(p.strip())
                 for p in (st.session_state.get("hex_files_raw") or "").splitlines()
                 if p.strip()]
    roi_ids   = [int(x.strip())
                 for x in (st.session_state.get("roi_ids_raw") or "").split(",")
                 if x.strip().isdigit()]

    stitch_pairs = [{"file_l": p["file_l"], "file_r": p["file_r"],
                     "corners_l": p["corners_l"], "corners_r": p["corners_r"]}
                    for p in pairs]

    extract_params   = {"hex_files": hex_files, "roi_ids": roi_ids,
                        "resolution": resolution, "block_size": block_size,
                        "save_path": work_path/"extract_out"}
    georef_params    = {"ref_path": st.session_state.get("ref_path",""),
                        "shp_path": st.session_state.get("shp_path") or None,
                        "zone": utm_zone, "hemisphere": hemisphere, "opt": {}}
    rasterize_params = {"save_path": st.session_state.get("save_path",
                                                            str(work_path/"output")),
                        "do_clean": do_clean, "do_median": do_median,
                        "do_denoise": do_denoise, "export_ortho": export_ortho}

    active_steps = [s for s, flag in [
        ("stitch", run_stitch_cb), ("extract", run_extract_cb),
        ("georef", run_georef_cb), ("rasterize", run_rasterize_cb)] if flag]

    st.session_state["log_lines"]        = []
    st.session_state["step_states"]      = {s:"idle" for s in ["stitch","extract","georef","rasterize"]}
    st.session_state["pipeline_running"] = True

    log_q  = Queue()
    prog_q = Queue()

    def _run_thread():
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from pipeline_runner import (run_stitch, run_extract,
                                         run_georef, run_rasterize, RunResult)

            def log_cb(msg):      log_q.put(("log", msg))
            def prog_cb(s,t,m):   log_q.put(("log",m)); prog_q.put((s,t,m))

            results: dict[str,Any] = {}

            if "stitch" in active_steps:
                log_q.put(("log","Stitch adımı başlıyor…"))
                for idx, sp in enumerate(stitch_pairs):
                    log_q.put(("log",f"  Kare {idx+1}/{len(stitch_pairs)} işleniyor…"))
                    r = run_stitch(sp, log_cb=log_cb, progress_cb=prog_cb)
                    if not r.success:
                        log_q.put(("log",f"HATA (stitch kare {idx+1}): {r.error}"))
                        log_q.put(("done", results)); return
                results["stitch"] = True
                log_q.put(("log",f"Stitch adımı başarıyla tamamlandı. ({len(stitch_pairs)} kare)"))

            if "extract" in active_steps:
                r = run_extract(extract_params, log_cb=log_cb, progress_cb=prog_cb)
                results["extract"] = r
                if not r.success: log_q.put(("done",results)); return

            if "georef" in active_steps:
                if "extract" in results and hasattr(results["extract"],"output"):
                    georef_params.setdefault("win_obj",
                        results["extract"].output.get("bundle_result",{}).get("win_obj"))
                r = run_georef(georef_params, log_cb=log_cb, progress_cb=prog_cb)
                results["georef"] = r
                if not r.success: log_q.put(("done",results)); return

            if "rasterize" in active_steps:
                if "georef" in results and hasattr(results["georef"],"output"):
                    rasterize_params.setdefault("win_obj",
                        results["georef"].output.get("opti_result",{}).get("win_obj"))
                r = run_rasterize(rasterize_params, log_cb=log_cb, progress_cb=prog_cb)
                results["rasterize"] = r

            log_q.put(("done", results))
        except Exception as exc:
            import traceback
            log_q.put(("error", f"{exc}\n{traceback.format_exc()}"))

    threading.Thread(target=_run_thread, daemon=True).start()

    step_weights = {"stitch":15,"extract":55,"georef":20,"rasterize":10}
    total_weight = sum(step_weights[s] for s in active_steps)
    done_weight  = 0
    progress_bar = progress_placeholder.progress(0, text="Başlıyor…")
    final_results: dict[str,Any] = {}

    while True:
        updated = False; drain_done = False
        while True:
            try: msg = log_q.get_nowait()
            except Empty: break
            kind = msg[0]
            if kind == "log":
                _append_log(msg[1]); updated = True
                txt = msg[1].lower()
                for s in active_steps:
                    if f"{s} adımı başlıyor"  in txt: st.session_state["step_states"][s]="running"
                    if f"{s} adımı başarıyla" in txt:
                        st.session_state["step_states"][s]="done"
                        done_weight += step_weights.get(s,0)
                    if f"hata ({s}" in txt: st.session_state["step_states"][s]="error"
            elif kind == "done":
                final_results = msg[1]; drain_done = True; break
            elif kind == "error":
                _append_log(f"KRİTİK HATA: {msg[1]}"); drain_done = True; break

        if updated: _log_widget(log_placeholder)
        try:
            while True:
                _,_,pmsg = prog_q.get_nowait()
                pct = int(done_weight/total_weight*100) if total_weight else 0
                progress_bar.progress(min(pct,99), text=pmsg)
        except Empty: pass

        if drain_done: break
        time.sleep(0.3)

    st.session_state["last_results"]     = final_results
    st.session_state["pipeline_running"] = False
    progress_bar.progress(100, text="Tamamlandı")

    ras = final_results.get("rasterize")
    if ras and hasattr(ras,"success") and ras.success:
        st.success("✅ Pipeline başarıyla tamamlandı!")
        dem_path = ras.output.get("dem_path")
        if dem_path and Path(str(dem_path)).exists():
            with output_placeholder.container():
                st.markdown("#### 🗺️ DEM Önizleme")
                try:
                    import rasterio, matplotlib.pyplot as plt, numpy as np
                    with rasterio.open(str(dem_path)) as src:
                        dem = src.read(1); nd = src.nodata
                    if nd is not None: dem = np.where(dem==nd, np.nan, dem)
                    fig, ax = plt.subplots(figsize=(10,5), facecolor="#060e14")
                    ax.set_facecolor("#060e14")
                    im = ax.imshow(dem, cmap="terrain", aspect="auto")
                    plt.colorbar(im, ax=ax, label="Yükseklik (m)", shrink=0.8)
                    ax.set_title("DEM Çıktısı", color="#2dd4bf", fontsize=12)
                    ax.tick_params(colors="#94a3b8")
                    for sp in ax.spines.values(): sp.set_edgecolor("#1e3448")
                    st.pyplot(fig); plt.close(fig)
                except Exception as e:
                    st.info(f"DEM kaydedildi: `{dem_path}`\nÖnizleme hatası: {e}")
    else:
        st.error("❌ Pipeline hata ile sonuçlandı. Log'u inceleyin.")

    _log_widget(log_placeholder)
    st.rerun()

if not st.session_state["pipeline_running"]:
    _log_widget(log_placeholder)

st.divider()
st.markdown(
    '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;'
    'color:#334155;text-align:center;">'
    'HEXIMAP · Python Port · '
    '<a href="https://github.com/Ksatura/heximap" style="color:#2dd4bf;'
    'text-decoration:none;">github.com/Ksatura/heximap</a></p>',
    unsafe_allow_html=True,
)
