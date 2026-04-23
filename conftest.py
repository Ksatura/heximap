"""
conftest.py — HEXIMAP test konfigürasyonu
─────────────────────────────────────────
CI ortamında open3d kurulu olmadığında ilgili testler otomatik atlanır.

Kullanım:
    open3d gerektiren her test dosyasının başına şunu ekleyin:

        open3d = pytest.importorskip("open3d")

    Veya tek bir test fonksiyonu için:

        @pytest.mark.skipif(
            not HAVE_OPEN3D, reason="open3d kurulu değil (CI ortamı)"
        )
        def test_something():
            ...
"""

import os
import pytest

# CI ortamında open3d'nin varlığını kontrol et
try:
    import open3d  # noqa: F401
    HAVE_OPEN3D = True
except ImportError:
    HAVE_OPEN3D = False

# Fixture: open3d'ye bağımlı testleri atla
skip_if_no_open3d = pytest.mark.skipif(
    not HAVE_OPEN3D,
    reason="open3d kurulu değil — CI ortamında atlanıyor (HEXIMAP_CI=1)"
)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_open3d: open3d kütüphanesi gerektiren testler (CI'da atlanır)"
    )


def pytest_collection_modifyitems(config, items):
    """open3d işaretli testleri CI'da otomatik atla."""
    if os.environ.get("HEXIMAP_CI") == "1" and not HAVE_OPEN3D:
        skip_marker = pytest.mark.skip(
            reason="open3d mevcut değil (HEXIMAP_CI=1, requirements-ci.txt kullanılıyor)"
        )
        for item in items:
            if "requires_open3d" in item.keywords:
                item.add_marker(skip_marker)
