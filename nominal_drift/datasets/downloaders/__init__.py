"""
nominal_drift.datasets.downloaders
===================================

Dataset download layer for crystal structure benchmark datasets.

Public API
----------
DownloadResult   — Result object from each downloader
BaseDownloader   — Abstract base for all downloaders
Perov5Downloader, MP20Downloader, MPTS52Downloader, Carbon24Downloader
"""

from nominal_drift.datasets.downloaders.base_downloader import (
    BaseDownloader,
    DownloadResult,
)

__all__ = [
    "BaseDownloader",
    "DownloadResult",
]
