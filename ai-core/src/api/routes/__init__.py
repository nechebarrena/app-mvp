"""API Routes."""

from .videos import router as videos_router
from .bench import info_router, bench_router, assets_router

__all__ = ["videos_router", "info_router", "bench_router", "assets_router"]
