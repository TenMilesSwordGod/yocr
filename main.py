"""Dev entrypoint: python main.py  (equivalent to `uv run yocr serve`)."""

import uvicorn

from yocr.app import configure_logging
from yocr.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        "yocr.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
    )
