"""FastAPI and WebSocket transport layer exposing faro-cv to the Next.js workbench."""

from faro_cv.api.app import app, create_app

__all__ = ["app", "create_app"]
