from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

# Single active presentation entry point. Older score-layout wrappers are not
# imported here and therefore cannot override the current dissertation-style view.
from dissertation_example_layout_app import APP_VERSION, app

logger = logging.getLogger("tone_metric.dissertation")


def _request_meta(request: Request) -> str:
    return (
        f"path={request.url.path} "
        f"content_type={request.headers.get('content-type', '')!r} "
        f"content_length={request.headers.get('content-length', '')!r}"
    )


@app.exception_handler(RequestValidationError)
async def log_request_validation_error(request: Request, exc: RequestValidationError):
    detail = exc.errors()
    if request.url.path.startswith("/api/analyze"):
        message = f"ANALYZE_REQUEST_VALIDATION {_request_meta(request)} detail={detail}"
        logger.error(message)
        print(message, flush=True)
    return JSONResponse(status_code=422, content={"detail": detail})


@app.exception_handler(HTTPException)
async def log_http_exception(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api/analyze"):
        message = (
            f"ANALYZE_HTTP_EXCEPTION status={exc.status_code} "
            f"{_request_meta(request)} detail={exc.detail}"
        )
        logger.error(message)
        print(message, flush=True)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/diagnostics", include_in_schema=False)
def diagnostics():
    return {
        "ok": True,
        "version": APP_VERSION,
        "exception_logging": True,
        "request_validation_logging": True,
        "visualization": "exact-uploaded-pdf-dissertation-style-analysis-band-per-score-system",
        "legacy_layout_wrappers_active": False,
    }
