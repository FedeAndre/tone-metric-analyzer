from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

# Import only the current exact-PDF presentation wrapper. Older visualization
# wrappers remain in the repository for history but are not imported or executed.
from dissertation_exact_pdf_app_v2 import APP_VERSION, app

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
        "visualization": "exact-uploaded-pdf-overlay-single-omr-pass",
        "legacy_visualization_imported": False,
    }
