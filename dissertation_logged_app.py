from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response

from dissertation_app import APP_VERSION, app

logger = logging.getLogger("tone_metric.dissertation")


@app.exception_handler(HTTPException)
async def log_http_exception(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api/analyze"):
        logger.error(
            "ANALYZE_HTTP_EXCEPTION status=%s path=%s detail=%s",
            exc.status_code,
            request.url.path,
            exc.detail,
        )
        print(
            f"ANALYZE_HTTP_EXCEPTION status={exc.status_code} path={request.url.path} detail={exc.detail}",
            flush=True,
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/diagnostics", include_in_schema=False)
def diagnostics():
    return {"ok": True, "version": APP_VERSION, "exception_logging": True}
