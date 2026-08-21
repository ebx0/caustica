"""A dependency-free local web server for the phantom studio.

Stdlib only, on purpose. The library's dependency policy is numpy + scipy +
pydantic + h5py, and a demo GUI is not a good enough reason to add a web
framework to a simulation package — nor to make the user install one before
they can look at a phantom. ``http.server`` is more than fast enough when
the client is on localhost and the payloads are numpy buffers.

Wire format:

* JSON for everything structural (catalog, specs, build summaries);
* raw ``application/octet-stream`` for pixel and voxel data, with the shape
  and value range carried in ``X-`` headers. Encoding a 200x300 slice as PNG
  server-side would cost more than shipping the bytes and let the browser
  recolour instantly when the user switches field or colour map.

Security posture: binds to 127.0.0.1 by default and is a single-user tool.
It writes files only under the phantom module's export directory.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from uwcem_phantoms.spec import PhantomSpec

from apps.phantom_studio.studio import Studio

# Fully resolved, not just its parent: the traversal guard below compares
# against target.resolve(), so any unresolved symlink component in this
# path would make every legitimate file look like an escape attempt.
STATIC_DIR = (Path(__file__).resolve().parent / "static").resolve()
MAX_BODY = 4 << 20  # a spec is a few kB; anything larger is a mistake or an attack


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "caustica-phantom-studio/1.0"
    studio: Studio

    # ------------------------------------------------------------- plumbing

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - stdlib hook
        if self.server.verbose:  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _binary(self, data, headers: dict) -> None:
        payload = memoryview(data)
        self._responded = True
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(payload.nbytes))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Meta", json.dumps(headers))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, exc: Exception, status: int = 400) -> None:
        if getattr(self, "_responded", False):
            # The body already started going out; a second status line would
            # corrupt the stream, and on a socket the client has closed it
            # raises again and escapes into socketserver's handle_error, which
            # prints a two-stage traceback even with logging silenced. A
            # cancelled fetch is normal browser behaviour, not an incident.
            self.close_connection = True
            return
        self._json(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc() if self.server.verbose else None,  # type: ignore[attr-defined]
            },
            status,
        )

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError(f"request body of {length} bytes is too large")
        return json.loads(self.rfile.read(length) or b"{}")

    # ----------------------------------------------------------------- GET

    def do_GET(self) -> None:  # noqa: N802 - stdlib hook
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path.startswith("/api/"):
                self._api_get(url.path, query)
            else:
                self._static(url.path)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True  # the client went away mid-response
        except KeyError as exc:
            self._error(exc, 404)
        except Exception as exc:  # noqa: BLE001 - the browser must see the message
            self._error(exc, 500)

    def _api_get(self, path: str, q: dict) -> None:
        s = self.studio
        if path == "/api/state":
            self._json(s.state())
        elif path == "/api/progress":
            msg, frac = s.progress
            self._json({"message": msg, "fraction": frac})
        elif path == "/api/tissues":
            self._json(s.tissues(q.get("model", "detailed"), float(q.get("f0", 1.0))))
        elif path == "/api/slice":
            data, meta = s.slice_bytes(
                q["build"], int(q["axis"]), int(q["index"]), q.get("field", "labels")
            )
            self._binary(data, meta)
        elif path == "/api/volume":
            data, meta = s.volume_bytes(
                q["build"], int(q.get("max_dim", 144)), q.get("field", "labels")
            )
            self._binary(data, meta)
        else:
            raise KeyError(f"no such endpoint: {path}")

    def handle_one_request(self) -> None:  # noqa: D102 - stdlib hook
        self._responded = False
        super().handle_one_request()

    def _static(self, path: str) -> None:
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (STATIC_DIR / rel).resolve()
        if not target.is_file() or STATIC_DIR not in target.parents:
            self.send_error(404, "not found")
            return
        if target.is_symlink():  # resolve() already followed it; refuse anyway
            self.send_error(404, "not found")
            return
        body = target.read_bytes()
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._responded = True
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # ---------------------------------------------------------------- POST

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook
        url = urlparse(self.path)
        try:
            body = self._body()
            s = self.studio
            if url.path == "/api/plan":
                self._json(s.plan(PhantomSpec.model_validate(body["spec"])))
            elif url.path == "/api/build":
                record = s.build(
                    PhantomSpec.model_validate(body["spec"]),
                    preview=bool(body.get("preview", True)),
                )
                self._json(record.describe())
            elif url.path == "/api/export":
                self._json(
                    s.export(
                        body["build"],
                        body.get("name") or None,
                        body.get("store_properties", "auto"),
                    )
                )
            elif url.path == "/api/fetch":
                self._json(s.fetch(body["phantom_id"], bool(body.get("with_pval", True))))
            else:
                raise KeyError(f"no such endpoint: {url.path}")
        except KeyError as exc:
            self._error(exc, 404)
        except Exception as exc:  # noqa: BLE001 - the browser must see the message
            self._error(exc, 400)


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, studio: Studio, verbose: bool):
        self.verbose = verbose
        handler = type("BoundHandler", (StudioHandler,), {"studio": studio})
        super().__init__(address, handler)


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    preview_mvox: float = 10.0,
    open_browser: bool = True,
    verbose: bool = False,
) -> None:
    """Run the studio until interrupted."""
    studio = Studio(preview_mvox=preview_mvox)
    server = StudioServer((host, port), studio, verbose)
    url = f"http://{host}:{server.server_port}/"
    print(f"phantom studio: {url}")
    print("  Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
