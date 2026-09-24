"""bench.datasets.download against a local HTTP server with Range support (no network)."""
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bench.datasets import download as D

PAYLOAD = bytes(range(256)) * 4096          # 1 MiB


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: bytes):
        rng = self.headers.get("Range")
        if rng and self.command == "GET":
            start = int(rng.split("=")[1].split("-")[0])
            part = body[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            self.send_header("Content-Length", str(len(part)))
            self.end_headers()
            self.wfile.write(part)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command == "GET":
            self.wfile.write(body)

    def do_HEAD(self):
        self._send(PAYLOAD)

    def do_GET(self):
        self._send(PAYLOAD)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/file.bin"
    srv.shutdown()


def test_fetch_resumes_partial_file(server, tmp_path):
    dest = tmp_path / "file.bin"
    (tmp_path / "file.bin.part").write_bytes(PAYLOAD[:300_000])
    sha = hashlib.sha256(PAYLOAD).hexdigest()
    r = D.fetch(server, dest, sha)
    assert dest.read_bytes() == PAYLOAD and r.sha256 == sha and not r.skipped
    assert not (tmp_path / "file.bin.part").exists()
    assert D.fetch(server, dest, sha).skipped


def test_fetch_rejects_wrong_checksum(server, tmp_path):
    with pytest.raises(RuntimeError, match="sha256"):
        D.fetch(server, tmp_path / "f.bin", "0" * 64)
    assert not (tmp_path / "f.bin").exists()


def test_fetch_many_and_checksums_file(server, tmp_path):
    rs = D.fetch_many([(server, tmp_path / f"f{i}.bin", None) for i in range(3)], workers=3)
    assert [r.path.name for r in rs] == ["f0.bin", "f1.bin", "f2.bin"]
    ck = tmp_path / "x.sha256"
    ck.write_text(f"# comment\n{rs[0].sha256}  audio/f0.bin\n")
    assert D.load_checksums(ck) == {"audio/f0.bin": rs[0].sha256}
    assert D.disk_usage(tmp_path) >= 3 * len(PAYLOAD)
