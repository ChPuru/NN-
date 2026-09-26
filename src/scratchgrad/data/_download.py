"""Mirror-walking HTTP fetch with md5 verification. Standard library only.

Shared by the MNIST/Fashion-MNIST, CIFAR-10 and text loaders so that the
"try mirrors in order, send a browser User-Agent, verify the checksum, cache on
disk" logic exists exactly once.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

__all__ = ["md5_of", "fetch_bytes", "download_to", "DownloadError"]

# Several of these hosts (notably yann.lecun.com and the GitHub raw CDN) return
# 403 or reset the connection for the default urllib agent string.
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class DownloadError(RuntimeError):
    """Every mirror failed, or a downloaded file failed its checksum."""


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming md5 of a file, so a 163 MB tarball never lands in RAM twice."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch_bytes(urls: Sequence[str], timeout: float = 120.0) -> tuple[bytes, str]:
    """Return ``(payload, url_that_worked)``, trying ``urls`` in order.

    Args:
        urls: Mirrors, most-trusted first.
        timeout: Per-request socket timeout in seconds.

    Raises:
        DownloadError: If every mirror failed; the message lists each failure so
            a corporate proxy is distinguishable from a dead mirror.
    """
    problems: list[str] = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), url
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            problems.append(f"  {url}\n    {type(exc).__name__}: {exc}")
    raise DownloadError("all mirrors failed:\n" + "\n".join(problems))


def download_to(
    path: Path,
    urls: Sequence[str],
    md5: str | None = None,
    timeout: float = 120.0,
) -> Path:
    """Download ``urls`` into ``path`` unless a checksum-valid copy is cached.

    Args:
        path: Destination file; parent directories are created.
        urls: Mirrors, most-trusted first.
        md5: Expected checksum. A cached file whose md5 disagrees is re-fetched
            (it is usually a truncated download or an HTML error page saved
            under a ``.gz`` name); a *fresh* download whose md5 disagrees raises.
        timeout: Per-request socket timeout in seconds.

    Returns:
        ``path``.

    Raises:
        DownloadError: On checksum mismatch after a fresh download, or when no
            mirror responded.
    """
    path = Path(path)
    if path.exists() and (md5 is None or md5_of(path) == md5):
        return path

    payload, _ = fetch_bytes(urls, timeout=timeout)
    if md5 is not None:
        got = hashlib.md5(payload).hexdigest()
        if got != md5:
            raise DownloadError(
                f"checksum mismatch for {path.name}: expected {md5}, got {got}. "
                "The mirror served the wrong file (Fashion-MNIST and MNIST share "
                "filenames) or the transfer was truncated."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(payload)
    # Rename only after a complete write: an interrupted download must not leave
    # a short file that the cache check above would then have to catch.
    tmp.replace(path)
    return path
