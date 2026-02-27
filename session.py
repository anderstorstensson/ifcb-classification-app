import atexit
import logging
import os
import shutil
import tempfile
import threading

IMAGE_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif', '.webp',
}

IMAGES_PER_PAGE = 100
MAX_ZIP_BYTES = 200 * 1024 * 1024  # 200 MB
MAX_ZIP_FILES = 10_000
MAX_SINGLE_FILE_BYTES = 50 * 1024 * 1024  # 50 MB

log = logging.getLogger(__name__)

_active_session_dirs = set()
_session_lock = threading.Lock()


@atexit.register
def _cleanup_all_sessions():
    with _session_lock:
        for d in _active_session_dirs:
            shutil.rmtree(d, ignore_errors=True)
        _active_session_dirs.clear()


def init_session(session):
    if session is None:
        session_dir = tempfile.mkdtemp(prefix="ifcb-session-")
        with _session_lock:
            _active_session_dirs.add(session_dir)
        return {"dir": session_dir, "paths": []}
    return session


def save_image(pil_image, session, original_name=""):
    session = init_session(session)
    idx = len(session["paths"])
    path = os.path.join(session["dir"], f"{idx:06d}.png")
    pil_image.save(path)
    w, h = pil_image.size
    return {
        **session,
        "paths": [*session["paths"], path],
        "sizes": [*session.get("sizes", []), w * h],
        "names": [*session.get("names", []), original_name],
    }


def extract_roi(name):
    stem = os.path.splitext(name)[0]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return str(int(parts[1]))
    return ""


def sorted_indices(session, sort_by_dim):
    if not session:
        return []
    n = len(session["paths"])
    if sort_by_dim:
        sizes = session.get("sizes", [])
        return sorted(range(n), key=lambda i: sizes[i] if i < len(sizes) else 0, reverse=True)
    names = session.get("names", [])
    return sorted(range(n), key=lambda i: names[i] if i < len(names) else "")


def gallery_page(session, page, sort_by_dim=True):
    if not session:
        return []
    indices = sorted_indices(session, sort_by_dim)
    names = session.get("names", [])
    paths = session["paths"]
    start = page * IMAGES_PER_PAGE
    page_indices = indices[start:start + IMAGES_PER_PAGE]
    result = []
    for i in page_indices:
        name = names[i] if i < len(names) else ""
        roi = extract_roi(name)
        caption = f"ROI {roi}" if roi else None
        result.append((paths[i], caption))
    return result


def page_count(session):
    n = len(session["paths"]) if session else 0
    return max(1, -(-n // IMAGES_PER_PAGE))


def page_info_text(session, page):
    n = len(session["paths"]) if session else 0
    if n == 0:
        return "No images uploaded"
    pages = page_count(session)
    return f"Page {page + 1} of {pages} ({n} images)"
