from pathlib import Path
import sys

_APP_DIR = Path(__file__).resolve().parent.parent / "app"
_APP_MYSITE_DIR = _APP_DIR / "mysite"

# Vercel imports mysite/asgi.py from the repo root, while the Django app lives in app/.
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

if _APP_MYSITE_DIR.exists():
    __path__.append(str(_APP_MYSITE_DIR))
