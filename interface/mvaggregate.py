from pathlib import Path
import importlib.util

_LEGACY = Path(__file__).resolve().parents[1] / "VARS interface" / "interface" / "mvaggregate.py"
_spec = importlib.util.spec_from_file_location("_legacy_interface_mvaggregate", _LEGACY)
_module = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_module)

for name in getattr(_module, "__all__", None) or [n for n in dir(_module) if not n.startswith("_")]:
    globals()[name] = getattr(_module, name)
