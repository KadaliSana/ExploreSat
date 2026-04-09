"""Empty conftest – keeps pytest happy with the src/ layout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
