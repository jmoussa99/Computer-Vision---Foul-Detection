import sys
from pathlib import Path

# Make the repository root importable before loading the UI module.
ROOT = Path(__file__).resolve().parents[1]
UI_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from PyQt5.QtWidgets import QApplication

from video_window_v2 import VideoWindowV2


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoWindowV2()
    window.showMaximized()
    sys.exit(app.exec_())
