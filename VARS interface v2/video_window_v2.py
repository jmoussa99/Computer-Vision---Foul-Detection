from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.nn as nn

# Reuse the existing model/config from the original UI package.
ROOT = Path(__file__).resolve().parents[1]
LEGACY_UI_DIR = ROOT / "VARS interface"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LEGACY_UI_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_UI_DIR))

from PyQt5 import QtCore
from PyQt5.QtCore import QDir, QUrl, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFont, QKeySequence, QPixmap
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QShortcut,
    QSlider,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QComboBox,
)
from torchvision.io.video import read_video
from torchvision.models.video import MViT_V2_S_Weights
from types import SimpleNamespace

from cv_foul_detection.bodypart import assign_bodypart
from cv_foul_detection.features import ClipFeatureExtractor, FeatureConfig
from scripts.recognize_foul_bodypart import _locate_contact, _draw_overlay, _context_text

from interface.config.classes import (
    INVERSE_EVENT_DICTIONARY_action_class,
    INVERSE_EVENT_DICTIONARY_offence_severity_class,
)
from interface.model import MVNetwork
from cv_foul_detection.pose import COCO_SKELETON, PoseEstimator


class SkeletonRenderWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, input_path: str, output_path: str):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path

    def run(self) -> None:
        try:
            generate_skeleton_video(self.input_path, self.output_path, self.progress.emit)
            self.finished.emit(self.output_path)
        except Exception as exc:  # pragma: no cover - UI safety net
            self.error.emit(str(exc))


def _draw_skeleton(frame, players, kp_threshold: float) -> None:
    for player in players:
        kp = player.keypoints
        for a, b in COCO_SKELETON:
            if kp[a, 2] >= kp_threshold and kp[b, 2] >= kp_threshold:
                pa = (int(kp[a, 0]), int(kp[a, 1]))
                pb = (int(kp[b, 0]), int(kp[b, 1]))
                cv2.line(frame, pa, pb, (0, 215, 255), 2)
        for j in range(kp.shape[0]):
            if kp[j, 2] >= kp_threshold:
                cv2.circle(frame, (int(kp[j, 0]), int(kp[j, 1])), 3, (0, 215, 255), -1)


def generate_skeleton_video(input_path: str, output_path: str, progress_cb=None) -> None:
    last_progress = -1

    def emit_progress(percent: int) -> None:
        nonlocal last_progress
        if progress_cb is None:
            return
        percent = max(0, min(100, int(percent)))
        if percent <= last_progress:
            return
        last_progress = percent
        try:
            progress_cb(percent)
        except Exception:
            pass

    args = SimpleNamespace(
        max_frames=120,
        frame_stride=1,
        resize_width=480,
        min_contact_area=450,
        field_top_ratio=0.35,
        contact_distance_ratio=0.18,
        contact_motion_p95=18.0,
        contact_box_padding=12,
        peak_strategy="contact",

        contact_source="pose",
        contact_box_scale=0.4,
        max_pose_frames=8,
        contact_center_frac=0.6,
        contact_motion_weight=0.6,
        contact_center_weight=0.3,
        require_two_players=True,
        max_contact_distance_ratio=0.35,

        keypoint_score_threshold=2.0,
        person_score_threshold=0.6,

        # unused for pose mode, but _draw_overlay / shared code may expect them
        pre_model="mvit_v2_s",
        saliency_weight=1.0,
        saliency_min=0.15,
        saliency_alpha=0.45,
    )

    extractor = ClipFeatureExtractor(
        FeatureConfig(
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            resize_width=args.resize_width,
            min_contact_area=args.min_contact_area,
            field_top_ratio=args.field_top_ratio,
            contact_distance_ratio=args.contact_distance_ratio,
            contact_motion_p95=args.contact_motion_p95,
            contact_box_padding=args.contact_box_padding,
            peak_strategy=args.peak_strategy,
        )
    )

    emit_progress(5)

    pose = PoseEstimator(
        device=None,
        person_score_threshold=args.person_score_threshold,
        keypoint_score_threshold=args.keypoint_score_threshold,
    )

    emit_progress(18)

    located = _locate_contact(
        Path(input_path),
        extractor,
        pose,
        args,
        prediction={},
    )

    if located is None:
        raise RuntimeError(
            "No clear two-player contact found. Try setting require_two_players=False."
        )

    contact_point = located["contact_point"]
    players = located["players"]

    emit_progress(52)

    assignment = (
        assign_bodypart(
            contact_point,
            players,
            keypoint_score_threshold=args.keypoint_score_threshold,
        )
        if contact_point is not None and players
        else None
    )

    context_text = ""

    from scripts.recognize_foul_bodypart import _write_foul_overlay_video

    emit_progress(65)

    def write_progress(percent: int) -> None:
        # _write_foul_overlay_video reports frame progress in [50, 99].
        # Keep the overall dialog monotonic and reserve the final tick for close.
        scaled = 65 + round((max(50, min(99, percent)) - 50) * (34.0 / 49.0))
        emit_progress(min(scaled, 99))

    _write_foul_overlay_video(
        Path(input_path),
        Path(output_path),
        Path(output_path).with_name(f"{Path(output_path).stem}_box.mp4"),
        extractor,
        pose,
        located,
        assignment,
        context_text,
        args,
        bodypart_deep=None,
        progress_cb=write_progress,
    )

    emit_progress(100)


class VideoWindowV2(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.show_prediction = True
        self.active_view_index = 0
        self.worker: SkeletonRenderWorker | None = None
        self.progress_dialog: QProgressDialog | None = None
        self.render_progress_current = 0
        self.render_progress_target = 0
        self.playback_rate = 0.25

        self._load_model()
        self._build_ui()
        self._connect_signals()

    def _load_model(self) -> None:
        self.model = MVNetwork(net_name="mvit_v2_s", agr_type="attention")
        path = LEGACY_UI_DIR / "interface" / "14_model.pth.tar"
        load = torch.load(str(path).replace("\\", "/"), map_location=torch.device("cpu"))
        self.model.load_state_dict(load["state_dict"])
        self.model.eval()
        self.softmax = nn.Softmax(dim=1)

    def _build_ui(self) -> None:
        self.setWindowTitle("Video Assistant Referee System")
        self.setStyleSheet(
            "QMainWindow { background: #0f1b2d; }\n"
            "QLabel#AppTitle { color: #f4f6ff; font-size: 28px; font-weight: 600; }\n"
            "QLabel#AppSubtitle { color: #9fb3d9; font-size: 12px; }\n"
            "QFrame#Sidebar { background: #13263f; border-radius: 12px; }\n"
            "QFrame#Panel { background: #0f2238; border-radius: 10px; }\n"
            "QLabel { color: #e5edf7; }\n"
            "QPushButton { background: #f0c75e; color: #1d1d1d; border-radius: 6px; padding: 8px 12px; }\n"
            "QPushButton:disabled { background: #6d6d6d; color: #cfcfcf; }\n"
            "QSlider::groove:horizontal { height: 6px; background: #1f3550; border-radius: 3px; }\n"
            "QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: #f0c75e; border-radius: 7px; }\n"
        )

        base_font = QFont("Bahnschrift", 10)
        self.setFont(base_font)

        root = QWidget(self)
        self.setCentralWidget(root)

        # Header
        title = QLabel("VARS Studio")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Multi-view review with prediction + skeleton export")
        subtitle.setObjectName("AppSubtitle")
        header = QVBoxLayout()
        header.addWidget(title)
        header.addWidget(subtitle)

        # Video grid
        self.mediaPlayers = []
        self.videoWidgets = []
        self.frame_duration_ms = 40
        self.files: list[str] = []

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)

        for i in range(4):
            player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
            widget = QVideoWidget()
            player.setVideoOutput(widget)
            self.mediaPlayers.append(player)
            self.videoWidgets.append(widget)
            grid.addWidget(widget, i // 2, i % 2)

        # Transport
        self.playButton = QPushButton("Play")
        self.playButton.setEnabled(False)
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.speedCombo = QComboBox()
        self.speedCombo.setFixedWidth(90)
        self.speedCombo.addItem("1.0x", 1.0)
        self.speedCombo.addItem("0.5x", 0.5)
        self.speedCombo.addItem("0.3x", 0.3)
        self.speedCombo.addItem("0.25x", 0.25)
        self.speedCombo.addItem("0.2x", 0.2)
        self.speedCombo.setCurrentIndex(self.speedCombo.findData(self.playback_rate))
        transport = QHBoxLayout()
        transport.addWidget(self.playButton)
        transport.addWidget(QLabel("Speed"))
        transport.addWidget(self.speedCombo)
        transport.addWidget(self.positionSlider)

        # Status
        self.statusLabel = QLabel("")
        self.statusLabel.setStyleSheet("color: #9fb3d9;")

        # Sidebar
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(12)

        self.openButton = QPushButton("Open files")
        self.renderButton = QPushButton("Output video")
        self.renderButton.setEnabled(False)

        action_panel = self._panel("Actions", [self.openButton, self.renderButton])

        # Predictions
        self.decisionTitle = QLabel("Groundtruth")
        self.decisionTitle.setAlignment(Qt.AlignLeft)
        self.predictionTitle = QLabel("VARS Prediction")
        self.predictionTitle.setAlignment(Qt.AlignLeft)

        self.actionText = QLabel("")
        self.offenceText = QLabel("")
        self.prediction1Text = QLabel("")
        self.prediction2Text = QLabel("")
        self.prediction3Text = QLabel("")
        self.prediction4Text = QLabel("")

        prediction_font = QFont("Bahnschrift", 16)
        prediction_font.setBold(True)
        for label in [
            self.decisionTitle,
            self.predictionTitle,
            self.actionText,
            self.offenceText,
            self.prediction1Text,
            self.prediction2Text,
            self.prediction3Text,
            self.prediction4Text,
        ]:
            label.setFont(prediction_font)

        prediction_panel = self._panel(
            "Predictions",
            [
                self.decisionTitle,
                self.offenceText,
                self.actionText,
                QLabel(""),
                self.predictionTitle,
                self.prediction1Text,
                self.prediction2Text,
                self.prediction3Text,
                self.prediction4Text,
            ],
        )

        # View controls
        self.showVid1 = QPushButton("Show video 1")
        self.showVid2 = QPushButton("Show video 2")
        self.showVid3 = QPushButton("Show video 3")
        self.showVid4 = QPushButton("Show video 4")
        self.showAllVid = QPushButton("Show all videos")

        view_panel = self._panel(
            "Views",
            [self.showVid1, self.showVid2, self.showVid3, self.showVid4, self.showAllVid],
        )

        sidebar_layout.addWidget(action_panel)
        sidebar_layout.addWidget(prediction_panel)
        sidebar_layout.addWidget(view_panel)
        sidebar_layout.addStretch(1)

        # Main layout
        main = QHBoxLayout(root)
        left = QVBoxLayout()
        left.addLayout(header)
        left.addLayout(grid)
        left.addLayout(transport)
        left.addWidget(self.statusLabel)
        main.addLayout(left, 3)
        main.addWidget(sidebar, 1)

        # Logo overlay
        path_image = LEGACY_UI_DIR / "interface" / "vars_logo.png"
        self.logo = QLabel(self)
        self.logo.setGeometry(QtCore.QRect(500, 0, 1000, 900))
        self.logo.setPixmap(QPixmap(str(path_image)))

        for widget in self.videoWidgets:
            widget.hide()

        self.render_progress_timer = QTimer(self)
        self.render_progress_timer.setInterval(40)
        self.render_progress_timer.timeout.connect(self._advance_render_progress)

        self._set_prediction_visibility(False)

    def _panel(self, title: str, widgets: list[QWidget]) -> QFrame:
        panel = QFrame()
        panel.setObjectName("Panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header = QLabel(title)
        header.setStyleSheet("font-weight: 600; color: #f4f6ff;")
        layout.addWidget(header)
        for widget in widgets:
            layout.addWidget(widget)
        return panel

    def _connect_signals(self) -> None:
        self.playButton.clicked.connect(self.play)
        self.openButton.clicked.connect(self.openFile)
        self.renderButton.clicked.connect(self.outputSkeletonVideo)
        self.positionSlider.sliderMoved.connect(self.setPosition)
        self.speedCombo.currentIndexChanged.connect(self._speed_combo_changed)

        QShortcut(QKeySequence("Space"), self).activated.connect(self.play)

        self.showVid1.clicked.connect(self.enlargeV1)
        self.showVid2.clicked.connect(self.enlargeV2)
        self.showVid3.clicked.connect(self.enlargeV3)
        self.showVid4.clicked.connect(self.enlargeV4)
        self.showAllVid.clicked.connect(self.allVideos)

        for player in self.mediaPlayers:
            player.stateChanged.connect(self.mediaStateChanged)
            player.positionChanged.connect(self.positionChanged)
            player.durationChanged.connect(self.durationChanged)
            player.error.connect(self.handleError)
            player.setMuted(True)

    def _set_prediction_visibility(self, visible: bool) -> None:
        self.decisionTitle.setVisible(visible)
        self.predictionTitle.setVisible(visible)
        self.actionText.setVisible(visible)
        self.offenceText.setVisible(visible)
        self.prediction1Text.setVisible(visible)
        self.prediction2Text.setVisible(visible)
        self.prediction3Text.setVisible(visible)
        self.prediction4Text.setVisible(visible)

    def _set_view_buttons_visibility(self, count: int) -> None:
        self.showVid1.setVisible(count >= 2)
        self.showVid2.setVisible(count >= 2)
        self.showVid3.setVisible(count >= 3)
        self.showVid4.setVisible(count >= 4)
        self.showAllVid.setVisible(count >= 2)

    def keyPressEvent(self, event):
        if event.text() == "a" and self.mediaPlayers[0].state() != QMediaPlayer.PlayingState:
            position = self.mediaPlayers[0].position()
            if position > self.frame_duration_ms:
                for player in self.mediaPlayers:
                    player.setPosition(position - self.frame_duration_ms)
                    self.setFocus()

        if event.text() == "d" and self.mediaPlayers[0].state() != QMediaPlayer.PlayingState:
            position = self.mediaPlayers[0].position()
            duration = self.mediaPlayers[0].duration()
            if position < duration - self.frame_duration_ms:
                for player in self.mediaPlayers:
                    player.setPosition(position + self.frame_duration_ms)
                    self.setFocus()

        if event.key() == Qt.Key_F1:
            self._set_playback_rate(1)
        if event.key() == Qt.Key_F2:
            self._set_playback_rate(0.5)
        if event.key() == Qt.Key_F3:
            self._set_playback_rate(0.3)
        if event.key() == Qt.Key_F4:
            self._set_playback_rate(0.25)
        if event.key() == Qt.Key_F5:
            self._set_playback_rate(0.2)

        if event.text() == "s":
            for player in self.mediaPlayers:
                player.setPosition(2500)
                player.play()
                player.setMuted(True)

        if event.text() == "k":
            for player in self.mediaPlayers:
                player.setPosition(3000)

        if event.text() == "o":
            self.openFile()

    def _set_playback_rate(self, rate: float) -> None:
        self.playback_rate = rate
        position = self.mediaPlayers[0].position()
        for player in self.mediaPlayers:
            player.setPlaybackRate(rate)
            player.setPosition(position)
            self.setFocus()
        combo_index = self.speedCombo.findData(rate)
        if combo_index >= 0 and self.speedCombo.currentIndex() != combo_index:
            self.speedCombo.blockSignals(True)
            self.speedCombo.setCurrentIndex(combo_index)
            self.speedCombo.blockSignals(False)

    def _speed_combo_changed(self) -> None:
        self._set_playback_rate(float(self.speedCombo.currentData()))

    def _apply_playback_rate(self) -> None:
        for player in self.mediaPlayers:
            player.setPlaybackRate(self.playback_rate)

    def openFile(self) -> None:
        for widget in self.videoWidgets:
            widget.hide()

        files, _ = QFileDialog.getOpenFileNames(self, "Select up to 4 files", QDir.homePath())
        if not files:
            self.allVideos()
            return

        self.files = files
        self.active_view_index = 0
        self.logo.hide()
        self.statusLabel.setText("")
        self.renderButton.setEnabled(True)

        self._set_prediction_visibility(False)
        self._set_view_buttons_visibility(len(files))

        if self.show_prediction:
            self._run_prediction(files)
            self._set_prediction_visibility(True)

        for widget in self.videoWidgets:
            widget.hide()

        for idx, widget in enumerate(self.videoWidgets):
            if idx < len(files):
                widget.show()

        for player, file_path in zip(self.mediaPlayers, files):
            player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
            player.setMuted(True)
        self._apply_playback_rate()

        self.playButton.setEnabled(True)
        self.setPosition(2500)
        self.play()

    def _run_prediction(self, files: list[str]) -> None:
        factor = (85 - 65) / (((85 - 65) / 25) * 21)
        for num_view in range(len(files)):
            video, _, _ = read_video(files[num_view], output_format="THWC", pts_unit="sec")
            frames = video[65:85, :, :, :]
            final_frames = None
            transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()

            for j in range(len(frames)):
                if j % factor < 1:
                    if final_frames is None:
                        final_frames = frames[j, :, :, :].unsqueeze(0)
                    else:
                        final_frames = torch.cat((final_frames, frames[j, :, :, :].unsqueeze(0)), 0)

            final_frames = final_frames.permute(0, 3, 1, 2)
            final_frames = transforms_model(final_frames)

            if num_view == 0:
                videos = final_frames.unsqueeze(0)
            else:
                final_frames = final_frames.unsqueeze(0)
                videos = torch.cat((videos, final_frames), 0)

        videos = videos.unsqueeze(0)
        pred = self.model(videos)

        pred_action = pred[1].unsqueeze(0)
        prediction = self.softmax(pred_action)
        values, index = torch.topk(prediction, 2)
        self.prediction3Text.setText(
            f"{INVERSE_EVENT_DICTIONARY_action_class[index[0][0].item()]}: {values[0][0].item():.2f}"
        )
        self.prediction4Text.setText(
            f"{INVERSE_EVENT_DICTIONARY_action_class[index[0][1].item()]}: {values[0][1].item():.2f}"
        )

        pred_offence = pred[0].unsqueeze(0)
        prediction = self.softmax(pred_offence)
        values, index = torch.topk(prediction, 2)
        self.prediction1Text.setText(
            f"{INVERSE_EVENT_DICTIONARY_offence_severity_class[index[0][0].item()]}: {values[0][0].item():.2f}"
        )
        self.prediction2Text.setText(
            f"{INVERSE_EVENT_DICTIONARY_offence_severity_class[index[0][1].item()]}: {values[0][1].item():.2f}"
        )

        path1 = files[0].rsplit("/", 1)[0]
        index_value = ""
        for i in range(1, 5):
            val = path1[-i]
            if val == "_":
                break
            index_value += val
        index_value = index_value[::-1]

        path = path1.rsplit("/", 1)[0]
        annotations_path = os.path.join(path, "annotations.json")
        if os.path.exists(annotations_path):
            with open(annotations_path, "r", encoding="utf-8") as json_file:
                data_json = json.load(json_file)
            self.actionText.setText(data_json["Actions"][index_value]["Action class"])
            severity = data_json["Actions"][index_value]["Severity"]
            severity_text = {
                "1.0": "+ No card",
                "2.0": "+ Borderline NC/YC",
                "3.0": "+ Yellow card",
                "4.0": "+ Borderline YC/RC",
                "5.0": "+ Red card",
            }.get(severity, "")
            offence_severity_text = data_json["Actions"][index_value]["Offence"] + severity_text
            self.offenceText.setText(offence_severity_text)

    def play(self) -> None:
        self._apply_playback_rate()
        for player in self.mediaPlayers:
            if player.state() == QMediaPlayer.PlayingState:
                player.pause()
            else:
                player.play()

    def mediaStateChanged(self, state) -> None:
        if self.mediaPlayers[0].state() == QMediaPlayer.PlayingState:
            self.playButton.setText("Pause")
        else:
            self.playButton.setText("Play")

    def positionChanged(self, position) -> None:
        self.positionSlider.setValue(position)

    def durationChanged(self, duration) -> None:
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position) -> None:
        for player in self.mediaPlayers:
            player.setPosition(position)

    def handleError(self) -> None:
        self.playButton.setEnabled(False)
        QMessageBox.warning(self, "Playback error", self.mediaPlayers[0].errorString())

    def _set_active_view(self, index: int) -> None:
        self.active_view_index = index

    def enlargeV1(self) -> None:
        self._show_single_view(0)

    def enlargeV2(self) -> None:
        self._show_single_view(1)

    def enlargeV3(self) -> None:
        self._show_single_view(2)

    def enlargeV4(self) -> None:
        self._show_single_view(3)

    def _show_single_view(self, index: int) -> None:
        for widget in self.videoWidgets:
            widget.hide()
        self._set_active_view(index)
        if index < len(self.files):
            self.videoWidgets[index].show()
        for player, file_path in zip(self.mediaPlayers, self.files):
            player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
            player.setMuted(True)
        self._apply_playback_rate()
        self.playButton.setEnabled(True)
        self.setPosition(2500)
        self.play()

    def allVideos(self) -> None:
        for idx, widget in enumerate(self.videoWidgets):
            if idx < len(self.files):
                widget.show()
            else:
                widget.hide()
        self._set_active_view(0)
        for player, file_path in zip(self.mediaPlayers, self.files):
            player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
            player.setMuted(True)
        self._apply_playback_rate()
        if self.files:
            self.playButton.setEnabled(True)
            self.setPosition(2500)
            self.play()

    def outputSkeletonVideo(self) -> None:
        if not self.files:
            QMessageBox.information(self, "No video", "Open a video first.")
            return

        source_path = self.files[min(self.active_view_index, len(self.files) - 1)]
        base = os.path.splitext(os.path.basename(source_path))[0]
        default_name = f"{base}_skeleton.mp4"
        initial_dir = os.path.dirname(source_path)
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save skeleton video",
            os.path.join(initial_dir, default_name),
            "MP4 Files (*.mp4)",
        )
        if not output_path:
            return

        self.renderButton.setEnabled(False)
        self.statusLabel.setText("Rendering skeleton overlay...")
        self.worker = SkeletonRenderWorker(source_path, output_path)
        self.progress_dialog = QProgressDialog("Rendering skeleton video...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setValue(0)
        self.render_progress_current = 0
        self.render_progress_target = 0
        self.render_progress_timer.start()
        self.progress_dialog.canceled.connect(self._cancel_render)
        self.worker.finished.connect(self._render_finished)
        self.worker.error.connect(self._render_failed)
        self.worker.progress.connect(self._update_render_progress)
        self.worker.start()

    def _render_finished(self, output_path: str) -> None:
        self.renderButton.setEnabled(True)
        self.statusLabel.setText(f"Saved: {output_path}")
        if self.progress_dialog:
            self.render_progress_timer.stop()
            self.progress_dialog.setValue(100)
            self.progress_dialog.close()
        msg = QMessageBox(self)
        msg.setWindowTitle("Skeleton video saved")
        msg.setText("Saving complete. Do you want to play the video?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)
        msg.setStyleSheet("QLabel{color:#000000;} QPushButton{color:#000000;}")
        reply = msg.exec_()
        if reply == QMessageBox.Yes:
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_path))

    def _render_failed(self, message: str) -> None:
        self.renderButton.setEnabled(True)
        self.statusLabel.setText("Render failed")
        if self.progress_dialog:
            self.render_progress_timer.stop()
            self.progress_dialog.close()
        QMessageBox.warning(self, "Render failed", message)

    def _update_render_progress(self, percent: int) -> None:
        self.render_progress_target = max(self.render_progress_target, min(100, int(percent)))

    def _advance_render_progress(self) -> None:
        if self.progress_dialog:
            if self.render_progress_current < self.render_progress_target:
                gap = self.render_progress_target - self.render_progress_current
                step = max(1, min(4, gap // 3 or 1))
                self.render_progress_current += step
                if self.render_progress_current > self.render_progress_target:
                    self.render_progress_current = self.render_progress_target
                self.progress_dialog.setValue(self.render_progress_current)

    def _cancel_render(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
        self.renderButton.setEnabled(True)
        self.statusLabel.setText("Render canceled")
        if self.progress_dialog:
            self.render_progress_timer.stop()
            self.progress_dialog.close()
