from PyQt6.QtWidgets import (
    QWidget,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QFileDialog,
)
from .FacialEmotionsDetector import FacialEmotionsDetector
from PyQt6.QtCore import Qt, QDir
from typing import Tuple
import platform
from pathlib import PureWindowsPath, PurePosixPath



class MainWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emotions_detector = FacialEmotionsDetector()
        self.resize(300, 100)
        self.setWindowTitle("FactialEmotionDetector")
        self.main_layout = QHBoxLayout()
        self.setLayout(self.main_layout)
        self.menu_widget = QWidget()
        menu_layout = QVBoxLayout()
        self.menu_widget.setLayout(menu_layout)
        menu_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.webcam_btn = QPushButton("Detect emotions from your webcam")
        self.video_btn = QPushButton("Detect emotions in a video")
        self.photo_btn = QPushButton("Detect emotions on a photo")

        self.webcam_btn.clicked.connect(slot=self.connect_to_webcam)
        self.video_btn.clicked.connect(self.process_video)
        self.photo_btn.clicked.connect(self.process_photo)

        menu_layout.addWidget(self.webcam_btn)
        menu_layout.addWidget(self.video_btn)
        menu_layout.addWidget(self.photo_btn)

        self.main_layout.addWidget(self.menu_widget)
        self.show()

    def connect_to_webcam(self):
        code, message = self.emotions_detector.detect_emotions('webcam')
        if code != 0:
            self.show_message('Error', message)

    def process_photo(self):
        code, photo_path = self.open_file_dialog()
        if code == 0:
            code, message = self.emotions_detector.detect_emotions_on_photo(photo_path)
            if code != 0:
                self.show_message('Error', message)

    def open_file_dialog(self) -> Tuple[int, str]:
        # Arguments: parent_widget, dialog_title, initial_directory, file_filters
        dialog = QFileDialog()
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        file_path, _ = dialog.getOpenFileName(
            self,
            "Open File",
            "",  # Empty string for initial directory defaults to last used or user's home
            "All Files (*);;Text Files (*.txt);;Python Files (*.py)"
        )
        if file_path:
            return 0, file_path
        else:
            return -1, None

    def process_video(self):
        code, video_path = self.open_file_dialog()
        if code == 0:
            code, message = self.emotions_detector.detect_emotions(video_path)
            if code != 0:
                self.show_message('Error', message)

    def show_message(self, title: str, msg: str) -> None:
        dlg = QMessageBox(self)
        dlg.setWindowTitle(title)
        dlg.setText(msg)
        dlg.exec()

