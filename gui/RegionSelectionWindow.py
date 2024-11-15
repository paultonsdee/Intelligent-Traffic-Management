# --------------------- Region Selection Window --------------------- #

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QMessageBox, QFileDialog
import json
import os
from config.config_logger import logger
from config.Config import Config

class RegionSelectionWindow(QDialog):
    """
    A dialog window for selecting regions on an image.
    """
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Regions")
        self.setGeometry(100, 100, 1280, 720)
        self.image = image
        self.parent = parent  # Reference to MainWindow
        self.setup_ui()
        self.initialize_properties()
        self.selected_region = []  # Current selected points
        self.region1 = []
        self.region2 = []
        self.region3 = []
        self.logger = logger.getChild(self.__class__.__name__)

        # Apply modern stylesheet
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
                color: #000000;
            }
            QPushButton {
                background-color: #E0E0E0;
                border: 1px solid #AAAAAA;
                padding: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #CCCCCC;
            }
        """)

    def setup_ui(self):
        self.layout = QVBoxLayout(self)
        # Canvas to display image
        self.canvas = QLabel(self)
        self.canvas.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.layout.addWidget(self.canvas)
        # Convert image to QImage and display
        height, width, channel = self.image.shape
        bytes_per_line = 3 * width
        qt_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qt_image)
        self.canvas.setPixmap(self.pixmap)
        # Buttons
        button_layout = QHBoxLayout()
        self.btn_save_region1 = QPushButton("Save to Region 1")
        self.btn_save_region1.clicked.connect(self.save_region1)
        button_layout.addWidget(self.btn_save_region1)

        self.btn_save_region2 = QPushButton("Save to Region 2")
        self.btn_save_region2.clicked.connect(self.save_region2)
        button_layout.addWidget(self.btn_save_region2)

        self.btn_save_region3 = QPushButton("Save to Region 3")
        self.btn_save_region3.clicked.connect(self.save_region3)
        button_layout.addWidget(self.btn_save_region3)

        self.btn_load_regions = QPushButton("Load Regions")
        self.btn_load_regions.clicked.connect(self.load_regions)
        button_layout.addWidget(self.btn_load_regions)

        self.btn_fast_load_regions = QPushButton("Fast Load Regions")
        self.btn_fast_load_regions.clicked.connect(lambda: self.load_regions("fast"))
        button_layout.addWidget(self.btn_fast_load_regions)

        self.btn_reset = QPushButton("Reset Selection")
        self.btn_reset.clicked.connect(self.reset_selection)
        button_layout.addWidget(self.btn_reset)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self.undo_last_point)
        button_layout.addWidget(self.btn_undo)

        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self.accept)
        button_layout.addWidget(self.btn_ok)

        self.layout.addLayout(button_layout)

        # Mouse event
        self.canvas.mousePressEvent = self.on_canvas_click

    def initialize_properties(self):
        self.current_polygon = []
        self.polygons = []
        self.max_points = 7  # Requires 7 points per region

    def on_canvas_click(self, event):
        x = event.pos().x()
        y = event.pos().y()
        if len(self.current_polygon) >= self.max_points:
            self.logger.info("Maximum points reached.")
            return

        self.current_polygon.append((x, y))

        self.canvas.setPixmap(self.pixmap.copy())
        self.draw_polygon(self.current_polygon)

        painter = QPainter(self.canvas.pixmap())
        painter.setPen(QPen(Qt.red, 5))
        for x, y in self.current_polygon:
            painter.drawEllipse(x - 3, y - 3, 6, 6)
        painter.end()
        self.canvas.update()

        self.logger.info(f"Selected {len(self.current_polygon)} points. Point selected: ({x}, {y})")
        self.logger.info(f"Region selected: {self.current_polygon}")

    def undo_last_point(self):
        if self.current_polygon:
            self.current_polygon.pop()
            self.canvas.setPixmap(self.pixmap.copy())
            self.draw_polygon(self.current_polygon)
            painter = QPainter(self.canvas.pixmap())
            painter.setPen(QPen(Qt.red, 5))
            for x, y in self.current_polygon:
                painter.drawEllipse(x - 3, y - 3, 6, 6)
            painter.end()
            self.canvas.update()
        else:
            self.logger.info("No points to undo.")

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_Z:
            self.undo_last_point()

    def draw_polygon(self, polygon):
        painter = QPainter(self.canvas.pixmap())
        painter.setPen(QPen(Qt.blue, 2))
        for i in range(len(polygon)):
            p1 = QPoint(polygon[i][0], polygon[i][1])
            p2 = QPoint(polygon[(i + 1) % len(polygon)][0], polygon[(i + 1) % len(polygon)][1])
            painter.drawLine(p1, p2)
        painter.end()
        self.canvas.update()

    def save_region1(self):
        self.save_region(self.region1, "Region 1")

    def save_region2(self):
        self.save_region(self.region2, "Region 2")

    def save_region3(self):
        self.save_region(self.region3, "Region 3")

    def save_region(self, region, region_name):
        region.extend(self.current_polygon)
        self.save_regions_to_file()
        QMessageBox.information(self, "Saved", f"{region_name} saved.")
        self.logger.info(f"{region_name} saved with points: {self.current_polygon}")
        self.reset_selection()

    def save_regions_to_file(self):
        data = {
            "region1": self.region1,
            "region2": self.region2,
            "region3": self.region3
        }
        try:
            with open(Config.REGIONS_DEFAULT_PATH, "w") as f:
                json.dump(data, f, indent=4)
            self.logger.info("Regions saved to regions.json")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Cannot save regions: {e}")
            self.logger.error(f"Failed to save regions: {e}")

    def load_regions(self, mode="default"):
        if mode == "fast":
            file_path = Config.REGIONS_DEFAULT_PATH
            if not os.path.exists(file_path):
                QMessageBox.warning(self, "Error", "Regions file not found.")
                self.logger.warning("Regions file not found.")
                return
        else:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Load Bounding Boxes",
                "assets",
                "JSON Files (*.json);;All Files (*)",
                options=options
            )
        if file_path:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                self.region1 = data.get("region1", [])
                self.region2 = data.get("region2", [])
                self.region3 = data.get("region3", [])
                if not (self.region1 and self.region2 and self.region3):
                    QMessageBox.warning(self, "Error", "File does not contain all regions.")
                    self.logger.warning("Loaded regions are incomplete.")
                    return
                else:
                    self.draw_loaded_regions()
                    if not mode == "fast":
                        QMessageBox.information(self, "Loaded", "Bounding boxes loaded from file.")
                    self.logger.info("Bounding boxes loaded from file.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Cannot load bounding boxes: {e}")
                self.logger.error(f"Failed to load bounding boxes: {e}")

    def draw_loaded_regions(self):
        self.canvas.setPixmap(self.pixmap.copy())
        for region in [self.region1, self.region2, self.region3]:
            if region:
                self.draw_polygon(region)

    def reset_selection(self):
        self.current_polygon.clear()
        self.canvas.setPixmap(self.pixmap.copy())
        self.canvas.update()
        self.logger.info("Selection reset.")

    def accept(self):
        if not self.region1 or not self.region2 or not self.region3:
            QMessageBox.warning(self, "Error", "Not all regions have been selected.")
            return
        super().accept()
