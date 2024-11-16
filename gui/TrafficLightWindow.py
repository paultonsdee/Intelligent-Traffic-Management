# --------------------- Traffic Light Window --------------------- #

from PyQt5.QtCore import pyqtSignal, QTimer, Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QCheckBox
from config.config_logger import logger

class TrafficLightWindow(QWidget):
    """
    A window that simulates a traffic light with countdown and a "No Parking" option.
    Emits signals when the state changes.
    """
    state_changed = pyqtSignal(str)  # Emits 'green', 'yellow', 'red'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Traffic Light")
        self.setGeometry(50, 50, 100, 400)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.logger = logger.getChild(self.__class__.__name__)
        self.setup_ui()
        self.setup_timer()
        self.current_state = 'green'
        self.set_state('green')  # Initialize with green
        self.logger.info("TrafficLightWindow initialized.")

        # Apply modern stylesheet
        self.setStyleSheet("""
            QWidget {
                background-color: #FFFFFF;
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
            QLabel {
                color: #000000;
            }
            QCheckBox {
                color: #000000;
            }
        """)

    def setup_ui(self):
        layout = QVBoxLayout()

        # Green light button
        self.btn_green = QPushButton()
        self.btn_green.setFixedSize(50, 50)
        self.btn_green.setStyleSheet("background-color: green; border-radius: 25px;")
        self.btn_green.clicked.connect(lambda: self.set_state('green'))
        layout.addWidget(self.btn_green, alignment=Qt.AlignCenter)

        # Yellow light button
        self.btn_yellow = QPushButton()
        self.btn_yellow.setFixedSize(50, 50)
        self.btn_yellow.setStyleSheet("background-color: grey; border-radius: 25px;")
        self.btn_yellow.clicked.connect(lambda: self.set_state('yellow'))
        layout.addWidget(self.btn_yellow, alignment=Qt.AlignCenter)

        # Red light button
        self.btn_red = QPushButton()
        self.btn_red.setFixedSize(50, 50)
        self.btn_red.setStyleSheet("background-color: grey; border-radius: 25px;")
        self.btn_red.clicked.connect(lambda: self.set_state('red'))
        layout.addWidget(self.btn_red, alignment=Qt.AlignCenter)

        # Countdown label
        self.lbl_countdown = QLabel("30")
        self.lbl_countdown.setAlignment(Qt.AlignCenter)
        self.lbl_countdown.setStyleSheet("font-size: 24px; color: #000000;")
        layout.addWidget(self.lbl_countdown, alignment=Qt.AlignCenter)

        # "No Parking" checkbox
        self.checkbox_no_parking = QCheckBox("No Parking")
        layout.addWidget(self.checkbox_no_parking, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def setup_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_countdown)
        self.time_left = 30  # Initial time for green light
        self.timer.start(1000)  # Update every second

    def update_countdown(self):
        self.time_left -= 1
        self.lbl_countdown.setText(str(self.time_left))
        if self.time_left <= 0:
            self.transition_state()

    def transition_state(self):
        if self.current_state == 'green':
            self.set_state('yellow')
            self.time_left = 5
        elif self.current_state == 'yellow':
            self.set_state('red')
            self.time_left = 35
        elif self.current_state == 'red':
            self.set_state('green')
            self.time_left = 30

    def set_state(self, state):
        self.current_state = state
        
        # Reset light colors
        self.btn_green.setStyleSheet("background-color: grey; border-radius: 25px;")
        self.btn_yellow.setStyleSheet("background-color: grey; border-radius: 25px;")
        self.btn_red.setStyleSheet("background-color: grey; border-radius: 25px;")
        # Set current light color
        if state == 'green':
            self.btn_green.setStyleSheet("background-color: green; border-radius: 25px;")
            self.time_left = 30
        elif state == 'yellow':
            self.btn_yellow.setStyleSheet("background-color: yellow; border-radius: 25px;")
            self.time_left = 5
        elif state == 'red':
            self.btn_red.setStyleSheet("background-color: red; border-radius: 25px;")
            self.time_left = 35
        self.state_changed.emit(state)
        self.logger.info(f"Traffic light state changed to {state}")

    def is_no_parking_checked(self):
        return self.checkbox_no_parking.isChecked()
