import sys
from PyQt5.QtWidgets import QApplication

from gui.MainWindow import MainWindow
from config.config_logger import logger

# --------------------- Main Function --------------------- #

def main():
    """
    The main function to start the application.
    """
    try: 
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)

if __name__ == "__main__":
    main()
