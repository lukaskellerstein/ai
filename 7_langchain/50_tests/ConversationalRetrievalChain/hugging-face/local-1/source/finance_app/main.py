import logging
import logging.config
import sys

from PyQt5.QtWidgets import QApplication

from ui.windows.main.main_window import MainWindow

# set logging from config file
logging.config.fileConfig("logging.conf")

# create logger
log = logging.getLogger("CellarLogger")


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        app.setStartDragDistance(1)  # pixels
        app.setStartDragTime(1)  # ms

        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        log.fatal(e)
    except:
        log.fatal("Something else went wrong")
