import sys
from PyQt5 import QtWidgets 
from main_window import MainWindow

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    display_image_widget = MainWindow()
    display_image_widget.show()
    sys.exit(app.exec_())
