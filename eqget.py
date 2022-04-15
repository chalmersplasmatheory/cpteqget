#!/usr/bin/env python3

from PyQt5 import QtWidgets
from pathlib import Path
import sys

pt = str((Path(__file__).parent / "..").resolve().absolute())
sys.path.append(pt)

from cpteqget import MainWindow


def main(argv):
    app = QtWidgets.QApplication(sys.argv)

    win = MainWindow(argv)
    win.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))


