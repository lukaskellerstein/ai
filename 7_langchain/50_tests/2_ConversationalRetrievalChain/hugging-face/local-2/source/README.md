
# Status


| Feature  | Description  |
|---|---|
|  Table View | Stocks watchlist  |
|  Tree View | Futures watchlist  |
|  Stylesheets | Add stylesheets definitions  |
|  Added TreeView delegates | 1. to render custom widgets in cells. 2. Only way howto override existing stylesheets (BackgroundRole doesn't work) |



# Qt

# Model/View 

QAbstractTableModel <- QStandardItemModel

https://doc.qt.io/qtforpython/PySide2/QtCore/QAbstractItemModel.html
https://doc.qt.io/qtforpython/PySide2/QtGui/QStandardItemModel.html


# Styling

1. Define styles in stylesheets
2. Define `QStyledItemDelegate` in *.py file
 
QAbstractItemDelegate <- QStyledItemDelegate

https://doc.qt.io/qtforpython/PySide2/QtWidgets/QAbstractItemDelegate.html
https://doc.qt.io/qtforpython/PySide2/QtWidgets/QStyledItemDelegate.html

# Resources

Command `pyrcc5 -o resources.py resources.qrc`


# Others

Icons:
https://www.pngrepo.com/collection/handy-icon-collection/

Colors:
https://material.io/resources/color/#!
