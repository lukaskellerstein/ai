import logging


from PyQt5.QtCore import QModelIndex, Qt, QPoint
from PyQt5.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem
from PyQt5.QtGui import (
    QPainter,
    QBrush,
    QColor,
    QFont,
    QImage,
)

from helpers import getColorByYieldValue


from typing import Any

# create logger
log = logging.getLogger("CellarLogger")

# HELPER methods
def drawText(
    painter: QPainter, options: QStyleOptionViewItem, value: QModelIndex
) -> Any:
    return painter.drawText(options.rect, Qt.AlignHCenter, value)


class MyRenderDelegate(QStyledItemDelegate):
    def __init__(self, view, parent=None):
        super(MyRenderDelegate, self).__init__(parent)
        log.debug("----INIT----")

    def paint(
        self,
        painter: QPainter,
        options: QStyleOptionViewItem,
        index: QModelIndex,
    ):

        # super(MyRenderDelegate, self).paint(painter, options, index)

        # log.debug("Running...")
        # log.debug(locals())

        # painter.setRenderHint(QtGui.QPainter.Antialiasing)
        # painter.setPen(QtGui.QColor(255, 255, 255))
        # painter.setBrush(QtGui.QColor(10, 10, 10))
        # painter.drawRect(options.rect)

        rowIndex: int = index.row()
        columnIndex: int = index.column()

        # ----------------------------------------------------------
        # use Custom painter
        # ----------------------------------------------------------
        if columnIndex == 7:  # last
            value: Any = index.data()
            painter.save()
            painter.fillRect(options.rect, QBrush(QColor("blue")))
            painter.setPen(QColor("yellow"))
            painter.setFont(QFont("Bold"))
            drawText(painter, options, value)
            painter.restore()

            # self.initStyleOption(options, index)
            # widget = options.widget
            # # print(type(widget))

            # style = (
            #     widget.style() if widget is not None else QtWidgets.QApplication.style()
            # )

            # if widget is not None:
            #     print("widget styles")
            #     print(widget.style())
            #     print(widget)
            #     print(widget.styleSheet())
            # else:
            #     print("app styles")
            #     print(QtWidgets.QApplication.style())

            # VARIANT 0
            # drawText(painter, options, value)

            # VARIANT 1
            # style.drawControl(
            #     QtWidgets.QStyle.CE_ItemViewItem, options, painter, widget
            # )
            # painter.restore()

            # VARIANT 2 - Custom drawControl
            # self.drawControl(options, painter, widget, style)  # <-------------

            # VARIANT 3 - Custom drawing with respect margin
            # proxy = style.proxy()
            # textRect = proxy.subElementRect(
            #     QtWidgets.QStyle.SE_ItemViewItemText, options, widget
            # )
            # print(textRect)

            # textMargin = proxy.pixelMetric(
            #     QtWidgets.QStyle.PM_FocusFrameHMargin, None, widget
            # )
            # print(textMargin)
            # textRectWithMargin = options.rect.adjusted(textMargin, 0, -textMargin, 0)
            # print(textRectWithMargin)

            # textSpacing = proxy.pixelMetric(
            #     QtWidgets.QStyle.PM_DefaultLayoutSpacing, None, widget
            # )
            # print(textSpacing)

            # textMargin2 = style.pixelMetric(
            #     QtWidgets.QStyle.PM_FocusFrameHMargin, None, widget
            # )
            # print(textMargin2)
            # textSpacing2 = style.pixelMetric(
            #     QtWidgets.QStyle.PM_DefaultLayoutSpacing, None, widget
            # )
            # print(textSpacing2)

            # painter.drawText(textRectWithMargin, options.displayAlignment, value)

        elif columnIndex == 14:  # change
            value = float(index.data())
            color = getColorByYieldValue(value)
            painter.save()
            painter.fillRect(options.rect, QBrush(QColor(color)))
            drawText(painter, options, f"{value:.1f} %")
            painter.restore()
        elif columnIndex == 19:  # delete button
            node: Any = index.internalPointer()
            if node.childCount() > 0:

                # VARIANT 1
                image: QImage = QImage(":/assets/delete-icon")

                width: int = options.rect.width()
                height: int = options.rect.height()

                image2: QImage = image.scaled(
                    width, height, Qt.KeepAspectRatio
                )

                # // Position our pixmap
                x: float = options.rect.center().x() - (
                    image2.rect().width() / 2
                )
                y: float = options.rect.center().y() - (
                    image2.rect().height() / 2
                )

                painter.drawImage(QPoint(x, y), image2)

                # VARIANT 2
                # image = QtGui.QImage(":/assets/delete-icon")
                # pixmap = QtGui.QPixmap.fromImage(image)

                # options.decorationAlignment = Qt.AlignHCenter
                # painter.drawPixmap(options.rect, pixmap)
                # painter.drawPixmap(
                #     options.rect.x() + 3,
                #     options.rect.y() + 3,
                #     options.rect.height() - (2 * 3),
                #     options.rect.height() - (2 * 3),
                #     pixmap,
                # )

                # VARIANT 3
                # self.initStyleOption(options, index)
                # widget = options.widget

                # # print(type(widget))

                # label = QtWidgets.QLabel()
                # pixmap = QtGui.QPixmap(":/assets/delete-icon")
                # label.setPixmap(pixmap)

                # style = (
                #     widget.style()
                #     if widget is not None
                #     else QtWidgets.QApplication.style()
                # )

                # style.drawControl(
                #     QtWidgets.QStyle.CE_ItemViewItem, options, painter, label
                # )
        else:
            # ----------------------------------------------------------
            # ELSE use QSS styling
            # ----------------------------------------------------------

            # ver 1.
            super(MyRenderDelegate, self).paint(painter, options, index)

            # ver 2.
            # self.initStyleOption(options, index)
            # widget = options.widget
            # style = (
            #     widget.style() if widget is not None else QtWidgets.QApplication.style()
            # )

            # if widget is not None:
            #     print("widget styles")
            #     print(widget.style())
            #     print(widget)
            # else:
            #     print("app styles")
            #     print(QtWidgets.QApplication.style())

            # style.drawControl(
            #     QtWidgets.QStyle.CE_ItemViewItem, options, painter, widget
            # )

    # def drawControl(self, options, painter, widget, style):
    #     painter.save()
    #     painter.setClipRect(options.rect)

    #     proxy = style.proxy()

    #     checkRect = proxy.subElementRect(
    #         QStyle.SE_ItemViewItemCheckIndicator, options, widget
    #     )
    #     iconRect = proxy.subElementRect(
    #         QStyle.SE_ItemViewItemDecoration, options, widget
    #     )
    #     textRect = proxy.subElementRect(
    #         QStyle.SE_ItemViewItemText, options, widget
    #     )

    #     # draw the background
    #     proxy.drawPrimitive(
    #         QStyle.PE_PanelItemViewItem, options, painter, widget
    #     )

    #     # draw the text
    #     # VARIANT 1
    #     self.viewItemDrawText(
    #         painter, options, textRect, proxy, widget
    #     )  # <-------------

    #     # VARIANT 2
    #     # textMargin = (
    #     #     proxy.pixelMetric(QtWidgets.QStyle.PM_FocusFrameHMargin, None, widget) + 1
    #     # )
    #     # print(textMargin)
    #     # textRect = options.rect.adjusted(textMargin, 0, -textMargin, 0)

    #     # painter.drawText(textRect, options.displayAlignment, options.text)

    #     painter.restore()

    # def viewItemDrawText(self, painter, options, textRect, proxy, widget):
    #     textMargin = (
    #         proxy.pixelMetric(QStyle.PM_FocusFrameHMargin, None, widget) + 1
    #     )
    #     textRect = options.rect.adjusted(textMargin, 0, -textMargin, 0)

    #     textOption = QTextOption()
    #     textOption.setTextDirection(options.direction)
    #     textOption.setAlignment(
    #         QStyle.visualAlignment(options.direction, options.displayAlignment)
    #     )

    #     # ----------------------------
    #     # call elideText ---- FINISH
    #     # ----------------------------
    #     paintPosition = None
    #     newText = self.calculateElidedText(
    #         options.text,
    #         textOption,
    #         options.font,
    #         textRect,
    #         options.displayAlignment,
    #         options.textElideMode,
    #         0,
    #         True,
    #         paintPosition,
    #     )

    #     # print(paintPosition)
    #     # print(options.text)
    #     # print(newText)

    #     textLayout = QTextLayout(newText, options.font)
    #     textLayout.setTextOption(textOption)
    #     self.viewItemTextLayout(textLayout, textRect.width())  # <-------------
    #     textLayout.draw(painter, paintPosition)

    # def viewItemTextLayout(
    #     self, textLayout, lineWidth, maxHeight=-1, lastVisibleLine=-1
    # ):
    #     height = 0
    #     widthUsed = 0

    #     textLayout.beginLayout()
    #     i = 0
    #     while True:
    #         line = textLayout.createLine()
    #         if line.isValid() == False:
    #             break

    #         line.setLineWidth(lineWidth)
    #         line.setPosition(QPointF(0, height))
    #         height += line.height()
    #         widthUsed = max(widthUsed, line.naturalTextWidth())
    #         # we assume that the height of the next line is the same as the current one
    #         if (
    #             maxHeight > 0
    #             and lastVisibleLine
    #             and height + line.height() > maxHeight
    #         ):
    #             nextLine = textLayout.createLine()
    #             lastVisibleLine = i if nextLine.isValid() else -1
    #             break

    #         i += 1

    #     textLayout.endLayout()
    #     return QSizeF(widthUsed, height)

    # def calculateElidedText(
    #     self,
    #     text: str,
    #     textOption: QTextOption,
    #     font: QFont,
    #     textRect: QRect,
    #     valign: Qt.Alignment,
    #     textElideMode: Qt.TextElideMode,
    #     flags: int,
    #     lastVisibleLineShouldBeElided: bool,
    #     paintStartPosition: QPointF,
    # ):

    #     textLayout = QTextLayout(text, font)
    #     textLayout.setTextOption(textOption)

    #     # In AlignVCenter mode when more than one line is displayed and the height only allows
    #     # some of the lines it makes no sense to display those. From a users perspective it makes
    #     # more sense to see the start of the text instead something inbetween.
    #     vAlignmentOptimization = paintStartPosition and valign.testFlag(
    #         Qt.AlignVCenter
    #     )

    #     lastVisibleLine = -1
    #     self.viewItemTextLayout(
    #         textLayout,
    #         textRect.width(),
    #         textRect.height() if vAlignmentOptimization else -1,
    #         lastVisibleLine,
    #     )

    #     boundingRect = textLayout.boundingRect()
    #     # don't care about LTR/RTL here, only need the height
    #     layoutRect = QStyle.alignedRect(
    #         Qt.LayoutDirectionAuto,
    #         valign,
    #         boundingRect.size().toSize(),
    #         textRect,
    #     )

    #     if paintStartPosition:
    #         paintStartPosition = QPointF(textRect.x(), layoutRect.top())

    #     height = 0
    #     lineCount = textLayout.lineCount()

    #     for i in range(lineCount):
    #         line = textLayout.lineAt(i)
    #         height += line.height()

    #         # above visible rect
    #         if height + layoutRect.top() <= textRect.top():
    #             if paintStartPosition:
    #                 aaa = paintStartPosition.ry() + line.height()
    #                 # print("0000000000000000000000000000000000")
    #                 # print(aaa)
    #                 # print("0000000000000000000000000000000000")
    #             continue

    #         start = line.textStart()
    #         length = line.textLength()
    #         drawElided = line.naturalTextWidth() > textRect.width()
    #         elideLastVisibleLine = lastVisibleLine == i
    #         if (
    #             drawElided == False
    #             and i + 1 < lineCount
    #             and lastVisibleLineShouldBeElided
    #         ):
    #             nextLine = textLayout.lineAt(i + 1)
    #             nextHeight = height + nextLine.height() / 2
    #             # elide when less than the next half line is visible
    #             if (
    #                 nextHeight + layoutRect.top()
    #                 > textRect.height() + textRect.top()
    #             ):
    #                 elideLastVisibleLine = True

    #         text = textLayout.text().mid(start, length)
    #         if drawElided or elideLastVisibleLine:
    #             if elideLastVisibleLine:
    #                 if text.endsWith("/\n"):
    #                     text.chop(1)
    #                 text += QChar(0x2026)
    #             engine = QStackTextEngine(text, font)
    #             ret += engine.elidedText(
    #                 textElideMode, textRect.width(), flags
    #             )

    #             # no newline for the last line (last visible or real)
    #             # sometimes drawElided is true but no eliding is done so the text ends
    #             # with QChar::LineSeparator - don't add another one. This happened with
    #             # arabic text in the testcase for QTBUG-72805
    #             if (i < lineCount - 1) and ret.endsWith(
    #                 "/\n"
    #             ) == False:
    #                 ret += "/\n"
    #         else:
    #             ret += text

    #         # below visible text, can stop
    #         if (height + layoutRect.top() >= textRect.bottom()) and (
    #             lastVisibleLine >= 0 and lastVisibleLine == i
    #         ):
    #             break
