import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QGroupBox, QScrollArea, QSplitter, QTabWidget,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QStatusBar,
    QAction, QMenuBar, QToolBar, QMessageBox, QDialog, QTextEdit,
    QDialogButtonBox, QFileDialog, QDockWidget, QGridLayout, QFrame,
    QProgressDialog
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QPoint, QRect, QObject, QThread
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor

from utils.file_manager import FileManager
from utils.dicom_reader import DicomReader
from processors.enhancer import Enhancer
from processors.analyzer import Analyzer
from processors.transformer import Transformer


class ImageCanvas(QLabel):
    point_clicked = pyqtSignal(int, int)
    roi_selected = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_image = None
        self.display_image = None
        self.scale_factor = 1.0
        self.mode = 'view'
        self.points = []
        self.roi_start = None
        self.roi_end = None
        self.drawing = False
        # 标注相关
        self.annotations = []  # [(type, color, data, text), ...]
        self.annot_start = None
        self.annot_type = 'line'  # 'line', 'rect', 'text'
        self.annot_color = QColor(255, 0, 0)
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { background-color: #1e1e1e; border: 1px solid #444; }")
        self.setMouseTracking(True)

    def set_image(self, image):
        if image is None:
            self.clear()
            self.original_image = None
            self.display_image = None
            return
        self.original_image = image.copy()
        self.display_image = image.copy()
        self.points = []
        self.annotations = []
        self._update_display()

    def _update_display(self):
        if self.display_image is None:
            return
        image = self.display_image
        if len(image.shape) == 2:
            h, w = image.shape
            bytes_per_line = w
            q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
        else:
            h, w, ch = image.shape
            bytes_per_line = ch * w
            q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.scale_factor = scaled.width() / pixmap.width() if pixmap.width() > 0 else 1.0
        self.setPixmap(scaled)

    def _image_to_display(self, img_x, img_y):
        """将图像坐标转换为控件显示坐标"""
        pixmap = self.pixmap()
        if pixmap is None or self.original_image is None:
            return 0, 0
        img_w = self.original_image.shape[1]
        img_h = self.original_image.shape[0]
        display_w = pixmap.width()
        display_h = pixmap.height()
        offset_x = (self.width() - display_w) / 2
        offset_y = (self.height() - display_h) / 2
        dx = offset_x + img_x / img_w * display_w
        dy = offset_y + img_y / img_h * display_h
        return dx, dy

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.original_image is None or self.pixmap() is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制测量点
        if self.mode in ('measure_distance', 'measure_angle', 'region_growing') and self.points:
            colors = [
                ('#ff4444', '#ff6666'),
                ('#44ff44', '#66ff66'),
                ('#4488ff', '#66aaff'),
            ]
            for i, (ix, iy) in enumerate(self.points):
                dx, dy = self._image_to_display(ix, iy)
                ci = i % 3
                pen = QPen(QColor(colors[ci][0]), 2)
                painter.setPen(pen)
                painter.setBrush(QColor(colors[ci][1]))
                painter.drawEllipse(QPoint(int(dx), int(dy)), 6, 6)
                # 标注序号
                painter.setPen(QPen(QColor('#ffffff'), 1))
                font = QFont('Arial', 10, QFont.Bold)
                painter.setFont(font)
                painter.drawText(int(dx) + 10, int(dy) - 10, str(i + 1))

            # 画距离测量连线
            if self.mode == 'measure_distance' and len(self.points) >= 2:
                p1 = self.points[0]
                p2 = self.points[1]
                d1x, d1y = self._image_to_display(p1[0], p1[1])
                d2x, d2y = self._image_to_display(p2[0], p2[1])
                painter.setPen(QPen(QColor('#ffff00'), 2, Qt.DashLine))
                painter.drawLine(int(d1x), int(d1y), int(d2x), int(d2y))

            # 画角度测量连线
            if self.mode == 'measure_angle' and len(self.points) >= 3:
                p0 = self.points[0]
                p1 = self.points[1]  # 顶点
                p2 = self.points[2]
                d0x, d0y = self._image_to_display(p0[0], p0[1])
                d1x, d1y = self._image_to_display(p1[0], p1[1])
                d2x, d2y = self._image_to_display(p2[0], p2[1])
                painter.setPen(QPen(QColor('#ffff00'), 2, Qt.DashLine))
                painter.drawLine(int(d1x), int(d1y), int(d0x), int(d0y))
                painter.drawLine(int(d1x), int(d1y), int(d2x), int(d2y))
                # 画角度弧线
                painter.setPen(QPen(QColor('#ff8800'), 2))
                r = 20
                import math
                a1 = math.atan2(d0y - d1y, d0x - d1x)
                a2 = math.atan2(d2y - d1y, d2x - d1x)
                rect = QRect(int(d1x - r), int(d1y - r), int(r * 2), int(r * 2))
                painter.drawArc(rect, int(-a1 * 180 / math.pi * 16), int((a1 - a2) * 180 / math.pi * 16))

        # 绘制 ROI 拖拽框
        if self.drawing and self.mode == 'roi' and self.roi_start and self.roi_end:
            start = self.roi_start
            end = self.roi_end
            p1x, p1y = self._image_to_display(min(start[0], end[0]), min(start[1], end[1]))
            p2x, p2y = self._image_to_display(max(start[0], end[0]), max(start[1], end[1]))
            painter.setPen(QPen(QColor('#00ff00'), 2, Qt.DashLine))
            painter.setBrush(QColor(0, 255, 0, 30))
            painter.drawRect(QRect(int(p1x), int(p1y), int(p2x - p1x), int(p2y - p1y)))

        # 绘制标注
        for ann in self.annotations:
            ann_type = ann[0]
            color = ann[1]
            if ann_type == 'line':
                _, _, p1, p2 = ann
                d1x, d1y = self._image_to_display(p1[0], p1[1])
                d2x, d2y = self._image_to_display(p2[0], p2[1])
                painter.setPen(QPen(color, 3))
                painter.drawLine(int(d1x), int(d1y), int(d2x), int(d2y))
            elif ann_type == 'rect':
                _, _, p1, p2, txt = ann
                x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
                d1x, d1y = self._image_to_display(x1, y1)
                d2x, d2y = self._image_to_display(x2, y2)
                painter.setPen(QPen(color, 2))
                painter.setBrush(QColor(color.red(), color.green(), color.blue(), 20))
                painter.drawRect(QRect(int(d1x), int(d1y), int(d2x - d1x), int(d2y - d1y)))
                if txt:
                    painter.setPen(QPen(color, 1))
                    painter.setFont(QFont('Arial', 10))
                    painter.drawText(int(d1x) + 4, int(d1y) - 6, txt)
            elif ann_type == 'text':
                _, _, pos, txt = ann
                dx, dy = self._image_to_display(pos[0], pos[1])
                painter.setPen(QPen(color, 1))
                painter.setFont(QFont('Arial', 12, QFont.Bold))
                painter.drawText(int(dx), int(dy), txt)

        # 正在进行的标注(拖拽中)
        if self.drawing and self.mode == 'annotate' and self.annot_start and self.roi_end:
            p1 = self.annot_start
            p2 = self.roi_end
            d1x, d1y = self._image_to_display(p1[0], p1[1])
            d2x, d2y = self._image_to_display(p2[0], p2[1])
            painter.setPen(QPen(self.annot_color, 2, Qt.DashLine))
            if self.annot_type == 'line':
                painter.drawLine(int(d1x), int(d1y), int(d2x), int(d2y))
            elif self.annot_type == 'rect':
                x1, y1 = min(d1x, d2x), min(d1y, d2y)
                x2, y2 = max(d1x, d2x), max(d1y, d2y)
                painter.setBrush(QColor(self.annot_color.red(), self.annot_color.green(),
                                         self.annot_color.blue(), 20))
                painter.drawRect(QRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def set_mode(self, mode):
        self.mode = mode
        self.points = []
        self.annot_start = None
        self.roi_end = None
        self.drawing = False
        if mode == 'view':
            self.setCursor(Qt.ArrowCursor)
        elif mode == 'measure_distance':
            self.setCursor(Qt.CrossCursor)
        elif mode == 'measure_angle':
            self.setCursor(Qt.CrossCursor)
        elif mode == 'roi':
            self.setCursor(Qt.CrossCursor)
        elif mode == 'region_growing':
            self.setCursor(Qt.CrossCursor)
        elif mode == 'annotate':
            self.setCursor(Qt.CrossCursor)

    def set_annot_type(self, t):
        self.annot_type = t

    def set_annot_color(self, c):
        self.annot_color = c

    def clear_annotations(self):
        self.annotations = []
        self.repaint()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        pos = self._map_to_image(event.pos())
        if pos is None:
            return
        x, y = pos
        if self.mode == 'measure_distance':
            self.points.append((x, y))
            self.repaint()
            if len(self.points) >= 2:
                self.point_clicked.emit(x, y)
        elif self.mode == 'measure_angle':
            self.points.append((x, y))
            self.repaint()
            if len(self.points) >= 3:
                self.point_clicked.emit(x, y)
        elif self.mode == 'region_growing':
            self.points = [(x, y)]
            self.repaint()
            self.point_clicked.emit(x, y)
        elif self.mode == 'roi':
            self.roi_start = (x, y)
            self.roi_end = (x, y)
            self.drawing = True
        elif self.mode == 'annotate':
            if self.annot_type == 'text':
                # 弹出输入框
                from PyQt5.QtWidgets import QInputDialog
                txt, ok = QInputDialog.getText(self, "标注文字", "请输入标注文字:")
                if ok and txt:
                    self.annotations.append(('text', self.annot_color, (x, y), txt))
                    self.repaint()
            else:
                self.annot_start = (x, y)
                self.roi_end = (x, y)
                self.drawing = True

    def mouseMoveEvent(self, event):
        if self.drawing and self.mode in ('roi', 'annotate'):
            pos = self._map_to_image(event.pos())
            if pos:
                self.roi_end = pos
                self.repaint()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        if self.drawing and self.mode == 'roi' and self.roi_start and self.roi_end:
            x1 = min(self.roi_start[0], self.roi_end[0])
            y1 = min(self.roi_start[1], self.roi_end[1])
            x2 = max(self.roi_start[0], self.roi_end[0])
            y2 = max(self.roi_start[1], self.roi_end[1])
            self.roi_selected.emit(x1, y1, x2 - x1, y2 - y1)
            self.drawing = False
            self.roi_start = None
            self.roi_end = None
        elif self.drawing and self.mode == 'annotate' and self.annot_start and self.roi_end:
            p1 = self.annot_start
            p2 = self.roi_end
            if self.annot_type == 'line':
                self.annotations.append(('line', self.annot_color, p1, p2))
            elif self.annot_type == 'rect':
                self.annotations.append(('rect', self.annot_color, p1, p2, ''))
            self.drawing = False
            self.annot_start = None
            self.roi_end = None
            self.repaint()

    def _map_to_image(self, pos):
        if self.original_image is None:
            return None
        pixmap = self.pixmap()
        if pixmap is None:
            return None
        img_w = self.original_image.shape[1]
        img_h = self.original_image.shape[0]
        label_w = self.width()
        label_h = self.height()
        display_w = pixmap.width()
        display_h = pixmap.height()
        offset_x = (label_w - display_w) / 2
        offset_y = (label_h - display_h) / 2
        rel_x = pos.x() - offset_x
        rel_y = pos.y() - offset_y
        if rel_x < 0 or rel_x > display_w or rel_y < 0 or rel_y > display_h:
            return None
        img_x = int(rel_x / display_w * img_w)
        img_y = int(rel_y / display_h * img_h)
        img_x = max(0, min(img_x, img_w - 1))
        img_y = max(0, min(img_y, img_h - 1))
        return (img_x, img_y)


class HistogramWidget(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4, 3), dpi=80)
        self.fig.patch.set_facecolor('#2b2b2b')
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self._style_axis()

    def _style_axis(self):
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        for spine in self.ax.spines.values():
            spine.set_color('#555')

    def plot_histogram(self, image):
        self.ax.clear()
        self._style_axis()
        if image is None:
            self.draw()
            return
        if len(image.shape) == 2:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            self.ax.plot(hist, color='#00ff88', linewidth=1)
            self.ax.fill_between(range(256), hist.flatten(), alpha=0.3, color='#00ff88')
        else:
            colors = {'R': ('#ff4444', 0), 'G': ('#44ff44', 1), 'B': ('#4444ff', 2)}
            for name, (color, idx) in colors.items():
                hist = cv2.calcHist([image], [idx], None, [256], [0, 256])
                self.ax.plot(hist, color=color, linewidth=1, label=name)
                self.ax.fill_between(range(256), hist.flatten(), alpha=0.15, color=color)
            self.ax.legend(fontsize=8, facecolor='#2b2b2b', edgecolor='#555', labelcolor='white')
        self.ax.set_title('直方图', fontsize=10)
        self.ax.set_xlabel('像素值', fontsize=8)
        self.ax.set_ylabel('频数', fontsize=8)
        self.fig.tight_layout()
        self.draw()


class ProcessingWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, operation):
        super().__init__()
        self.operation = operation

    def run(self):
        try:
            result = self.operation()
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("医学图像处理工具 - BME")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(self._get_stylesheet())

        self.file_manager = FileManager()
        self.enhancer = Enhancer()
        self.analyzer = Analyzer()
        self.transformer = Transformer()

        self.current_image = None
        self.processed_image = None
        self.history = []
        self.history_index = -1
        self.processing_thread = None
        self.processing_worker = None
        self.processing_dialog = None
        self.toolbar = None
        self.left_panel = None
        self.right_panel = None
        self.viewer_tabs = None

        self._init_ui()
        self._init_menu()
        self._init_toolbar()
        self._init_statusbar()

    def _get_stylesheet(self):
        return """
            QMainWindow, QWidget {
                background-color: #1f232a;
                color: #e6edf3;
            }
            QGroupBox {
                color: #f0f6fc;
                font-weight: bold;
                border: 1px solid #39414f;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 14px;
                background-color: #242933;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #f0f6fc;
                background-color: #242933;
            }
            QLabel {
                color: #d0d7de;
                background: transparent;
            }
            QPushButton {
                background-color: #2f81f7;
                color: #ffffff;
                border: 1px solid #1f6feb;
                border-radius: 6px;
                padding: 6px 12px;
                min-height: 24px;
                font-weight: 500;
            }
            QPushButton:hover { background-color: #3b8cff; }
            QPushButton:pressed { background-color: #1f6feb; }
            QPushButton:disabled {
                background-color: #30363d;
                color: #7d8590;
                border-color: #30363d;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #39414f;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #58a6ff;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
                background-color: #0d1117;
                color: #e6edf3;
                border: 1px solid #39414f;
                border-radius: 6px;
                padding: 4px 8px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #161b22;
                color: #e6edf3;
                selection-background-color: #2f81f7;
            }
            QTabWidget::pane {
                border: 1px solid #39414f;
                background-color: #161b22;
                border-radius: 8px;
            }
            QTabBar::tab {
                background-color: #242933;
                color: #9da7b3;
                padding: 8px 16px;
                border: 1px solid #39414f;
                border-bottom: none;
                border-radius: 6px 6px 0 0;
            }
            QTabBar::tab:selected {
                background-color: #161b22;
                color: #ffffff;
            }
            QDockWidget { color: #e6edf3; }
            QDockWidget::title {
                background-color: #242933;
                padding: 6px;
            }
            QScrollArea {
                border: none;
                background-color: #1f232a;
            }
            QScrollBar:vertical {
                background: #161b22;
                width: 10px;
                margin: 2px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #39414f;
                min-height: 24px;
                border-radius: 5px;
            }
            QStatusBar {
                background-color: #161b22;
                color: #c9d1d9;
                border-top: 1px solid #30363d;
            }
            QMenuBar {
                background-color: #161b22;
                color: #e6edf3;
            }
            QMenuBar::item {
                background: transparent;
                padding: 6px 10px;
            }
            QMenuBar::item:selected { background-color: #242933; }
            QMenu {
                background-color: #161b22;
                color: #e6edf3;
                border: 1px solid #39414f;
            }
            QMenu::item:selected {
                background-color: #2f81f7;
                color: #ffffff;
            }
            QToolBar {
                background-color: #161b22;
                border: none;
                spacing: 4px;
                padding: 4px;
            }
            QToolBar QToolButton {
                color: #d0d7de;
                padding: 6px 10px;
                border-radius: 6px;
            }
            QToolBar QToolButton:hover {
                background-color: #242933;
            }
            QDialog, QProgressDialog {
                background-color: #1f232a;
                color: #e6edf3;
            }
            QToolTip {
                background-color: #161b22;
                color: #e6edf3;
                border: 1px solid #39414f;
            }
        """

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)

        self.left_panel = self._create_left_panel()
        splitter.addWidget(self.left_panel)

        center_panel = self._create_center_panel()
        splitter.addWidget(center_panel)

        self.right_panel = self._create_right_panel()
        splitter.addWidget(self.right_panel)

        splitter.setSizes([260, 600, 300])
        main_layout.addWidget(splitter)

    def _create_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(240)
        scroll.setMaximumWidth(320)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(6)

        layout.addWidget(self._create_file_group())
        layout.addWidget(self._create_enhance_group())
        layout.addWidget(self._create_filter_group())
        layout.addWidget(self._create_transform_group())
        layout.addWidget(self._create_morphology_group())
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def _create_file_group(self):
        group = QGroupBox("文件操作")
        layout = QVBoxLayout()

        btn_open = QPushButton("打开图像")
        btn_open.clicked.connect(self.open_image)
        layout.addWidget(btn_open)

        btn_save = QPushButton("保存图像")
        btn_save.clicked.connect(self.save_image)
        layout.addWidget(btn_save)

        btn_info = QPushButton("图像信息")
        btn_info.clicked.connect(self.show_image_info)
        layout.addWidget(btn_info)

        btn_dicom = QPushButton("DICOM元数据")
        btn_dicom.clicked.connect(self.show_dicom_metadata)
        layout.addWidget(btn_dicom)

        group.setLayout(layout)
        return group

    def _create_enhance_group(self):
        group = QGroupBox("图像增强")
        layout = QGridLayout()

        layout.addWidget(QLabel("亮度:"), 0, 0)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_label = QLabel("0")
        self.brightness_label.setMinimumWidth(30)
        self.brightness_slider.valueChanged.connect(
            lambda v: self.brightness_label.setText(str(v))
        )
        layout.addWidget(self.brightness_slider, 0, 1)
        layout.addWidget(self.brightness_label, 0, 2)

        layout.addWidget(QLabel("对比度:"), 1, 0)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        self.contrast_label = QLabel("1.0")
        self.contrast_label.setMinimumWidth(30)
        self.contrast_slider.valueChanged.connect(
            lambda v: self.contrast_label.setText(f"{v / 100:.1f}")
        )
        layout.addWidget(self.contrast_slider, 1, 1)
        layout.addWidget(self.contrast_label, 1, 2)

        btn_apply_bc = QPushButton("应用亮度/对比度")
        btn_apply_bc.clicked.connect(self.apply_brightness_contrast)
        layout.addWidget(btn_apply_bc, 2, 0, 1, 3)

        btn_hist_eq = QPushButton("直方图均衡化")
        btn_hist_eq.clicked.connect(self.apply_histogram_eq)
        layout.addWidget(btn_hist_eq, 3, 0, 1, 3)

        layout.addWidget(QLabel("CLAHE裁剪:"), 4, 0)
        self.clahe_spin = QDoubleSpinBox()
        self.clahe_spin.setRange(0.1, 40.0)
        self.clahe_spin.setValue(2.0)
        self.clahe_spin.setSingleStep(0.5)
        layout.addWidget(self.clahe_spin, 4, 1, 1, 2)

        btn_clahe = QPushButton("自适应直方图均衡化(CLAHE)")
        btn_clahe.clicked.connect(self.apply_clahe)
        layout.addWidget(btn_clahe, 5, 0, 1, 3)

        layout.addWidget(QLabel("Gamma:"), 6, 0)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 5.0)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setSingleStep(0.1)
        layout.addWidget(self.gamma_spin, 6, 1, 1, 2)

        btn_gamma = QPushButton("Gamma校正")
        btn_gamma.clicked.connect(self.apply_gamma)
        layout.addWidget(btn_gamma, 7, 0, 1, 3)

        btn_invert = QPushButton("图像反转")
        btn_invert.clicked.connect(self.apply_invert)
        layout.addWidget(btn_invert, 8, 0, 1, 3)

        group.setLayout(layout)
        return group

    def _create_filter_group(self):
        group = QGroupBox("滤波与噪声")
        layout = QGridLayout()

        layout.addWidget(QLabel("滤波类型:"), 0, 0)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["高斯滤波", "中值滤波", "双边滤波"])
        layout.addWidget(self.filter_combo, 0, 1, 1, 2)

        layout.addWidget(QLabel("核大小:"), 1, 0)
        self.filter_ksize_spin = QSpinBox()
        self.filter_ksize_spin.setRange(1, 31)
        self.filter_ksize_spin.setValue(5)
        self.filter_ksize_spin.setSingleStep(2)
        layout.addWidget(self.filter_ksize_spin, 1, 1, 1, 2)

        btn_filter = QPushButton("应用滤波")
        btn_filter.clicked.connect(self.apply_filter)
        layout.addWidget(btn_filter, 2, 0, 1, 3)

        layout.addWidget(QLabel("噪声类型:"), 3, 0)
        self.noise_combo = QComboBox()
        self.noise_combo.addItems(["高斯噪声", "椒盐噪声"])
        layout.addWidget(self.noise_combo, 3, 1, 1, 2)

        layout.addWidget(QLabel("噪声强度:"), 4, 0)
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.1, 100.0)
        self.noise_spin.setValue(25.0)
        self.noise_spin.setSingleStep(5.0)
        layout.addWidget(self.noise_spin, 4, 1, 1, 2)

        btn_noise = QPushButton("添加噪声")
        btn_noise.clicked.connect(self.add_noise)
        layout.addWidget(btn_noise, 5, 0, 1, 3)

        group.setLayout(layout)
        return group

    def _create_transform_group(self):
        group = QGroupBox("几何变换")
        layout = QGridLayout()

        btn_rotate_cw = QPushButton("顺时针90°")
        btn_rotate_cw.clicked.connect(lambda: self.apply_rotate(-90))
        layout.addWidget(btn_rotate_cw, 0, 0)

        btn_rotate_ccw = QPushButton("逆时针90°")
        btn_rotate_ccw.clicked.connect(lambda: self.apply_rotate(90))
        layout.addWidget(btn_rotate_ccw, 0, 1)

        btn_flip_h = QPushButton("水平翻转")
        btn_flip_h.clicked.connect(self.apply_flip_h)
        layout.addWidget(btn_flip_h, 1, 0)

        btn_flip_v = QPushButton("垂直翻转")
        btn_flip_v.clicked.connect(self.apply_flip_v)
        layout.addWidget(btn_flip_v, 1, 1)

        layout.addWidget(QLabel("缩放:"), 2, 0)
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 5.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        layout.addWidget(self.scale_spin, 2, 1)

        btn_resize = QPushButton("缩放")
        btn_resize.clicked.connect(self.apply_resize)
        layout.addWidget(btn_resize, 3, 0, 1, 2)

        group.setLayout(layout)
        return group

    def _create_morphology_group(self):
        group = QGroupBox("形态学操作")
        layout = QGridLayout()

        layout.addWidget(QLabel("核大小:"), 0, 0)
        self.morph_ksize_spin = QSpinBox()
        self.morph_ksize_spin.setRange(1, 21)
        self.morph_ksize_spin.setValue(3)
        self.morph_ksize_spin.setSingleStep(2)
        layout.addWidget(self.morph_ksize_spin, 0, 1)

        btn_erode = QPushButton("腐蚀")
        btn_erode.clicked.connect(self.apply_erode)
        layout.addWidget(btn_erode, 1, 0)

        btn_dilate = QPushButton("膨胀")
        btn_dilate.clicked.connect(self.apply_dilate)
        layout.addWidget(btn_dilate, 1, 1)

        btn_open = QPushButton("开运算")
        btn_open.clicked.connect(self.apply_morph_open)
        layout.addWidget(btn_open, 2, 0)

        btn_close = QPushButton("闭运算")
        btn_close.clicked.connect(self.apply_morph_close)
        layout.addWidget(btn_close, 2, 1)

        group.setLayout(layout)
        return group

    def _create_center_panel(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.viewer_tabs = QTabWidget()

        self.canvas = ImageCanvas()
        self.canvas.point_clicked.connect(self._on_canvas_click)
        self.canvas.roi_selected.connect(self._on_roi_selected)
        self.viewer_tabs.addTab(self.canvas, "图像视图")

        self.histogram_widget = HistogramWidget()
        self.viewer_tabs.addTab(self.histogram_widget, "直方图")

        layout.addWidget(self.viewer_tabs)
        return widget

    def _create_right_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(260)
        scroll.setMaximumWidth(340)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(6)

        layout.addWidget(self._create_edge_group())
        layout.addWidget(self._create_threshold_group())
        layout.addWidget(self._create_measure_group())
        layout.addWidget(self._create_dicom_window_group())
        layout.addWidget(self._create_history_group())
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def _create_edge_group(self):
        group = QGroupBox("边缘检测")
        layout = QGridLayout()

        layout.addWidget(QLabel("算法:"), 0, 0)
        self.edge_combo = QComboBox()
        self.edge_combo.addItems(["Sobel", "Canny", "Laplacian"])
        layout.addWidget(self.edge_combo, 0, 1)

        layout.addWidget(QLabel("阈值1:"), 1, 0)
        self.edge_thresh1_spin = QSpinBox()
        self.edge_thresh1_spin.setRange(0, 500)
        self.edge_thresh1_spin.setValue(100)
        layout.addWidget(self.edge_thresh1_spin, 1, 1)

        layout.addWidget(QLabel("阈值2:"), 2, 0)
        self.edge_thresh2_spin = QSpinBox()
        self.edge_thresh2_spin.setRange(0, 500)
        self.edge_thresh2_spin.setValue(200)
        layout.addWidget(self.edge_thresh2_spin, 2, 1)

        btn_edge = QPushButton("边缘检测")
        btn_edge.clicked.connect(self.apply_edge_detection)
        layout.addWidget(btn_edge, 3, 0, 1, 2)

        group.setLayout(layout)
        return group

    def _create_threshold_group(self):
        group = QGroupBox("阈值分割")
        layout = QGridLayout()

        layout.addWidget(QLabel("方法:"), 0, 0)
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems(["全局阈值", "Otsu自适应", "自适应阈值"])
        layout.addWidget(self.threshold_combo, 0, 1)

        layout.addWidget(QLabel("阈值:"), 1, 0)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(127)
        layout.addWidget(self.threshold_spin, 1, 1)

        btn_threshold = QPushButton("阈值分割")
        btn_threshold.clicked.connect(self.apply_threshold)
        layout.addWidget(btn_threshold, 2, 0, 1, 2)

        btn_contour = QPushButton("轮廓检测")
        btn_contour.clicked.connect(self.apply_contour)
        layout.addWidget(btn_contour, 3, 0, 1, 2)

        layout.addWidget(QLabel("区域生长容差:"), 4, 0)
        self.region_tolerance_spin = QSpinBox()
        self.region_tolerance_spin.setRange(1, 100)
        self.region_tolerance_spin.setValue(10)
        layout.addWidget(self.region_tolerance_spin, 4, 1)

        btn_region = QPushButton("区域生长(点击图像)")
        btn_region.clicked.connect(self.start_region_growing)
        layout.addWidget(btn_region, 5, 0, 1, 2)

        group.setLayout(layout)
        return group

    def _create_measure_group(self):
        group = QGroupBox("测量与标注")
        layout = QVBoxLayout()
        layout.setSpacing(4)

        btn_distance = QPushButton("距离测量(点击两点)")
        btn_distance.clicked.connect(self.start_measure_distance)
        layout.addWidget(btn_distance)

        btn_angle = QPushButton("角度测量(点击三点)")
        btn_angle.clicked.connect(self.start_measure_angle)
        layout.addWidget(btn_angle)

        btn_roi = QPushButton("ROI区域选择(拖拽)")
        btn_roi.clicked.connect(self.start_roi)
        layout.addWidget(btn_roi)

        self.measure_result_label = QLabel("测量结果: 无")
        self.measure_result_label.setWordWrap(True)
        self.measure_result_label.setStyleSheet("color: #00ff88; padding: 4px;")
        layout.addWidget(self.measure_result_label)

        # 标注工具分隔
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #555;")
        layout.addWidget(sep)

        annot_label = QLabel("-- 图像标注 --")
        annot_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(annot_label)

        btn_line = QPushButton("画线标注")
        btn_line.clicked.connect(self.start_annotate_line)
        layout.addWidget(btn_line)

        btn_rect = QPushButton("矩形标注")
        btn_rect.clicked.connect(self.start_annotate_rect)
        layout.addWidget(btn_rect)

        btn_text = QPushButton("文字标注")
        btn_text.clicked.connect(self.start_annotate_text)
        layout.addWidget(btn_text)

        btn_clear_annot = QPushButton("清除标注")
        btn_clear_annot.clicked.connect(self.clear_annotations)
        layout.addWidget(btn_clear_annot)

        group.setLayout(layout)
        return group

    def _create_dicom_window_group(self):
        group = QGroupBox("DICOM窗宽窗位")
        layout = QGridLayout()

        layout.addWidget(QLabel("窗位(Center):"), 0, 0)
        self.wl_center_spin = QDoubleSpinBox()
        self.wl_center_spin.setRange(-5000, 5000)
        self.wl_center_spin.setValue(127.5)
        self.wl_center_spin.setSingleStep(10)
        layout.addWidget(self.wl_center_spin, 0, 1)

        layout.addWidget(QLabel("窗宽(Width):"), 1, 0)
        self.wl_width_spin = QDoubleSpinBox()
        self.wl_width_spin.setRange(1, 10000)
        self.wl_width_spin.setValue(255)
        self.wl_width_spin.setSingleStep(10)
        layout.addWidget(self.wl_width_spin, 1, 1)

        btn_apply_wl = QPushButton("应用窗宽窗位")
        btn_apply_wl.clicked.connect(self.apply_window_level)
        layout.addWidget(btn_apply_wl, 2, 0, 1, 2)

        btn_preset_brain = QPushButton("脑窗")
        btn_preset_brain.clicked.connect(lambda: self._set_dicom_preset(40, 80))
        layout.addWidget(btn_preset_brain, 3, 0)

        btn_preset_lung = QPushButton("肺窗")
        btn_preset_lung.clicked.connect(lambda: self._set_dicom_preset(-600, 1500))
        layout.addWidget(btn_preset_lung, 3, 1)

        btn_preset_bone = QPushButton("骨窗")
        btn_preset_bone.clicked.connect(lambda: self._set_dicom_preset(400, 1800))
        layout.addWidget(btn_preset_bone, 4, 0)

        btn_preset_abdomen = QPushButton("腹部窗")
        btn_preset_abdomen.clicked.connect(lambda: self._set_dicom_preset(40, 400))
        layout.addWidget(btn_preset_abdomen, 4, 1)

        group.setLayout(layout)
        return group

    def _create_history_group(self):
        group = QGroupBox("操作历史")
        layout = QVBoxLayout()

        btn_undo = QPushButton("撤销")
        btn_undo.clicked.connect(self.undo)
        layout.addWidget(btn_undo)

        btn_redo = QPushButton("重做")
        btn_redo.clicked.connect(self.redo)
        layout.addWidget(btn_redo)

        btn_reset = QPushButton("恢复原图")
        btn_reset.clicked.connect(self.reset_image)
        layout.addWidget(btn_reset)

        group.setLayout(layout)
        return group

    def _init_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件")
        file_menu.addAction("打开", self.open_image, "Ctrl+O")
        file_menu.addAction("保存", self.save_image, "Ctrl+S")
        file_menu.addSeparator()
        file_menu.addAction("退出", self.close, "Ctrl+Q")

        edit_menu = menubar.addMenu("编辑")
        edit_menu.addAction("撤销", self.undo, "Ctrl+Z")
        edit_menu.addAction("重做", self.redo, "Ctrl+Y")
        edit_menu.addAction("恢复原图", self.reset_image)

        view_menu = menubar.addMenu("视图")
        view_menu.addAction("图像信息", self.show_image_info)
        view_menu.addAction("直方图", self._switch_to_histogram)
        view_menu.addAction("DICOM元数据", self.show_dicom_metadata)

        help_menu = menubar.addMenu("帮助")
        help_menu.addAction("关于", self.show_about)

    def _init_toolbar(self):
        self.toolbar = QToolBar("工具栏")
        self.toolbar.setIconSize(QSize(20, 20))
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)

        self.toolbar.addAction("打开", self.open_image)
        self.toolbar.addAction("保存", self.save_image)
        self.toolbar.addSeparator()
        self.toolbar.addAction("撤销", self.undo)
        self.toolbar.addAction("重做", self.redo)
        self.toolbar.addAction("恢复原图", self.reset_image)
        self.toolbar.addSeparator()
        self.toolbar.addAction("直方图", self._switch_to_histogram)

    def _init_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("就绪 - 请打开图像文件")

    def _history_message(self):
        return f"历史: {self.history_index + 1}/{len(self.history)}"

    def _refresh_display(self, status_message=None):
        if self.current_image is None:
            self.canvas.set_image(None)
            self.histogram_widget.plot_histogram(None)
            self.statusbar.showMessage(status_message or "就绪 - 请打开图像文件")
            return

        self.canvas.set_image(self.current_image)
        self.histogram_widget.plot_histogram(self.current_image)
        default_message = (
            f"图像大小: {self.current_image.shape[1]}x{self.current_image.shape[0]} | "
            f"{self._history_message()}"
        )
        self.statusbar.showMessage(status_message or default_message)

    def _push_history(self, image):
        if image is None:
            return
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        self.history.append(image.copy())
        self.history_index = len(self.history) - 1
        if len(self.history) > 30:
            self.history = self.history[-30:]
            self.history_index = len(self.history) - 1

    def _set_processing_state(self, is_processing, task_name=""):
        for widget in (self.left_panel, self.right_panel, self.viewer_tabs, self.toolbar, self.menuBar()):
            if widget is not None:
                widget.setEnabled(not is_processing)

        if is_processing:
            if QApplication.overrideCursor() is None:
                QApplication.setOverrideCursor(Qt.WaitCursor)
            self.processing_dialog = QProgressDialog(f"{task_name}中，请稍候...", None, 0, 0, self)
            self.processing_dialog.setWindowTitle("处理中")
            self.processing_dialog.setWindowModality(Qt.WindowModal)
            self.processing_dialog.setCancelButton(None)
            self.processing_dialog.setMinimumDuration(0)
            self.processing_dialog.setAutoClose(False)
            self.processing_dialog.setAutoReset(False)
            self.processing_dialog.show()
            self.statusbar.showMessage(f"{task_name}中，请稍候...")
        else:
            while QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
            if self.processing_dialog is not None:
                self.processing_dialog.close()
                self.processing_dialog.deleteLater()
                self.processing_dialog = None

    def _clear_processing_references(self):
        self.processing_thread = None
        self.processing_worker = None

    def _handle_processing_success(self, result, task_name, callback=None):
        try:
            if callback is not None:
                callback(result)
            elif isinstance(result, np.ndarray):
                self._apply_and_display(result)
            else:
                self.statusbar.showMessage(f"{task_name}完成")
        except Exception as exc:
            self._handle_processing_error(f"{task_name}完成后更新界面失败: {exc}")
            return
        self._set_processing_state(False)

    def _handle_processing_error(self, message):
        self._set_processing_state(False)
        self.statusbar.showMessage("处理失败")
        QMessageBox.critical(self, "处理失败", message)

    def _run_processing_task(self, task_name, operation, callback=None):
        if self.processing_thread is not None and self.processing_thread.isRunning():
            QMessageBox.information(self, "处理中", "当前仍有任务在执行，请稍候。")
            return

        self._set_processing_state(True, task_name)
        self.processing_thread = QThread(self)
        self.processing_worker = ProcessingWorker(operation)
        self.processing_worker.moveToThread(self.processing_thread)

        self.processing_thread.started.connect(self.processing_worker.run)
        self.processing_worker.finished.connect(
            lambda result, name=task_name, cb=callback: self._handle_processing_success(result, name, cb)
        )
        self.processing_worker.error.connect(self._handle_processing_error)
        self.processing_worker.finished.connect(self.processing_thread.quit)
        self.processing_worker.error.connect(self.processing_thread.quit)
        self.processing_worker.finished.connect(self.processing_worker.deleteLater)
        self.processing_worker.error.connect(self.processing_worker.deleteLater)
        self.processing_thread.finished.connect(self._clear_processing_references)
        self.processing_thread.finished.connect(self.processing_thread.deleteLater)
        self.processing_thread.start()

    def _apply_and_display(self, processed_image, push_history=True):
        if processed_image is None:
            return
        if push_history:
            self._push_history(processed_image)
        self.current_image = processed_image.copy()
        self._refresh_display()

    # ===== 文件操作 =====
    def open_image(self):
        image, success = self.file_manager.open_image(self)
        if success and image is not None:
            self.current_image = image.copy()
            self.processed_image = None
            self.history = [image.copy()]
            self.history_index = 0
            self._refresh_display(
                f"已打开: {self.file_manager.current_file_path} | "
                f"大小: {image.shape[1]}x{image.shape[0]}"
            )
            if self.file_manager.is_dicom:
                meta = self.file_manager.dicom_reader.metadata
                if 'WindowCenter' in meta:
                    try:
                        self.wl_center_spin.setValue(float(meta['WindowCenter']))
                    except ValueError:
                        pass
                if 'WindowWidth' in meta:
                    try:
                        self.wl_width_spin.setValue(float(meta['WindowWidth']))
                    except ValueError:
                        pass

    def save_image(self):
        if self.current_image is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return
        self.file_manager.save_image(self.current_image, self)

    def show_image_info(self):
        if self.current_image is None:
            QMessageBox.warning(self, "提示", "请先打开图像")
            return
        info = self.file_manager.get_image_info(self.current_image)
        dialog = QDialog(self)
        dialog.setWindowTitle("图像信息")
        dialog.setMinimumSize(400, 300)
        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        lines = []
        for key, value in info.items():
            lines.append(f"<b>{key}</b>: {value}")
        text_edit.setHtml("<br>".join(lines))
        layout.addWidget(text_edit)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.accepted.connect(dialog.accept)
        layout.addWidget(btn_box)
        dialog.exec_()

    def show_dicom_metadata(self):
        if not self.file_manager.is_dicom:
            QMessageBox.warning(self, "提示", "当前图像不是DICOM格式")
            return
        metadata = self.file_manager.dicom_reader.metadata
        dialog = DicomReader.show_metadata_dialog(metadata, self)
        dialog.exec_()

    # ===== 图像增强 =====
    def apply_brightness_contrast(self):
        if self.current_image is None:
            return
        brightness = self.brightness_slider.value()
        contrast = self.contrast_slider.value() / 100.0
        source_image = self.current_image.copy()
        self._run_processing_task(
            "亮度/对比度调整",
            lambda img=source_image, b=brightness, c=contrast: self.enhancer.adjust_brightness_contrast(img, b, c)
        )

    def apply_histogram_eq(self):
        if self.current_image is None:
            return
        source_image = self.current_image.copy()
        self._run_processing_task(
            "直方图均衡化",
            lambda img=source_image: self.enhancer.histogram_equalization(img)
        )

    def apply_clahe(self):
        if self.current_image is None:
            return
        source_image = self.current_image.copy()
        clip_limit = self.clahe_spin.value()
        self._run_processing_task(
            "CLAHE增强",
            lambda img=source_image, clip=clip_limit: self.enhancer.clahe(img, clip)
        )

    def apply_gamma(self):
        if self.current_image is None:
            return
        source_image = self.current_image.copy()
        gamma = self.gamma_spin.value()
        self._run_processing_task(
            "Gamma校正",
            lambda img=source_image, g=gamma: self.enhancer.gamma_correction(img, g)
        )

    def apply_invert(self):
        if self.current_image is None:
            return
        source_image = self.current_image.copy()
        self._run_processing_task(
            "图像反转",
            lambda img=source_image: self.enhancer.invert(img)
        )

    # ===== 滤波与噪声 =====
    def apply_filter(self):
        if self.current_image is None:
            return
        ksize = self.filter_ksize_spin.value()
        filter_type = self.filter_combo.currentText()
        source_image = self.current_image.copy()
        if filter_type == "高斯滤波":
            task_name = "高斯滤波"
            operation = lambda img=source_image, kernel=ksize: self.enhancer.gaussian_blur(img, kernel)
        elif filter_type == "中值滤波":
            task_name = "中值滤波"
            operation = lambda img=source_image, kernel=ksize: self.enhancer.median_blur(img, kernel)
        elif filter_type == "双边滤波":
            task_name = "双边滤波"
            operation = lambda img=source_image, kernel=ksize: self.enhancer.bilateral_filter(img, d=kernel)
        else:
            return
        self._run_processing_task(task_name, operation)

    def add_noise(self):
        if self.current_image is None:
            return
        noise_type = self.noise_combo.currentText()
        intensity = self.noise_spin.value()
        source_image = self.current_image.copy()
        if noise_type == "高斯噪声":
            task_name = "添加高斯噪声"
            operation = lambda img=source_image, sigma=intensity: self.enhancer.add_gaussian_noise(img, sigma=sigma)
        elif noise_type == "椒盐噪声":
            task_name = "添加椒盐噪声"
            operation = (
                lambda img=source_image, amount=intensity / 1000.0:
                self.enhancer.add_salt_pepper_noise(img, amount=amount)
            )
        else:
            return
        self._run_processing_task(task_name, operation)

    # ===== 几何变换 =====
    def apply_rotate(self, angle):
        if self.current_image is None:
            return
        source_image = self.current_image.copy()
        self._run_processing_task(
            "图像旋转",
            lambda img=source_image, a=angle: self.transformer.rotate(img, a)
        )

    def apply_flip_h(self):
        if self.current_image is None:
            return
        source_image = self.current_image.copy()
        self._run_processing_task(
            "水平翻转",
            lambda img=source_image: self.transformer.flip_horizontal(img)
        )

    def apply_flip_v(self):
        if self.current_image is None:
            return
        source_image = self.current_image.copy()
        self._run_processing_task(
            "垂直翻转",
            lambda img=source_image: self.transformer.flip_vertical(img)
        )

    def apply_resize(self):
        if self.current_image is None:
            return
        scale = self.scale_spin.value()
        source_image = self.current_image.copy()
        self._run_processing_task(
            "图像缩放",
            lambda img=source_image, s=scale: self.transformer.resize(img, s, s)
        )

    # ===== 形态学操作 =====
    def apply_erode(self):
        if self.current_image is None:
            return
        source_image = self.current_image.copy()
        kernel_size = self.morph_ksize_spin.value()
        self._run_processing_task(
            "腐蚀操作",
            lambda img=source_image, k=kernel_size: self.transformer.erode(img, k)
        )

    def apply_dilate(self):
        if self.current_image is None:
            return
        source_image = self.current_image.copy()
        kernel_size = self.morph_ksize_spin.value()
        self._run_processing_task(
            "膨胀操作",
            lambda img=source_image, k=kernel_size: self.transformer.dilate(img, k)
        )

    def apply_morph_open(self):
        if self.current_image is None:
            return
        source_image = self.current_image.copy()
        kernel_size = self.morph_ksize_spin.value()
        self._run_processing_task(
            "开运算",
            lambda img=source_image, k=kernel_size: self.transformer.morph_open(img, k)
        )

    def apply_morph_close(self):
        if self.current_image is None:
            return
        source_image = self.current_image.copy()
        kernel_size = self.morph_ksize_spin.value()
        self._run_processing_task(
            "闭运算",
            lambda img=source_image, k=kernel_size: self.transformer.morph_close(img, k)
        )

    # ===== 边缘检测 =====
    def apply_edge_detection(self):
        if self.current_image is None:
            return
        method = self.edge_combo.currentText()
        source_image = self.current_image.copy()
        if method == "Sobel":
            task_name = "Sobel边缘检测"
            operation = lambda img=source_image: self.analyzer.sobel_edge(img)
        elif method == "Canny":
            threshold1 = self.edge_thresh1_spin.value()
            threshold2 = self.edge_thresh2_spin.value()
            task_name = "Canny边缘检测"
            operation = (
                lambda img=source_image, t1=threshold1, t2=threshold2:
                self.analyzer.canny_edge(img, t1, t2)
            )
        elif method == "Laplacian":
            task_name = "Laplacian边缘检测"
            operation = lambda img=source_image: self.analyzer.laplacian_edge(img)
        else:
            return
        self._run_processing_task(task_name, operation)

    # ===== 阈值分割 =====
    def apply_threshold(self):
        if self.current_image is None:
            return
        method = self.threshold_combo.currentText()
        source_image = self.current_image.copy()
        if method == "全局阈值":
            threshold = self.threshold_spin.value()
            task_name = "全局阈值分割"
            operation = lambda img=source_image, th=threshold: self.analyzer.global_threshold(img, th)
        elif method == "Otsu自适应":
            task_name = "Otsu阈值分割"
            operation = lambda img=source_image: self.analyzer.otsu_threshold(img)
        elif method == "自适应阈值":
            task_name = "自适应阈值分割"
            operation = lambda img=source_image: self.analyzer.adaptive_threshold(img)
        else:
            return
        self._run_processing_task(task_name, operation)

    def apply_contour(self):
        if self.current_image is None:
            return
        source_image = self.current_image.copy()

        def on_success(payload):
            result, contours = payload
            if result is not None:
                self._apply_and_display(result)
                self.measure_result_label.setText(f"检测到轮廓: {len(contours)} 个")
                self.statusbar.showMessage(f"轮廓检测完成 | 检测到轮廓: {len(contours)} 个")

        self._run_processing_task(
            "轮廓检测",
            lambda img=source_image: self.analyzer.find_contours(img),
            on_success
        )

    def start_region_growing(self):
        if self.current_image is None:
            return
        self.canvas.set_mode('region_growing')
        self.statusbar.showMessage("区域生长模式: 请在图像上点击种子点")

    # ===== 测量工具 =====
    def start_measure_distance(self):
        if self.current_image is None:
            return
        self.canvas.set_mode('measure_distance')
        self.canvas.points = []
        self.statusbar.showMessage("距离测量: 请点击第一个点")

    def start_measure_angle(self):
        if self.current_image is None:
            return
        self.canvas.set_mode('measure_angle')
        self.canvas.points = []
        self.statusbar.showMessage("角度测量: 请依次点击三个点(端点-顶点-端点)")

    def start_roi(self):
        if self.current_image is None:
            return
        self.canvas.set_mode('roi')
        self.statusbar.showMessage("ROI选择: 请在图像上拖拽选择区域")

    # ===== 标注工具 =====
    def start_annotate_line(self):
        if self.current_image is None:
            return
        self.canvas.set_mode('annotate')
        self.canvas.set_annot_type('line')
        self.statusbar.showMessage("画线标注: 请在图像上拖拽画线")

    def start_annotate_rect(self):
        if self.current_image is None:
            return
        self.canvas.set_mode('annotate')
        self.canvas.set_annot_type('rect')
        self.statusbar.showMessage("矩形标注: 请在图像上拖拽画矩形")

    def start_annotate_text(self):
        if self.current_image is None:
            return
        self.canvas.set_mode('annotate')
        self.canvas.set_annot_type('text')
        self.statusbar.showMessage("文字标注: 请点击图像位置输入文字")

    def clear_annotations(self):
        self.canvas.clear_annotations()
        self.statusbar.showMessage("标注已清除")

    def _on_canvas_click(self, x, y):
        mode = self.canvas.mode
        if mode == 'region_growing':
            tolerance = self.region_tolerance_spin.value()
            self.canvas.set_mode('view')
            source_image = self.current_image.copy()

            def on_success(result):
                if result is not None:
                    self._apply_and_display(result)
                    self.statusbar.showMessage(f"区域生长完成 - 种子点: ({x}, {y}), 容差: {tolerance}")

            self._run_processing_task(
                "区域生长",
                lambda img=source_image, seed=(x, y), tol=tolerance: self.analyzer.region_growing(img, seed, tol),
                on_success
            )
        elif mode == 'measure_distance':
            points = self.canvas.points
            if len(points) >= 2:
                p1, p2 = points[0], points[1]
                dist_px, dist_mm = self.analyzer.measure_distance(p1, p2)
                text = f"距离: {dist_px:.1f} 像素"
                if dist_mm is not None:
                    text += f" | {dist_mm:.2f} mm"
                self.measure_result_label.setText(text)
                self.statusbar.showMessage(text)
                self.canvas.set_mode('view')

        elif mode == 'measure_angle':
            points = self.canvas.points
            if len(points) >= 3:
                angle = self.analyzer.measure_angle(points[0], points[1], points[2])
                text = f"角度: {angle:.1f}°"
                self.measure_result_label.setText(text)
                self.statusbar.showMessage(text)
                self.canvas.set_mode('view')

    def _on_roi_selected(self, x, y, w, h):
        if self.current_image is None:
            return
        roi = self.transformer.crop(self.current_image, x, y, w, h)
        if roi is not None:
            self._apply_and_display(roi)
            self.measure_result_label.setText(
                f"ROI: ({x},{y}) 大小 {w}x{h} | 面积: {w * h} 像素"
            )
        self.canvas.set_mode('view')

    # ===== DICOM窗宽窗位 =====
    def _set_dicom_preset(self, center, width):
        self.wl_center_spin.setValue(center)
        self.wl_width_spin.setValue(width)
        self.apply_window_level()

    def apply_window_level(self):
        if not self.file_manager.is_dicom:
            QMessageBox.warning(self, "提示", "当前图像不是DICOM格式，窗宽窗位功能仅适用于DICOM图像")
            return
        center = self.wl_center_spin.value()
        width = self.wl_width_spin.value()
        self._run_processing_task(
            "DICOM窗宽窗位调整",
            lambda c=center, w=width: self.file_manager.dicom_reader.get_windowed_image(c, w),
            lambda result, c=center, w=width: (
                self._apply_and_display(result),
                self.statusbar.showMessage(f"窗宽窗位: C={c}, W={w}")
            )
        )

    # ===== 历史操作 =====
    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.current_image = self.history[self.history_index].copy()
            self._refresh_display(f"撤销 | {self._history_message()}")

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_image = self.history[self.history_index].copy()
            self._refresh_display(f"重做 | {self._history_message()}")

    def reset_image(self):
        if len(self.history) > 0:
            self.current_image = self.history[0].copy()
            self.history = [self.history[0].copy()]
            self.history_index = 0
            self._refresh_display("已恢复原图")

    # ===== 其他 =====
    def _switch_to_histogram(self):
        if self.current_image is not None:
            self.histogram_widget.plot_histogram(self.current_image)
            self.viewer_tabs.setCurrentWidget(self.histogram_widget)

    def show_about(self):
        QMessageBox.about(
            self,
            "关于",
            "<h3>医学图像处理工具</h3>"
            "<p>生物医学工程 Python 大作业</p>"
            "<p>功能：图像增强、滤波、边缘检测、阈值分割、"
            "形态学操作、DICOM支持、测量工具等</p>"
            "<p>技术栈：PyQt5 + OpenCV + pydicom + matplotlib</p>"
        )
