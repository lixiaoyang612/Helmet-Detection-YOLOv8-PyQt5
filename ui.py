
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QCursor, QPalette, QBrush, QPixmap
from PyQt5.QtWidgets import QSizePolicy, QToolTip, QScrollArea
import warnings
warnings.filterwarnings("ignore")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        # 设置窗口属性
        self.setEnabled(True)
        self.setFont(QtGui.QFont("黑体", 9))
        self.setWindowTitle("电动车驾驶员佩戴安全帽检测识别系统")
        # 设定窗口的初始大小
        self.resize(1300, 800)
        # 设置窗口大小策略
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 确保窗口大小不会小于布局内容的最小大小
        self.setMinimumSize(1300, 800)

        # 设置背景图片
        self.set_background_image()

        # 创建中心组件
        main_widget = QtWidgets.QWidget(self)

        # 创建一个布局管理器
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        left_right_layout = QtWidgets.QHBoxLayout(main_widget)
        left_layout = QtWidgets.QVBoxLayout(main_widget)
        right_layout = QtWidgets.QVBoxLayout(main_widget)

        # --------------------- 标题 ---------------------
        label_title = QtWidgets.QLabel()
        label_title.setMinimumSize(QtCore.QSize(0, 0))
        label_title.setMaximumSize(QtCore.QSize(10000, 10000))
        label_title.setFont(QtGui.QFont("微软雅黑", 30, QtGui.QFont.Bold))
        label_title.setText("基于YOLO的电动车驾驶员佩戴安全帽检测识别系统")
        label_title.setAlignment(QtCore.Qt.AlignCenter)

        # --------------------- 分割线 ---------------------
        dividing_line_1 = self.create_dividing_line()
        dividing_line_2 = self.create_dividing_line()
        dividing_line_3 = self.create_dividing_line()
        dividing_line_4 = self.create_dividing_line()
        dividing_line_5 = self.create_dividing_line()
        dividing_line_6 = self.create_dividing_line()

        # --------------------- 左边 图像显示 ---------------------
        self.label_img = QtWidgets.QLabel()
        self.label_img.setText("图形展示区域")
        # 设置边框
        self.label_img.setFrameShape(QtWidgets.QFrame.Box)
        # 设置大小策略为Expanding，这样label_img会尽可能填充可用空间
        self.label_img.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding))
        self.label_img.setAlignment(QtCore.Qt.AlignCenter)
        self.label_img.setMinimumSize(800, 400)  # 设置最小大小

        # --------------------- 左边 表格显示 ---------------------
        self.table_widget = QtWidgets.QTableWidget(self)
        self.table_widget.setFont(QtGui.QFont("黑体", 9))
        # 初始化时不设置行数，后续可以添加
        self.table_widget.setRowCount(0)
        # 设置列数
        self.table_widget.setColumnCount(6)
        # 设置表头
        self.table_widget.setHorizontalHeaderLabels(
            ["图片名称", "录入时间", "识别结果", "目标数目", "推理用时", "保存路径"])
        # 设置表格自动适应控件大小
        self.table_widget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # 设置表格线宽
        self.table_widget.setLineWidth(0)
        # 设置滚动条的政策
        self.table_widget.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.table_widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        # 设置滚动模式
        self.table_widget.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.table_widget.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        # 设置是否自动滚动
        self.table_widget.setAutoScroll(True)
        # 设置单元格的编辑触发方式，这里设置为不允许编辑
        self.table_widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # 设置是否显示网格线
        self.table_widget.setShowGrid(True)
        # 设置网格线的样式
        self.table_widget.setGridStyle(QtCore.Qt.CustomDashLine)
        # 设置是否允许排序
        self.table_widget.setSortingEnabled(False)
        # 设置文本自动换行
        self.table_widget.setWordWrap(True)
        # 设置角落按钮是否启用
        self.table_widget.setCornerButtonEnabled(True)
        # 设置单元格鼠标悬停时的提示
        self.table_widget.setMouseTracking(True)  # 启用鼠标追踪
        self.table_widget.cellEntered.connect(self.showToolTip)
        self.table_widget.setMinimumSize(800, 200)  # 设置最小大小

        # 使用 QScrollArea 包裹 table_widget
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.table_widget)
        scroll_area.setWidgetResizable(True)  # 允许表格内容自适应

        # --------------------- 右边 按钮 ---------------------
        self.pushButton_img = self.create_button("", './static/icons/file.png')
        self.pushButton_dir = self.create_button("", './static/icons/dir.png')
        self.pushButton_video = self.create_button("", './static/icons/shipinwenjian.png')
        self.pushButton_cap = self.create_button("", './static/icons/shexiangtou.png')
        self.pushButton_weights = self.create_button("", './static/icons/weights.png')
        self.pushButton_start = self.create_button("开始运行 >", '')
        self.pushButton_end = self.create_button("停止运行 >", '')
        self.pushButton_export = self.create_button("导出数据 >", '')

        # --------------------- 右边 文本框 ---------------------
        self.lineEdit_cap_path = self.create_line_edit("选择相机源（摄像头）")
        self.lineEdit_classes = self.create_line_edit("推理指定类别")

        # --------------------- 右边 单选框 ---------------------
        self.doubleSpinBox_iou = self.create_double_spinbox(0.25)
        self.doubleSpinBox_conf = self.create_double_spinbox(0.45)

        # --------------------- 右边 标签框 ---------------------
        self.label_img_path = self.create_label("选择图片文件")
        self.label_dir_path = self.create_label("选择图片文件夹")
        self.label_video_path = self.create_label("选择视频文件")
        self.label_weights = self.create_label("选择weights文件")
        label_select_iou = self.create_label("IOU")
        label_select_conf = self.create_label("Conf")
        label_classes = self.create_label("类别")
        label_time = self.create_label("", "./static/icons/yongshi.png")
        self.label_time = self.create_label("0s")
        label_class = self.create_label("", "./static/icons/leibie.png")
        label_score = self.create_label("", "./static/icons/zhixindu.png")
        self.label_score = self.create_label("0.0")
        label_coor = self.create_label("目标位置")
        self.label_xmin = self.create_label("xmin: 0")
        self.label_ymin = self.create_label("ymin: 0")
        self.label_xmax = self.create_label("xmax: 0")
        self.label_ymax = self.create_label("ymax: 0")

        # 初始化下拉框
        self.comboBox_class = self.create_combobox(["None"])

        # 设置按钮的尺寸策略和最小尺寸
        button_size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        button_min_size = QtCore.QSize(100, 30)  # 可以根据需要调整按钮的最小尺寸
        for button in [self.pushButton_img, self.pushButton_dir, self.pushButton_video,
                       self.pushButton_cap, self.pushButton_weights, self.pushButton_start,
                       self.pushButton_end, self.pushButton_export]:
            button.setSizePolicy(button_size_policy)
            button.setMinimumSize(button_min_size)

        # 设置单选框和下拉框的尺寸策略和最小尺寸
        spinbox_size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        spinbox_min_size = QtCore.QSize(200, 30)
        for spinbox in [self.doubleSpinBox_iou, self.doubleSpinBox_conf, self.comboBox_class]:
            spinbox.setSizePolicy(spinbox_size_policy)
            spinbox.setMinimumSize(spinbox_min_size)

        # 设置标签的尺寸策略和最小尺寸
        label_size_policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        label_min_size = QtCore.QSize(200, 30)
        for label in [self.label_img_path, self.label_dir_path, self.label_video_path,
                      self.label_weights, self.lineEdit_cap_path, self.lineEdit_classes,
                      label_select_iou, label_select_conf, label_classes, label_time,
                      label_class, label_score, self.label_time, self.label_score,
                      label_coor, self.label_xmin, self.label_xmax, self.label_ymin, self.label_ymax]:
            label.setSizePolicy(label_size_policy)
            label.setMinimumSize(label_min_size)

        # --------------------- 将控件添加到布局中 ---------------------
        # 设置right_layout的尺寸策略
        right_widget = QtWidgets.QWidget()
        # 将right_layout设置为right_widget的布局
        right_widget.setLayout(right_layout)

        main_layout.addWidget(label_title)
        main_layout.addWidget(dividing_line_1)

        left_layout.addWidget(self.label_img)
        left_layout.addWidget(scroll_area)  # 使用 scroll_area 包裹 table_widget
        # 设置第一个控件的拉伸因子为2
        left_layout.setStretchFactor(self.label_img, 2)
        # 设置第二个控件的拉伸因子为1
        left_layout.setStretchFactor(scroll_area, 1)

        # 右边局部
        layout_img = QtWidgets.QHBoxLayout(main_widget)
        layout_img.addWidget(self.pushButton_img)
        layout_img.addWidget(self.label_img_path)
        right_layout.addLayout(layout_img)

        layout_dir = QtWidgets.QHBoxLayout(main_widget)
        layout_dir.addWidget(self.pushButton_dir)
        layout_dir.addWidget(self.label_dir_path)
        right_layout.addLayout(layout_dir)

        layout_video = QtWidgets.QHBoxLayout(main_widget)
        layout_video.addWidget(self.pushButton_video)
        layout_video.addWidget(self.label_video_path)
        right_layout.addLayout(layout_video)

        layout_cap = QtWidgets.QHBoxLayout(main_widget)
        layout_cap.addWidget(self.pushButton_cap)
        layout_cap.addWidget(self.lineEdit_cap_path)
        right_layout.addLayout(layout_cap)

        layout_weights = QtWidgets.QHBoxLayout(main_widget)
        layout_weights.addWidget(self.pushButton_weights)
        layout_weights.addWidget(self.label_weights)
        right_layout.addLayout(layout_weights)

        right_layout.addWidget(dividing_line_2)

        layout_iou = QtWidgets.QHBoxLayout(main_widget)
        layout_iou.addWidget(label_select_iou)
        layout_iou.addWidget(self.doubleSpinBox_iou)
        right_layout.addLayout(layout_iou)

        layout_conf = QtWidgets.QHBoxLayout(main_widget)
        layout_conf.addWidget(label_select_conf)
        layout_conf.addWidget(self.doubleSpinBox_conf)
        right_layout.addLayout(layout_conf)

        layout_classes = QtWidgets.QHBoxLayout(main_widget)
        layout_classes.addWidget(label_classes)
        layout_classes.addWidget(self.lineEdit_classes)
        right_layout.addLayout(layout_classes)

        right_layout.addWidget(dividing_line_3)

        layout_run = QtWidgets.QVBoxLayout(main_widget)
        layout_run_ = QtWidgets.QHBoxLayout(main_widget)
        layout_run_.addWidget(self.pushButton_start)
        layout_run_.addWidget(self.pushButton_end)
        layout_run.addLayout(layout_run_)
        layout_run.addWidget(self.pushButton_export)
        right_layout.addLayout(layout_run)

        right_layout.addWidget(dividing_line_4)

        layout_time = QtWidgets.QHBoxLayout(main_widget)
        layout_time.addWidget(label_time)
        layout_time.addWidget(self.label_time)
        right_layout.addLayout(layout_time)

        layout_class = QtWidgets.QHBoxLayout(main_widget)
        layout_class.addWidget(label_class)
        layout_class.addWidget(self.comboBox_class)
        right_layout.addLayout(layout_class)

        layout_conf = QtWidgets.QHBoxLayout(main_widget)
        layout_conf.addWidget(label_score)
        layout_conf.addWidget(self.label_score)
        right_layout.addLayout(layout_conf)

        right_layout.addWidget(dividing_line_5)

        layout_coor = QtWidgets.QVBoxLayout(main_widget)
        layout_coor.addWidget(label_coor)
        right_layout.addLayout(layout_coor)

        layout_xy_1 = QtWidgets.QHBoxLayout(main_widget)
        layout_xy_2 = QtWidgets.QHBoxLayout(main_widget)
        layout_xy_1.addWidget(self.label_xmin)
        layout_xy_1.addWidget(self.label_ymin)
        layout_xy_2.addWidget(self.label_xmax)
        layout_xy_2.addWidget(self.label_ymax)
        right_layout.addLayout(layout_xy_1)
        right_layout.addLayout(layout_xy_2)

        right_layout.addWidget(dividing_line_6)

        left_right_layout.addLayout(left_layout)
        left_right_layout.setStretchFactor(left_layout, 1)
        left_right_layout.addWidget(right_widget)
        left_right_layout.setStretchFactor(right_widget, 1)
        main_layout.addLayout(left_right_layout)

        # 设置main_widget的布局
        main_widget.setLayout(main_layout)

        # 设置中心组件
        self.setCentralWidget(main_widget)

    def set_background_image(self):
        """
        设置窗口的背景图片
        """
        palette = self.palette()
        background_image_path = './static/icons/backgroung.png'  # 替换为您的背景图片路径
        background_image = QtGui.QPixmap(background_image_path).scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        palette.setBrush(QPalette.Window, QBrush(background_image))
        self.setPalette(palette)

    def showToolTip(self, row, col):
        # 获取单元格数据
        item = self.table_widget.item(row, col)
        if item:
            # 设置工具提示文本
            text = item.text()
            # 获取鼠标当前位置的全局坐标
            global_pos = QCursor.pos()
            # 显示工具提示
            QToolTip.showText(global_pos, text)

    def get_line_edit_style(self):
        return "QLineEdit { border: 2px solid black; border-radius: 5px; }"

    def get_label_style(self):
        return "QLabel { background-color: #FFFFFF }"

    def create_label(self, text=None, icon_path=None):
        label = QtWidgets.QLabel(text)
        if icon_path:
            pixmap = QtGui.QPixmap(icon_path)
            # 指定图标的大小，例如40x40
            icon_size = QtCore.QSize(40, 27)
            scaled_pixmap = pixmap.scaled(icon_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
        if 'x' not in text and 'y' not in text:
            # 设置标签内容居中显示
            label.setAlignment(QtCore.Qt.AlignCenter)
        label.setFrameShape(QtWidgets.QFrame.Box)
        label.setStyleSheet(self.get_label_style())
        return label

    def create_line_edit(self, placeholder):
        line_edit = QtWidgets.QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        line_edit.setStyleSheet(self.get_line_edit_style())
        return line_edit

    def create_double_spinbox(self, default_value):
        spinbox = QtWidgets.QDoubleSpinBox()
        spinbox.setMinimum(0.0)
        spinbox.setMaximum(1.0)
        spinbox.setSingleStep(0.01)
        spinbox.setValue(default_value)
        return spinbox

    def create_button(self, text=None, icon_path=None):
        button = QtWidgets.QPushButton()
        button.setFont(QtGui.QFont("黑体", 10, QtGui.QFont.Bold))
        if text:
            button.setText(text)
        if icon_path:
            button.setIcon(QtGui.QIcon(icon_path))
            button.setIconSize(QtCore.QSize(40, 27))
        return button

    def create_dividing_line(self):
        dividing_line = QtWidgets.QFrame()
        dividing_line.setFont(QtGui.QFont("黑体", 10))
        dividing_line.setAcceptDrops(True)
        dividing_line.setToolTipDuration(-2)
        dividing_line.setFrameShape(QtWidgets.QFrame.HLine)
        dividing_line.setFrameShadow(QtWidgets.QFrame.Raised)
        dividing_line.setLineWidth(1)
        return dividing_line

    def create_combobox(self, items):
        combobox = QtWidgets.QComboBox()
        combobox.setFont(QtGui.QFont("黑体", 10))
        for item in items:
            combobox.addItem(item)
        return combobox


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())