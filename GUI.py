
import numpy
import sys
import traceback
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, Qt

import shutil
import os
import time
import random
import csv
import cv2
from datetime import datetime
from ui import MainWindow
from ultralytics import YOLO


class ErrorWindow(QWidget):
    def __init__(self, name="错误提示", error_message="未知错误"):
        super().__init__()
        self.setWindowTitle(name)
        self.setGeometry(100, 100, 600, 400)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.center()

        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlainText(error_message)
        self.text_edit.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        layout.addWidget(self.text_edit)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        self.setLayout(layout)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


class MainGui(MainWindow):
    def __init__(self):
        super().__init__()

        self.label_width, self.label_height = self.label_img.size().width(), self.label_img.size().height()

        # 单个文件名字
        self.img_name = None
        # 文件保存地址（绘图）
        self.result_img_name = None
        # 类别名（img,dir,video）
        self.start_type = None
        if self.start_type not in ["cap", "video"]:
            self.pushButton_end.setEnabled(False)

        self.img_path = None
        self.img_path_dir = None
        self.cap = None
        self.video = None
        self.video_path = None

        self.worker_thread = None

        self.all_result = []
        self.comboBox_name = []

        self.selected_text = None
        self.number = 1
        self.RowLength = 0
        self.input_time = 0

        # ------------------------- 图片文件存储位置 -----------------------------
        # 获取当前工程文件位置
        self.ProjectPath = os.getcwd()

        # 保存所有的输出文件
        self.output_dir = os.path.join(self.ProjectPath, 'output')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        run_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.result_time_path = os.path.join(self.output_dir, run_time)
        os.mkdir(self.result_time_path)

        self.result_img_path = os.path.join(self.result_time_path, 'img_result')
        os.mkdir(self.result_img_path)
        # 保存绘制好的图片结果
        self.result_org_img_path = os.path.join(self.result_time_path, 'org_img')
        os.mkdir(self.result_org_img_path)

        # ---------------------------- 模型参数 --------------------------------
        self.weights_file_old_name = ""
        self.weights_file_name = ""
        self.model = None
        self.names = None
        self.color = {"font": (255, 255, 255)}

        self.error_window = None
        self.handle_buttons()

    def handle_buttons(self):
        """
        按钮控件链接
        """
        # ---------------------------- 模型文件 ---------------------------------
        self.pushButton_weights.clicked.connect(self.SelectWeights)

        # ---------------------------- 推理文件 ---------------------------------
        self.pushButton_img.clicked.connect(self.SelectImg)
        self.pushButton_dir.clicked.connect(self.SelectImgFile)
        self.pushButton_video.clicked.connect(self.SelectVideo)
        self.pushButton_cap.clicked.connect(self.SelectCap)

        # ---------------------------- 开始推理 ---------------------------------
        self.pushButton_start.clicked.connect(self.Infer)

        # ---------------------------- 停止推理 ---------------------------------
        self.pushButton_end.clicked.connect(self.InferEnd)

        # ---------------------------- 导出数据
        self.pushButton_export.clicked.connect(self.write_csv)

        # ---------------------------- 表格点击事件绑定 ---------------------------------
        self.table_widget.cellClicked.connect(self.cell_clicked)

        # ---------------------------- 下拉框点击事件 ---------------------------------
        self.comboBox_class.activated.connect(self.onComboBoxActivatedDetection)

    def Confidence(self):
        """
        得到置信度的值
        """
        confidence = '%.2f' % self.doubleSpinBox_conf.value()
        return eval(confidence)

    def IOU(self):
        """
        得到 IOU 的值
        """
        iou = '%.2f' % self.doubleSpinBox_iou.value()
        return eval(iou)

    def SelectWeights(self):
        """
        模型权重选择
        """
        self.weights_file_name, _ = QFileDialog.getOpenFileName(self, "选择权重文件", "",
                                                                "所有文件(*.pt *.onnx *.torchscript *.engine "
                                                                "*.mlmodel *.pb *.tflite *openvino_model  "
                                                                "*saved_model *paddle_model)")
        if self.weights_file_name:
            self.label_weights.setText(os.path.split(self.weights_file_name)[-1])

    def SelectImg(self):
        """
        图片文件选择
        """
        self.img_path, filetype = QFileDialog.getOpenFileName(self, "选择推理文件", "",
                                                              "所有文件(*.jpg *.bmp *.dng" " *.jpeg *.jpg *.mpo"
                                                              " *.png *.tif *.tiff *.webp *.pfm)")
        if self.img_path == "":
            self.start_type = None
            return

        # 显示相对应的文字
        self.start_type = 'img'
        self.img_name = os.path.split(self.img_path)[-1]

        self.label_img_path.setText(self.img_name)
        self.label_dir_path.setText("选择图片文件夹")
        self.label_video_path.setText("选择视频文件")
        self.lineEdit_cap_path.clear()
        self.lineEdit_cap_path.setPlaceholderText("选择相机源（摄像头）")

        self.org_img_save_path = os.path.join(self.result_org_img_path, self.img_name)

        # 显示原图
        self.label_img.clear()
        # 加载图片并设置到 QLabel
        self.loadImage(self.img_path)
        shutil.copy(self.img_path, self.org_img_save_path)

    def SelectImgFile(self):
        """
        图片文件夹选择
        """
        self.img_path_dir = QFileDialog.getExistingDirectory(None, "选择文件夹")
        if self.img_path_dir == '':
            self.start_type = None
            return

        self.start_type = 'dir'

        self.label_img_path.setText("选择图片文件")
        self.label_dir_path.setText(os.path.split(self.img_path_dir)[-1])
        self.label_video_path.setText("选择视频文件")
        self.lineEdit_cap_path.clear()
        self.lineEdit_cap_path.setPlaceholderText("选择相机源（摄像头）")

        self.image_files = [os.path.join(self.img_path_dir, file) for file in os.listdir(self.img_path_dir) if
                            file.lower().endswith(
                                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))]
        if self.image_files:
            self.img_path = self.image_files[0]
            self.img_name = os.path.split(self.img_path)[-1]

            self.label_img.clear()
            self.loadImage(self.img_path)

    def SelectVideo(self):
        """
        视频文件选择
        """
        # 选择文件
        self.video_path, filetype = QFileDialog.getOpenFileName(self, "选择推理文件", "",
                                                                "所有文件(*.asf *.avi *.gif *.m4v *.mkv "
                                                                "*.mov *.mp4 *.mpeg *.mpg *.ts *.wmv)")
        if self.video_path == "":  # 未选择文件
            self.start_type = None
            return
        self.start_type = 'video'
        self.img_name = os.path.split(self.video_path)[-1]

        self.label_img_path.setText("选择图片文件")
        self.label_dir_path.setText("选择图片文件夹")
        self.label_video_path.setText(self.img_name)
        self.lineEdit_cap_path.clear()
        self.lineEdit_cap_path.setPlaceholderText("选择相机源（摄像头）")

        shutil.copy(self.video_path, os.path.join(self.result_org_img_path, self.img_name))

    def SelectCap(self):
        """
        摄像头
        """
        self.label_img_path.setText("选择图片文件")
        self.label_dir_path.setText("选择图片文件夹")
        self.label_video_path.setText("选择视频文件")

        if self.lineEdit_cap_path.text() != '':
            try:
                self.video_path = eval(self.lineEdit_cap_path.text())
            except:
                self.video_path = self.lineEdit_cap_path.text()
        else:
            self.show_error_window("摄像头选择错误", "未选择指定的摄像头！！！")
            return
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if frame is None:
            self.show_error_window("摄像头选择错误", "当前摄像头不能读取！！！")
        self.start_type = 'cap'
        self.img_name = 'camera.mp4'

    def Infer(self):
        """
        根据 start_type 进行推理
        """
        if self.weights_file_name:
            if self.weights_file_old_name != self.weights_file_name:
                self.model = YOLO(model=self.weights_file_name)
                self.names = self.model.names
                self.color.update(
                    {self.names[i]: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                     for i in range(len(self.names))})

            if self.start_type == 'video' or self.start_type == 'cap':
                self.cap = cv2.VideoCapture(self.video_path)

                self.pushButton_start.setEnabled(False)
                self.pushButton_end.setEnabled(True)

                if self.worker_thread is None or not self.worker_thread.isRunning():
                    # 开启线程，否则界面会卡死
                    self.worker_thread = WorkerThread(self)
                    self.worker_thread.start()

            elif self.start_type == 'img':
                self.pushButton_start.setEnabled(False)
                img = cv2.imread(self.img_path)
                self.predict_img(img)
                self.pushButton_start.setEnabled(True)

            elif self.start_type == 'dir':
                self.pushButton_start.setEnabled(False)
                if self.worker_thread is None or not self.worker_thread.isRunning():
                    self.worker_thread = WorkerThread(self)
                    self.worker_thread.start()
        else:
            self.show_error_window("推理错误", "模型文件未选择！")

    def InferEnd(self):
        """
        用于视频/摄像头停止推理
        """
        try:
            if self.worker_thread is not None and self.worker_thread.isRunning():
                self.worker_thread.stop()
                self.worker_thread = None

                self.pushButton_end.setEnabled(False)
                self.pushButton_start.setEnabled(True)

                self.comboBox_class.clear()
                self.comboBox_class.addItem('None')
                for index in self.results_index.keys():
                    self.comboBox_class.addItem(index)
                self.comboBox_class.setCurrentText("None")
        except:
            self.show_error_window("停止推理错误", traceback.format_exc())

    def predict_img(self, img):
        """
        推理图片
        """
        try:
            start_time = time.time()
            self.result_img_name = os.path.join(self.result_img_path, self.img_name)
            results = self.model(img, conf=self.Confidence(), iou=self.IOU(),
                                 classes=[eval(i) for i in self.lineEdit_classes.text().split(
                                     ",")] if self.lineEdit_classes.text() != '' else None,
                                 stream=True)

            self.all_result = []
            for result in results:
                orig_img = result.orig_img
                for boxs, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                           result.boxes.conf.tolist()):
                    self.all_result.extend([[round(i, 2) for i in boxs] + [self.names[int(cls)]] + [round(conf, 4)]])

            # 保存结果图片
            self.draw = self.draw_info(orig_img, self.all_result)
            cv2.imwrite(self.result_img_name, self.draw)
            self.results_index = {f'目标{index + 1}': result for index, result in enumerate(self.all_result)}
            self.input_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            self.comboBox_class.clear()
            self.comboBox_class.addItem('None')
            for index in self.results_index.keys():
                self.comboBox_class.addItem(index)
            self.comboBox_class.setCurrentText("None")

            self.loadImage(self.draw)

            self.clear_info()
            self.end_time = str(round(time.time() - start_time, 4)) + 's'
            self.label_time.setText(self.end_time)

            # 显示表格信息
            self.show_table()
            self.number += 1
        except:
            self.show_error_window("推理错误", traceback.format_exc())

    def loadImage(self, image):
        """
        显示图片
        """
        if isinstance(image, numpy.ndarray):
            # 将 BGR 图片转换为 RGB 图片
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 将 cv2 图片对象转换为 QImage 对象
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # 将 QImage 对象转换为 QPixmap 对象
            pixmap = QPixmap.fromImage(q_image)
        else:
            # 加载图片
            pixmap = QPixmap(image)

        # 根据 QLabel 的大小缩放图片，保持纵横比
        scaled_pixmap = pixmap.scaled(self.label_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # 设置缩放后的图片到 QLabel
        self.label_img.setPixmap(scaled_pixmap)

    def show_table(self):
        """
        将推理结果显示在表格中
        """
        # 获取当前行数
        currentRowCount = self.table_widget.rowCount()
        # 向表格添加一行数据
        self.table_widget.insertRow(currentRowCount)  # 在最后一行之后添加新行

        for col, content in enumerate(
                [self.org_img_save_path, self.input_time, self.all_result, len(self.all_result),
                 self.end_time, self.result_img_name]):
            item = QTableWidgetItem(str(content))
            self.table_widget.setItem(currentRowCount, col, item)

        # 滚动到底部
        self.table_widget.scrollToBottom()

    def cell_clicked(self, row):
        """
        表格点击事件
        """
        if self.table_widget.item(row, 1) is None:
            return
        # 图片路径
        self.org_img_save_path = self.table_widget.item(row, 0).text()
        # 识别结果
        self.all_result = eval(self.table_widget.item(row, 2).text())
        # 推理时间
        self.infer_time = self.table_widget.item(row, 4).text()
        # 保存路径
        self.result_img_name = self.table_widget.item(row, 5).text()

        self.results_index = {f'目标{index + 1}': result for index, result in enumerate(self.all_result)}

        self.label_img.clear()
        draw_img = cv2.imread(self.result_img_name)
        self.loadImage(draw_img)

        self.comboBox_class.clear()
        self.comboBox_class.addItem('None')
        for index in self.results_index.keys():
            self.comboBox_class.addItem(index)

        self.label_time.setText(self.infer_time)
        self.clear_info()

    def show_info(self, result):
        """
        显示的坐标和置信度
        """
        self.label_score.setText(str(result[5]))
        self.label_xmin.setText('xmix: ' + str(result[0]))
        self.label_ymin.setText('ymin: ' + str(result[1]))
        self.label_xmax.setText('xmax: ' + str(result[2]))
        self.label_ymax.setText('ymax: ' + str(result[3]))

    def clear_info(self):
        """
        清除显示的坐标和置信度
        """
        self.label_score.clear()
        self.label_xmin.setText("xmix: 0")
        self.label_ymin.setText("ymin: 0")
        self.label_xmax.setText("xmax: 0")
        self.label_ymax.setText("ymax: 0")

    def onComboBoxActivatedDetection(self):
        """
        单个目标查看
        """
        self.selected_text = self.comboBox_class.currentText()
        if self.selected_text != 'None':
            lst_info = self.results_index[self.selected_text]
            draw_img = cv2.imread(self.org_img_save_path)

            draw_img = self.draw_info(draw_img, [lst_info])
            self.label_img.clear()
            self.loadImage(draw_img)
            self.show_info(lst_info)
        else:
            self.loadImage(self.result_img_name)
            self.clear_info()

    def draw_info(self, draw_img, results):
        """
        绘制识别结果
        """
        lw = max(round(sum(draw_img.shape) / 2 * 0.003), 2)  # line width
        tf = max(lw - 1, 1)  # font thickness
        sf = lw / 3  # font scale
        for result in results:
            box, name, conf = result[:4], result[4], result[5]

            color = self.color[name]
            label = f'{name} {conf}'

            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            # 绘制矩形框
            cv2.rectangle(draw_img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
            # text width, height
            w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
            # label fits outside box
            outside = box[1] - h - 3 >= 0
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            # 绘制矩形框填充
            cv2.rectangle(draw_img, p1, p2, color, -1, cv2.LINE_AA)
            # 绘制标签
            cv2.putText(draw_img, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0, sf, self.color["font"], thickness=2, lineType=cv2.LINE_AA)
        return draw_img

    def write_csv(self):
        """
        导出推理文件信息
        """
        result_csv = os.path.join(self.result_time_path, 'result.csv')

        num_rows = self.table_widget.rowCount()
        num_cols = self.table_widget.columnCount()
        datas = []
        for row in range(num_rows):
            row_data = []
            for col in range(num_cols):
                item = self.table_widget.item(row, col)
                if item is not None:
                    row_data.append(item.text())
                else:
                    row_data.append('')
            datas.append(row_data)

        with open(result_csv, "w", newline="") as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(['序号', '图片名称', '录入时间', '识别结果', '目标数目', '推理用时', '保存路径'])
            for data in datas:
                writer.writerow(data)

        QMessageBox.information(None, "成功", f"数据已保存！save path {result_csv}", QMessageBox.Yes)

    def closeEvent(self, event):
        """
        界面关闭事件，询问用户是否关闭
        """
        reply = QMessageBox.question(self, '退出', "是否要退出该界面？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.worker_thread is not None:
                # 确保线程安全地停止
                self.worker_thread.terminate()

            self.close()
            event.accept()
        else:
            event.ignore()

    def show_error_window(self, name, error_message):
        """
        显示错误窗口
        """
        if self.error_window is None:
            self.error_window = ErrorWindow(name, error_message)
        else:
            self.error_window.text_edit.setPlainText(error_message)
        self.error_window.show()


class WorkerThread(QThread):
    """
    识别视频/摄像头/文件夹 进程
    """

    def __init__(self, main_window):
        super().__init__()
        self.running = True
        self.main_window = main_window

    def run(self):
        try:
            if self.main_window.start_type == 'video' or self.main_window.start_type == "cap":
                if not self.main_window.cap.isOpened():
                    raise ValueError("Unable to open video file or cam")
                video_name = self.main_window.img_name if '.mp4' in self.main_window.img_name else \
                    self.main_window.img_name.split(".")[0] + '.mp4'
                frame_num = 0
                save_path = os.path.join(self.main_window.result_img_path, video_name)
                fps = 30.0 if 'camera' in video_name else self.main_window.cap.get(cv2.CAP_PROP_FPS)
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                             (int(self.main_window.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                              int(self.main_window.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                while self.running:
                    self.main_window.img_name = video_name.split(".")[0] + '_' + str(frame_num) + '.jpg'

                    ret, frame = self.main_window.cap.read()
                    if ret:
                        self.main_window.org_img_save_path = os.path.join(self.main_window.result_org_img_path,
                                                                          self.main_window.img_name)
                        cv2.imwrite(self.main_window.org_img_save_path, frame)

                        self.main_window.predict_img(frame)

                        frame_num += 1

                        vid_writer.write(self.main_window.draw)
                    else:
                        break
                self.main_window.cap.release()
                vid_writer.release()

            elif self.main_window.start_type == 'dir':
                for img_path in self.main_window.image_files:
                    img = cv2.imread(img_path)
                    self.main_window.img_name = os.path.split(img_path)[-1]
                    self.main_window.org_img_save_path = os.path.join(self.main_window.result_org_img_path,
                                                                      self.main_window.img_name)
                    shutil.copy(img_path, self.main_window.org_img_save_path)
                    self.main_window.predict_img(img)

            self.main_window.pushButton_end.setEnabled(False)
            self.main_window.pushButton_start.setEnabled(True)
        except:
            self.main_window.show_error_window("线程运行错误", traceback.format_exc())
            self.stop()

    def stop(self):
        self.running = False
        self.wait()


def main():
    app = QApplication(sys.argv)
    window = MainGui()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
