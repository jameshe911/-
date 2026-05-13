import sys
import cv2
import numpy as np
import time
import os
from datetime import datetime
from ultralytics import YOLO
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from collections import defaultdict
# self前缀的语句，表明是在整个项目里面都使用的，更改数据的话，全部都会更改，没有self的就是普通的语句

MODEL_NAME = 'yolov8s.pt'
# 模型名称
CONF_THRESHOLD = 0.45
# 模型识别的时候，至少有多少的把握，才会真正标注出来
# 1.0就是100%正确
WARNING_ZONE = [(400, 200), (900, 200), (900, 550), (400, 550)]
# 异常行为识别区域

class DetectionWorker(QObject):
    frame_signal = pyqtSignal(np.ndarray)
    stats_signal = pyqtSignal(str)
    fps_signal = pyqtSignal(int)
    # 类似与函数的使用，但是是异步发生的，可以同时一起进行

    def __init__(self):
        super().__init__()
        self.running = True
        # 这个项目现在是可以运行的
        self.model = YOLO(MODEL_NAME)
        # 调用yolo模型
        self.zone_entry_time = {}
        self.frame_count = 0
        # 帧数计数器
        self.fps_timer = time.time()
    #     时间戳，为以后计算帧率做准备


    @pyqtSlot()
    def run(self):
        cap = cv2.VideoCapture(0)
        # 打开笔记本原有的摄像头
        while self.running:
            # 循环
            ret, frame = cap.read()
            # 读取每一帧的画面
            if not ret:
                continue
            #     如果没有读取成功，则跳过并且读取下一帧

            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            annotated = results.plot()
            boxes = results.boxes
            # 视频窗口中显现的id，编号，名称

            if boxes is not None:
                stats = {}
                for c in boxes.cls.int().tolist():
                    # 编号
                    name = results.names[c]
                    # 转化成中文名字，在数据库中有匹配的信息
                    stats[name] = stats.get(name, 0) + 1

                zone_np = np.array(WARNING_ZONE, np.int32)
                cv2.polylines(annotated, [zone_np], True, (0, 0, 255), 2)
                # 在画面上用红色画一个四边形。True表示首尾相连封闭，(0,0,255)是红色，2是线条粗细。
                current_intruders = set()
                # 集合记录什么东西在这个识别框中
                intruder_detail = []
                # 记录停留的时间

                if boxes.id is not None:
                    for box, tid in zip(boxes.xyxy, boxes.id.int().tolist()):
                        cx = int((box[0] + box[2]) / 2)
                        cy = int((box[1] + box[3]) / 2)
                        if cv2.pointPolygonTest(zone_np, (cx, cy), False) >= 0:
                            current_intruders.add(tid)
                            # 上面的都是判断是否进入标注框
                            cv2.circle(annotated, (cx, cy), 8, (0, 0, 255), -1)
                            now = time.time()
                            # 进来的话标注进入的时间
                            if tid not in self.zone_entry_time:
                                self.zone_entry_time[tid] = now
                            elapsed = now - self.zone_entry_time[tid]
                            if elapsed > 5:
                                intruder_detail.append(f"ID{tid}停留{elapsed:.1f}s")
                #                 这个就是判断时间是否到了5秒，到了的话，记录下来

                for tid in list(self.zone_entry_time.keys()):
                    if tid not in current_intruders:
                        del self.zone_entry_time[tid]

                txt = " | ".join([f"{k}:{v}" for k, v in stats.items()]) if stats else "无目标"
                if intruder_detail:
                    txt += " | ⚠️ 异常逗留: " + ", ".join(intruder_detail)
                self.stats_signal.emit(txt)
            #     这一串就是如果进来的异常对象，又退出去了，则需要重新进行计时

            self.frame_signal.emit(annotated)

            self.frame_count += 1
            if time.time() - self.fps_timer >= 1:
                self.fps_signal.emit(self.frame_count)
                self.frame_count = 0
                self.fps_timer = time.time()

            QThread.msleep(16)
        cap.release()
    #     释放摄像头资源

    def stop(self):
        self.running = False
#         之前不是有self.running = true，这里就是说明这个项目执行完毕，推出，无法再开启


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("冰晶天穹 · 异常行为检测")
        self.setGeometry(100, 40, 1550, 900)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.btn_start = QPushButton("▶ 启动")
        self.btn_stop = QPushButton("■ 停止")
        self.btn_snap = QPushButton("📸 截图")
        ctrl = QHBoxLayout()
        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        ctrl.addWidget(self.btn_snap)
        layout.addLayout(ctrl)

        self.lbl_stat = QLabel("🟢 待命")
        self.lbl_stat.setStyleSheet("font-size:18px; color:#A0D8EF; font-weight:bold;")
        self.lbl_fps = QLabel("FPS: --")
        self.lbl_fps.setStyleSheet("font-size:14px; color:#7BC8F2;")
        layout.addWidget(self.lbl_stat)
        layout.addWidget(self.lbl_fps)

        self.video_label = QLabel()
        self.video_label.setMinimumHeight(650)
        self.video_label.setStyleSheet("border:2px solid #5BA4CF; border-radius:20px; background:black;")
        layout.addWidget(self.video_label)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("S=截图 | 空格=启停 | 走进红框停留5秒触发警报")
        self.setStyleSheet("QMainWindow{background:#0A1628;} QPushButton{background:#1A3A5C;color:#D4EAF7;border-radius:8px;padding:6px 16px;}")

        self.worker = None
        self.thread = None
        self.detection_running = False

        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_snap.clicked.connect(self.snapshot)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            self.snapshot()
        elif event.key() == Qt.Key_Space:
            self.stop() if self.detection_running else self.start()

    def start(self):
        if self.thread and self.thread.isRunning():
            return
        self.thread = QThread()
        self.worker = DetectionWorker()
        self.worker.moveToThread(self.thread)
        # 这里是调用之前的后端书写的代码，放到视频显示窗口来应用
        self.thread.started.connect(self.worker.run)
        self.worker.frame_signal.connect(self.display)
        self.worker.stats_signal.connect(lambda t: self.lbl_stat.setText(t))
        self.worker.fps_signal.connect(lambda f: self.lbl_fps.setText(f"FPS: {f}"))
        self.thread.start()
        self.detection_running = True
        self.lbl_stat.setText("🟢 运行中")

    def display(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def snapshot(self):
        if self.video_label.pixmap() is None:
            return
        os.makedirs("snapshots", exist_ok=True)
        f = f"snapshots/{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        self.video_label.pixmap().save(f)
        self.status_bar.showMessage(f"📸 截图: {f}")

    def stop(self):
        if self.worker:
            self.worker.stop()
        if self.thread:
            self.thread.quit()
            self.thread.wait(2000)
        self.worker = None
        self.thread = None
        self.detection_running = False
        self.lbl_stat.setText("⚪ 已停止")

    def closeEvent(self, e):
        self.stop()
        e.accept()


if __name__ == "__main__":
    # 直接点开这个文件打开，则执行后续的操作
    app = QApplication(sys.argv)
    # 打开一个qt文件
    w = MainWindow()
    # 打开前端的窗口
    w.show()
    # 打开视频窗口
    sys.exit(app.exec_())
    # 循环调用后端的程序
    # 画面通过 cv2.VideoCapture(0) 从摄像头拿到手，交给 self.model.track() 这个YOLO大脑去检测和跟踪。
    # 大脑处理好之后，通过 frame_signal 信号把画好框的画面发射出去，
    # 前端 display 函数接收到信号后，把画面转成Qt格式显示在窗口里。