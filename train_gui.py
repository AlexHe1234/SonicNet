import sys
import time
from PyQt6.QtCore import QSize, QThread, QUrl
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, \
    QLabel, QLineEdit, QWidget, QGridLayout, \
    QProgressBar, QListWidget, QListWidgetItem, QDial
from PyQt6.QtGui import QPixmap
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtMultimedia import QMediaPlayer
from SonicNet.utils.cfg_from_gui import *
from SonicNet import train_yaml


class Thread(QThread):
    def __init__(self, func):
        super().__init__()
        self.func = func

    # run this one time
    def run(self):
        self.func()
        self.quit()
        self.wait()


class ThreadMultiTime(QThread):
    def __init__(self, func):
        super().__init__()
        self.func = func

    # run this forever
    def run(self):
        time.sleep(2)
        while True:
            self.func()
            time.sleep(0.1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.path = os.path.dirname(__file__)
        self.setWindowTitle("Version 0.0.1")
        self.setFixedSize(QSize(640, 480))
        self.init_a()
        self.setCentralWidget(self.containerA)
        # training thread
        self.train_class = train_yaml.Train()
        self.threadC1 = Thread(self.train_class.main)
        self.threadE1 = Thread(self.train_class.generate_vid)

    def init_a(self):
        self.layoutA = QGridLayout()
        # button to B
        self.buttonA = QPushButton()
        self.buttonA.setText('自动成像\nAutomatic Imaging')
        self.layoutA.addWidget(self.buttonA, 3, 1, 1, 1)
        self.buttonA.clicked.connect(self.a2b)
        # button to E
        self.buttonA2 = QPushButton('结果渲染\nModel Rendering')
        self.layoutA.addWidget(self.buttonA2, 3, 2, 1, 1)
        self.buttonA2.clicked.connect(self.a2d)
        # read me label
        self.labelA1 = QLabel(' ' * 75 + '超声成像演示程序\n' +
                              ' ' * 72 + 'Sonic reconstruction\n\n' +
                              ' ' * 70 + '请点击下方按键选择功能\n' +
                              ' ' * 48 + 'Select between functions with the buttons below' +
                              '\n'*6 + ' '*135 + '软件：何光昭 Alex He')
        self.layoutA.addWidget(self.labelA1, 2, 1, 1, 2)
        # credits
        # self.labelA2 = QLabel('小组成员（拼音字母排序）：段兴宇，付俊伟，何光昭，屈煊，宋刘洋，张鼎鼎')
        # self.layoutA1.addWidget(self.labelA2, 4, 1, 1, 1)
        # video player
        self.playerA1 = QMediaPlayer()
        self.playerA1.setSource(QUrl.fromLocalFile(self.path + '\\utils\\homepage_vid.mp4'))
        self.videoA1 = QVideoWidget()
        self.playerA1.setVideoOutput(self.videoA1)
        self.playerA1.play()
        self.playerA1.mediaStatusChanged.connect(self.restart_video)
        self.layoutA.addWidget(self.videoA1, 1, 1, 1, 2)
        # create container
        self.containerA = QWidget()
        self.containerA.setLayout(self.layoutA)
        return

    def init_b(self):
        self.layoutB = QGridLayout()
        # display input
        self.labelB1 = QLabel('\n' * 5 + ' ' * 80 + '在下方输入：\n' + ' ' * 75 + 'Please input below:')
        self.layoutB.addWidget(self.labelB1, 2, 1, 1, 4)
        # instructions
        self.labelB2 = QLabel(' ' * 62 + '请输入模型名称(不带空格),并单击开始：\n' + ' ' * 40 +
                              'Please input name of the model(no whitespace), before clicking start')
        self.layoutB.addWidget(self.labelB2, 1, 1, 1, 4)
        # blank
        # self.labelB3 = QLabel()
        # self.layoutB.addWidget(self.labelB3, 1, 1, 1, 4)
        # input box
        self.inputB1 = QLineEdit()
        self.inputB1.textChanged.connect(self.labelB1.setText)
        self.layoutB.addWidget(self.inputB1, 3, 1, 1, 4)
        # starting button
        self.buttonB1 = QPushButton('开始\nStart')
        self.buttonB1.clicked.connect(self.b2c)
        self.layoutB.addWidget(self.buttonB1, 4, 3, 1, 2)
        # return to home page
        self.buttonB2 = QPushButton('返回\nBack')
        self.layoutB.addWidget(self.buttonB2, 4, 1, 1, 2)
        self.buttonB2.clicked.connect(self.b2a)
        # create container
        self.containerB = QWidget()
        self.containerB.setLayout(self.layoutB)
        return

    def init_c(self):
        self.layoutC = QGridLayout()
        # render picture
        self.labelC1 = QLabel()
        self.pixmapC1 = QPixmap()
        self.labelC1.setPixmap(self.pixmapC1)
        self.layoutC.addWidget(self.labelC1, 1, 2, 1, 3)
        # plotted graph
        # self.labelC2 = QLabel()
        # self.pixmapC2 = QPixmap()
        # self.labelC2.setPixmap(self.pixmapC2)
        # self.layoutC.addWidget(self.labelC2, 2, 3, 1, 1)
        # progress.yaml bar
        self.barC1 = QProgressBar()
        self.barC1.setMaximum(100)
        self.barC1.setValue(0)
        self.layoutC.addWidget(self.barC1, 3, 1, 1, 4)
        # stop and return
        self.buttonC1 = QPushButton('停止\nStop')
        self.buttonC1.clicked.connect(self.c2a)
        self.layoutC.addWidget(self.buttonC1, 4, 1, 1, 5)
        # legend 1
        self.labelC3 = QLabel(' ' * 13 + '效果\n' + ' ' * 12 + 'Effect')
        self.layoutC.addWidget(self.labelC3, 1, 1, 1, 1)
        # legend 2
        # self.labelC4 = QLabel(' ' * 13 + '曲线\n' + ' ' * 12 + 'Curve')
        # self.layoutC.addWidget(self.labelC4, 1, 3, 1, 1)
        # progress.yaml
        self.labelC5 = QLabel('starting')
        self.layoutC.addWidget(self.labelC5, 3, 5, 1, 1)
        # update thread
        self.threadC2 = ThreadMultiTime(self.update_progress_c)
        # create container
        self.containerC = QWidget()
        self.containerC.setLayout(self.layoutC)
        return

    def init_d(self):
        self.layoutD = QGridLayout()
        self.tmp_list = os.listdir(self.path + '\\model')
        # title
        self.labelD1 = QLabel(' '*75 + '请选择要渲染的模型\n' +
                              ' '*73 + 'Please choose model')
        self.layoutD.addWidget(self.labelD1, 1, 1, 1, 4)
        # back
        self.buttonD1 = QPushButton('返回\nBack')
        self.layoutD.addWidget(self.buttonD1, 4, 1, 1, 2)
        self.buttonD1.clicked.connect(self.d2a)
        # render
        self.buttonD2 = QPushButton('渲染\nRender')
        self.layoutD.addWidget(self.buttonD2, 4, 3, 1, 2)
        # model list
        self.listD1 = QListWidget()
        for model in self.tmp_list:
            item = QListWidgetItem(model.split('.')[0])
            self.listD1.addItem(item)
        self.layoutD.addWidget(self.listD1, 2, 1, 2, 4)
        # if no model available
        if len(self.tmp_list) > 0:
            self.buttonD2.clicked.connect(self.d2e)
            self.listD1.setCurrentRow(0)
        # create container
        self.containerD = QWidget()
        self.containerD.setLayout(self.layoutD)
        return

    def init_e(self):
        self.layoutE = QGridLayout()
        # rendering instruction
        self.labelE1 = QLabel(' '*77 + '正在渲染，请稍后...\n' + ' '*72 +
                              'Rendering, please wait...')
        self.layoutE.addWidget(self.labelE1, 1, 1, 1, 4)
        # progress label
        self.labelE2 = QLabel('')
        self.layoutE.addWidget(self.labelE2, 2, 4, 1, 1)
        # progressbar
        self.barE1 = QProgressBar()
        self.layoutE.addWidget(self.barE1, 2, 1, 1, 3)
        # stop button
        self.buttonE1 = QPushButton('停止\nStop')
        self.layoutE.addWidget(self.buttonE1, 3, 1, 1, 4)
        self.buttonE1.clicked.connect(self.e2a)
        # create thread
        self.threadE2 = ThreadMultiTime(self.update_progress_e)
        # create container
        self.containerE = QWidget()
        self.containerE.setLayout(self.layoutE)
        return

    def init_f(self):
        self.layoutF = QGridLayout()
        # instruction
        self.labelF1 = QLabel(' '*73 + '旋转旋钮查看360度结果\n' +
                              ' '*65 + 'Turn the wheel to view 360 result')
        self.layoutF.addWidget(self.labelF1, 1, 1, 1, 4)
        # show image
        self.labelF2 = QLabel()
        self.pixmapF1 = QPixmap()
        self.pixmapF1.load(self.path + '\\cache\\rendered_tmp_000.jpg')
        self.labelF2.setPixmap(self.pixmapF1)
        self.layoutF.addWidget(self.labelF2, 2, 1, 1, 3)
        # dial
        self.dialF1 = QDial()
        self.dialF1.setRange(0, 99)
        self.dialF1.setSingleStep(1)
        self.dialF1.setValue(50)
        self.dialF1.valueChanged.connect(self.dial_changed)
        self.layoutF.addWidget(self.dialF1, 2, 4, 1, 1)
        # back
        self.buttonF = QPushButton('返回\nBack')
        self.layoutF.addWidget(self.buttonF, 3, 1, 1, 4)
        self.buttonF.clicked.connect(self.f2a)
        # create container
        self.containerF = QWidget()
        self.containerF.setLayout(self.layoutF)
        return

    def f2a(self):
        self.init_a()
        self.setCentralWidget(self.containerA)

    def e2a(self):
        self.threadE1.terminate()
        self.threadE2.terminate()
        self.init_a()
        self.setCentralWidget(self.containerA)
        return

    def d2e(self):
        self.save_render_model()
        self.init_e()
        self.setCentralWidget(self.containerE)
        self.start_rendering()
        self.start_rendering_progress()
        return

    def a2d(self):
        self.init_d()
        self.setCentralWidget(self.containerD)
        return

    def a2b(self):
        self.init_b()
        self.setCentralWidget(self.containerB)
        return

    def b2a(self):
        self.init_a()
        self.setCentralWidget(self.containerA)
        return

    def d2a(self):
        self.init_a()
        self.setCentralWidget(self.containerA)
        return

    def b2c(self):
        self.save_train_model()
        self.init_c()
        self.setCentralWidget(self.containerC)
        self.start_training()
        self.start_training_progress()
        return

    def c2a(self):
        self.threadC1.terminate()
        self.threadC2.terminate()
        self.init_a()
        self.setCentralWidget(self.containerA)
        return

    def e2f(self):
        self.threadE1.terminate()
        self.threadE2.terminate()
        self.init_f()
        self.setCentralWidget(self.containerF)
        return

    def dial_changed(self, value):
        value = int(value)
        if 0 <= value <= 9:
            string = '00' + str(value)
        elif 10 <= value <= 99:
            string = '0' + str(value)
        image_path = self.path +'\\cache\\rendered_tmp_' + string + '.jpg'
        self.pixmapF1.load(image_path)
        self.labelF2.setPixmap(self.pixmapF1)
        return

    def restart_video(self, status):
        if status == self.playerA1.MediaStatus.EndOfMedia:
            self.playerA1.play()
        return

    def save_train_model(self):
        train_model = self.inputB1.text()
        yml_save('train_name', train_model)
        return

    def save_render_model(self):
        render_model = self.tmp_list[self.listD1.currentRow()].split('.')[0]
        yml_save('render_name', render_model)
        return

    def update_progress_c(self):
        self.f = open(os.path.dirname(__file__) + '\\utils\\progress.yaml', 'r')
        self.progress = yaml.safe_load(self.f)
        yaml.dump(self.progress)
        self.f.close()
        if self.progress is None:
            return
        # # self.progress = {'percent': 0, 'status': 'yep'}
        self.labelC5.setText(self.progress['state'])
        self.barC1.setValue(int(self.progress['percent']))
        # update graph
        self.pixmapC1.load(os.path.dirname(__file__) + '\\utils\\progress_pic.jpg')
        self.labelC1.setPixmap(self.pixmapC1)
        # self.pixmapC2.load(os.path.dirname(__file__) + '\\utils\\loss_graph.jpg')
        # self.labelC2.setPixmap(self.pixmapC2)
        # finished
        if self.progress['percent'] == 100:
            self.buttonC1.setText('返回\nBack')
        return

    def update_progress_e(self):
        self.f = open(os.path.dirname(__file__) + '\\utils\\progress.yaml', 'r')
        self.progress = yaml.safe_load(self.f)
        yaml.dump(self.progress)
        self.f.close()
        if self.progress is None:
            return
        # # self.progress = {'percent': 0, 'status': 'yep'}
        self.labelE2.setText(self.progress['state'])
        self.barE1.setValue(int(self.progress['percent']))
        # finished
        if self.progress['percent'] == 100:
            self.buttonE1.setText('继续\nContinue')
            self.buttonE1.clicked.connect(self.e2f)
        return

    def start_training(self):
        self.threadC1.start()

    def start_training_progress(self):
        self.threadC2.start()

    def start_rendering(self):
        self.threadE1.start()

    def start_rendering_progress(self):
        self.threadE2.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
