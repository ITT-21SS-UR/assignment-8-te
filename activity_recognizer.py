import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib

from DIPPID import SensorUDP
from DIPPID_pyqtnode import BufferNode, DIPPIDNode
import numpy as np
from pylab import *
from scipy.fft import fft
from sklearn import svm


class FftNode(Node):

    nodeName = 'Fft'

    def __init__(self, name):
        terminals = {
            'accelX': dict(io='in'),
            'accelY': dict(io='in'),
            'accelZ': dict(io='in'),
            'frequency': dict(io='out'),
        }
        self.data_x = np.array([])
        self.data_y = np.array([])
        self.data_z = np.array([])
        self.frequency_x = []
        self.frequency_y = []
        self.frequency_z = []
        Node.__init__(self, name, terminals=terminals)

    def process(self, **kargs):
        #print(kargs['accelX'])
        #print(kargs['accelY'])
        #print(kargs['accelZ'])
        #self.data_x = np.append(self.data_x, kargs['accelX'])
        #self.data_y = np.append(self.data_y, kargs['accelY'])
        #self.data_z = np.append(self.data_z, kargs['accelZ'])
        #Y = fft.fft(y) / n  # fft computing and normalization\n",
        #Y = Y[0:int(n/2)]
        avg = []
        for i in range(len(kargs['accelX'])):
            avg.append((kargs['accelX'][i] + kargs['accelY'][i] + kargs['accelZ'][i])/3)
        # print(avg)
        frequency = np.abs(np.fft.fft(avg)/len(avg))[1:len(avg)//2]
        # print(frequency)
        self.frequency_x = np.abs(np.fft.fft(kargs['accelX'])/2)
        self.frequency_y = np.abs(np.fft.fft(kargs['accelY'])/2)
        self.frequency_z = np.abs(np.fft.fft(kargs['accelZ'])/2)
        #self.frequency_x = [np.abs(fft(l) / len(l))[1:len(l) // 2] for l in kargs['accelX']]
        #self.frequency_y = [np.abs(fft(l) / len(l))[1:len(l) // 2] for l in kargs['accelY']]
        #self.frequency_z = [np.abs(fft(l) / len(l))[1:len(l) // 2] for l in kargs['accelZ']]
        #print('dx: ', self.data_x, ' dy: ', self.data_y, ' dz: ', self.data_z)
        # print('x: ', self.frequency_x, ' y: ', self.frequency_y, ' z: ', self.frequency_z)
        return {'frequency': frequency}


fclib.registerNodeType(FftNode, [('Fft',)])


class SvmNode(Node):

    nodeName = 'Svm'

    TRAINING = 'training'
    PREDICTION = 'prediction'
    INACTIVE = 'inactive'
    # different modes (training, prediction, inactive)
    # ports dependent on mode
    # prediciton: in: sample, out: prediction
    # training: in: list of frequncy, train data, out:? (current recognition?) category as textfield

    # when train active, press button 1 to start motion, as long as same name it trains, adds activity when name and
    # button 1 is pressed first.

    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),
            'prediction': dict(io='out'),
        }
        self.predict = ''
        self.activities = []  # ['jump', 'work', 'walk', 'stand', 'hop']
        self.act_data = {}
        self.mode = self.INACTIVE
        self.recording = False
        self.c = svm.SVC()

        self._init_ui()

        # self.timer = QtCore.QTimer()
        # self.timer.timeout().connect(self.handle_input)
        # self.timer.start(100)
        Node.__init__(self, name, terminals=terminals)

    def _init_ui(self):
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout()
        self.mode_layout = QtGui.QGridLayout()
        self.list_layout = QtGui.QGridLayout()

        # mode buttons
        self.train_button = QtGui.QPushButton('Training')
        self.train_button.clicked.connect(self.show_training_mode)
        self.mode_layout.addWidget(self.train_button, 0, 0)
        self.pred_button = QtGui.QPushButton('Prediction')
        self.pred_button.clicked.connect(self.show_prediction_mode)
        self.mode_layout.addWidget(self.pred_button, 0, 1)
        self.inactive_button = QtGui.QPushButton('Inactive')
        self.inactive_button.clicked.connect(self.show_inactive_mode)
        self.mode_layout.addWidget(self.inactive_button, 0, 2)

        # instructions and name tag
        self.instructions = QtGui.QLabel()
        self.instructions.setText('This node is inactive. Choose one of the other '
                                  'two modes to train or predict an activity.')
        self.mode_layout.addWidget(self.instructions, 1, 0, 3, 3)
        self.act_name = QtGui.QLineEdit()
        self.act_name.setVisible(False)
        self.mode_layout.addWidget(self.act_name, 5, 0, 2, 2)

        self.init_activity()

        self.layout.addLayout(self.mode_layout)
        self.layout.addLayout(self.list_layout)
        self.ui.setLayout(self.layout)

    def ctrlWidget(self):
        return self.ui

    def init_activity(self):
        # activities recorded, delete and retrain activities
        header = QtGui.QLabel('Activities:')
        self.list_layout.addWidget(header, 7, 0)
        add_button = QtGui.QPushButton('Add Activity')
        add_button.clicked.connect(self.show_training_mode)
        self.list_layout.addWidget(add_button, 7, 2)
        if len(self.activities) != 0:
            for i in range(len(self.activities)):
                activity = QtGui.QLabel(self.activities[i])
                self.list_layout.addWidget(activity, i + 8, 0)
                delete_button = QtGui.QPushButton('Delete')
                delete_button.clicked.connect(lambda state, x=i: self.delete_activity(self.activities[x]))
                self.list_layout.addWidget(delete_button, i + 8, 1)
                retrain_button = QtGui.QPushButton('Retrain')
                retrain_button.clicked.connect(lambda state, x=i: self.retrain_activity(self.activities[x]))
                self.list_layout.addWidget(retrain_button, i + 8, 2)
        else:
            empty = QtGui.QLabel('No activities recorded')
            self.list_layout.addWidget(empty, 8, 1)

    def delete_activity(self, activity):
        self.activities.remove(activity)
        # https://stackoverflow.com/questions/4528347/clear-all-widgets-in-a-layout-in-pyqt
        for i in reversed(range(self.list_layout.count())):
            self.list_layout.itemAt(i).widget().setParent(None)
        self.init_activity()

    def retrain_activity(self, activity):
        self.mode = self.TRAINING
        self.instructions.setText('Press Button 1 and execute '
                                  'the activity. By releasing Button 1 you stop the current record.\n You can record '
                                  'multiple example of the same activity like that.')
        self.act_name.setText(activity)
        self.act_name.setVisible(True)

    def show_training_mode(self):
        self.mode = self.TRAINING
        self.instructions.setText('Enter name of the activity you want to train. Then press Button 1 and execute '
                                  'the activity. By releasing\nButton 1 you stop the current record. You can record '
                                  'multiple example of the same activity like that.')
        self.act_name.setText('')
        self.act_name.setVisible(True)

    def show_prediction_mode(self):
        self.mode = self.PREDICTION
        self.instructions.setText('Press Button 1 and execute an activity. '
                                  'By releasing Button 1 the predicting process starts.')
        self.act_name.setVisible(False)

    def show_inactive_mode(self):
        self.mode = self.INACTIVE
        self.instructions.setText('This node is inactive. Choose one of the other '
                                  'two modes to train or predict an activity.')
        self.act_name.setVisible(False)

    def handle_button(self, data):
        if int(data) == 0:
            self.recording = False
        else:
            self.recording = True

    def train_activity(self, kargs):
        activity_name = self.act_name.getText()
        if activity_name == '':
            return
        self.activities.append(activity_name)

        if self.recording:
            data = kargs['dataIn']
            self.act_data[activity_name].append(data)
            print(self.act_data)
        else:
            print('work')
            self.init_activity()
            categories = self.activities
            # training_data = stand_freq[1:] + walk_freq[1:] + hop_freq[1:]
            # self.c.fit(training_data, categories)

    def predict_activity(self, kargs):
        self.predict = self.c.predict(kargs['dataIn'])
        print('prediction: ' , self.predict)

    def process(self, **kargs):
        if self.mode == self.TRAINING:
            self.train_activity(kargs)
        elif self.mode == self.PREDICTION:
            self.predict_activity(kargs)
        return {'prediction': self.predict}


fclib.registerNodeType(SvmNode, [('Svm',)])


class DisplayTextNode(Node):

    nodeName = 'text'
    # displays text on screen

    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),
            'prediction': dict(io='out'),
        }
        # self.text = ''
        self._init_ui()
        Node.__init__(self, name, terminals=terminals)

    def _init_ui(self):
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()

        label = QtGui.QLabel("Prediction:")
        self.layout.addWidget(label)

        self.text = QtGui.QLabel()
        self.addr = "5700"
        self.text.setText(self.addr)
        self.layout.addWidget(self.text)

        self.ui.setLayout(self.layout)

    def ctrlWidget(self):
        return self.ui

    def process(self, **kargs):
        pred = kargs['dataIn'][0]
        pred = 'Help'
        self.text.setText(pred)
        # return {'prediction': self.text}


fclib.registerNodeType(DisplayTextNode, [('display',)])


def create_flowcharts():
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.setWindowTitle('Activity Recognizer')
    win.setMinimumSize(900,800)
    central = QtGui.QWidget()
    win.setCentralWidget(central)
    layout = QtGui.QGridLayout()
    central.setLayout(layout)

    fc = Flowchart(terminals={'out': dict(io='out')})
    layout.addWidget(fc.widget(), 0, 0, 2, 1)

    dippid_node = fc.createNode("DIPPID", pos=(0, 0))
    buffer_node_1 = fc.createNode("Buffer", pos=(150, -50))
    buffer_node_2 = fc.createNode("Buffer", pos=(150, 0))
    buffer_node_3 = fc.createNode("Buffer", pos=(150, 50))

    fft_node = fc.createNode("Fft", pos=(300, 0))
    svm_node = fc.createNode("Svm", pos=(450, 0))
    display_node = fc.createNode("text", pos=(600, 0))
    pw1 = pg.PlotWidget()
    layout.addWidget(pw1, 0, 1)
    pw1.setYRange(0, 1)

    pw1Node = fc.createNode('PlotWidget', pos=(0, -150))
    pw1Node.setPlot(pw1)

    fc.connectTerminals(dippid_node['accelX'], buffer_node_1['dataIn'])
    fc.connectTerminals(dippid_node['accelY'], buffer_node_2['dataIn'])
    fc.connectTerminals(dippid_node['accelZ'], buffer_node_3['dataIn'])
    fc.connectTerminals(buffer_node_1['dataOut'], fft_node['accelX'])
    fc.connectTerminals(buffer_node_2['dataOut'], fft_node['accelY'])
    fc.connectTerminals(buffer_node_3['dataOut'], fft_node['accelZ'])
    fc.connectTerminals(fft_node['frequency'], pw1Node['In'])
    #fc.connectTerminals(fft_node['frequency'], svm_node['dataIn'])
    #fc.connectTerminals(svm_node['prediction'], display_node['dataIn'])

    #sensor = SensorUDP(5700)
    #sensor.register_callback('button_1', svm_node.handle_button)

    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(QtGui.QApplication.instance().exec_())


if __name__ == '__main__':
    create_flowcharts()
