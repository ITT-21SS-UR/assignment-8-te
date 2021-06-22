import sys
import time
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.flowchart import Flowchart, Node
import pyqtgraph.flowchart.library as fclib
from sklearn.exceptions import NotFittedError
from DIPPID_pyqtnode import BufferNode, DIPPIDNode
import numpy as np
from pylab import *
from scipy.fft import fft
from sklearn import svm

# Authors: tg, ev
# equal workload distribution


class FftNode(Node):
    nodeName = 'FFT'

    def __init__(self, name):
        terminals = {
            'accelX': dict(io='in'),
            'accelY': dict(io='in'),
            'accelZ': dict(io='in'),
            'frequency': dict(io='out'),
        }
        self.frequency = None
        Node.__init__(self, name, terminals=terminals)

    def process(self, **kargs):
        avg = []
        for i in range(len(kargs['accelX'])):
            avg.append((kargs['accelX'][i] + kargs['accelY'][i] + kargs['accelZ'][i]) / 3)
        self.frequency = np.abs(np.fft.fft(avg) / len(avg))[1:len(avg) // 2]
        return {'frequency': self.frequency}


fclib.registerNodeType(FftNode, [('Fft',)])


class SvmNode(Node):
    nodeName = 'SVM'

    TRAINING = 'training'
    PREDICTION = 'prediction'
    INACTIVE = 'inactive'

    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),
            'prediction': dict(io='out'),
        }
        self.predict = ''
        self.activities = []
        self.act_data = {}
        self.reference = {}
        self.mode = self.INACTIVE
        self.recording = False
        self.c = svm.SVC()

        self._init_ui()

        Node.__init__(self, name, terminals=terminals)

    def _init_ui(self):
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout()
        self.mode_layout = QtGui.QGridLayout()
        self.mode_layout.setRowMinimumHeight(0, 50)
        self.mode_layout.setRowMinimumHeight(1, 100)
        self.list_layout = QtGui.QGridLayout()
        self.list_layout.setRowMinimumHeight(0, 50)

        # mode buttons
        self.train_button = QtGui.QPushButton('Training')
        self.train_button.clicked.connect(self.show_training_mode)
        self.mode_layout.addWidget(self.train_button, 0, 0)
        self.pred_button = QtGui.QPushButton('Prediction')
        self.pred_button.clicked.connect(self.show_prediction_mode)
        self.mode_layout.addWidget(self.pred_button, 0, 1)
        self.inactive_button = QtGui.QPushButton('Inactive')
        self.inactive_button.clicked.connect(self.show_inactive_mode)
        self.inactive_button.setDefault(True)
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

    # inits the activity list
    def init_activity(self):
        # activities recorded, delete and retrain activities
        header = QtGui.QLabel('Activities:')
        self.list_layout.addWidget(header, 0, 0)
        add_button = QtGui.QPushButton('Add Activity')
        add_button.clicked.connect(self.show_training_mode)
        self.list_layout.addWidget(add_button, 0, 2)
        if len(self.activities) != 0:
            for i in range(len(self.activities)):
                self.list_layout.setRowMinimumHeight(i + 1, 50)
                activity = QtGui.QLabel(self.activities[i])
                self.list_layout.addWidget(activity, i + 1, 0)
                delete_button = QtGui.QPushButton('Delete')
                delete_button.clicked.connect(lambda state, x=i: self.delete_activity(self.activities[x]))
                self.list_layout.addWidget(delete_button, i + 1, 1)
                retrain_button = QtGui.QPushButton('Retrain')
                retrain_button.clicked.connect(lambda state, x=i: self.retrain_activity(self.activities[x]))
                self.list_layout.addWidget(retrain_button, i + 1, 2)
        else:
            blank = QtGui.QLabel('No activities recorded')
            self.list_layout.addWidget(blank, 1, 1)

    def delete_activity(self, activity):
        # remove activity from ui
        self.activities.remove(activity)
        # https://stackoverflow.com/questions/4528347/clear-all-widgets-in-a-layout-in-pyqt
        for i in reversed(range(self.list_layout.count())):
            self.list_layout.itemAt(i).widget().setParent(None)
        self.init_activity()
        # remove data of activity
        if activity in self.act_data:
            self.act_data.pop(activity, None)
        # remove reference
        if activity in self.reference:
            self.reference.pop(activity, None)

    def retrain_activity(self, activity):
        # setup ui for training
        self.mode = self.TRAINING
        self.selected_button(self.mode)
        self.instructions.setText('Press Button 1 and execute '
                                  'the activity. By releasing Button 1 you stop the current record.\n You can record '
                                  'multiple example of the same activity like that.')
        self.act_name.setText(activity)
        self.act_name.setVisible(True)
        # remove already existing data
        if activity in self.act_data:
            self.act_data.pop(activity, None)

    def show_training_mode(self):
        self.mode = self.TRAINING
        self.selected_button(self.mode)
        self.instructions.setText('Enter name of the activity. Then press Button 1 and execute the activity to train. '
                                  'By releasing Button 1 you stop the\ncurrent record. You can record '
                                  'multiple example of the same activity like that. Record at least 2 activities for '
                                  'prediction.')
        self.act_name.setText('')
        self.act_name.setVisible(True)

    def show_prediction_mode(self):
        self.mode = self.PREDICTION
        self.selected_button(self.mode)
        self.instructions.setText('Press Button 1 and execute an activity. '
                                  'By releasing Button 1 the predicting process starts.')
        self.act_name.setVisible(False)

    def show_inactive_mode(self):
        self.mode = self.INACTIVE
        self.selected_button(self.mode)
        self.instructions.setText('This node is inactive. Choose one of the other '
                                  'two modes to train or predict an activity.')
        self.act_name.setVisible(False)

    # shows which mode is currently selected
    def selected_button(self, mode):
        if mode == self.TRAINING:
            self.train_button.setDefault(True)
            self.pred_button.setDefault(False)
            self.inactive_button.setDefault(False)
        elif mode == self.PREDICTION:
            self.train_button.setDefault(False)
            self.pred_button.setDefault(True)
            self.inactive_button.setDefault(False)
        elif mode == self.INACTIVE:
            self.train_button.setDefault(False)
            self.pred_button.setDefault(False)
            self.inactive_button.setDefault(True)

    def handle_button(self, data):
        if int(data) == 0:
            self.recording = False
        else:
            self.recording = True

    # adds data to dictionary and list if recording and trains the machine with the data when recording has stopped
    def train_activity(self, kargs):
        if self.recording:
            activity_name = str(self.act_name.text())
            if activity_name == '':
                return
            if activity_name not in self.activities:
                self.activities.append(activity_name)
            data = kargs['dataIn']
            if activity_name in self.act_data:
                self.act_data[activity_name].append(data)
            else:
                self.act_data[activity_name] = [data]
        else:
            self.init_activity()
            categories = []
            training_data = []
            i = 0
            for name in self.act_data:
                self.reference[name] = i
                for data in self.act_data[name]:
                    training_data.append(data)
                    categories.append(i)
                i = i + 1
            # train machine on data
            self.c.fit(training_data, categories)

    # predicts the activity when recording, returns if data is not trained
    def predict_activity(self, kargs):
        if self.recording:
            try:
                prediction = self.c.predict([kargs['dataIn']])
            except NotFittedError:
                return
            # sets output text to predicted activity
            for ref in self.reference:
                if self.reference[ref] == prediction[0]:
                    self.predict = ref

    def process(self, **kargs):
        if self.mode == self.TRAINING:
            self.train_activity(kargs)
        elif self.mode == self.PREDICTION:
            self.predict_activity(kargs)
        return {'prediction': self.predict}


fclib.registerNodeType(SvmNode, [('Svm',)])


class DisplayTextNode(Node):
    nodeName = 'PredictText'

    def __init__(self, name):
        terminals = {
            'dataIn': dict(io='in'),
            'prediction': dict(io='out'),
        }
        self._init_ui()
        Node.__init__(self, name, terminals=terminals)

    def _init_ui(self):
        self.ui = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()

        label = QtGui.QLabel("Last Prediction:")
        self.layout.addWidget(label)
        # where prediction is shown, default text if not connected
        self.text = QtGui.QLabel()
        self.standard = 'Connect Dippid to start training and prediction.'
        self.text.setText(self.standard)
        self.layout.addWidget(self.text)

        self.ui.setLayout(self.layout)

    def ctrlWidget(self):
        return self.ui

    def process(self, **kargs):
        prediction = kargs['dataIn']
        self.text.setText(prediction)


fclib.registerNodeType(DisplayTextNode, [('display',)])


def create_flowcharts():
    app = QtGui.QApplication([])
    win = QtGui.QMainWindow()
    win.setWindowTitle('Activity Recognizer')
    win.setMinimumSize(900, 800)
    central = QtGui.QWidget()
    win.setCentralWidget(central)
    layout = QtGui.QGridLayout()
    central.setLayout(layout)

    fc = Flowchart(terminals={'out': dict(io='out')})
    layout.addWidget(fc.widget(), 0, 0, 2, 1)

    # create nodes
    dippid_node = fc.createNode("DIPPID", pos=(0, 0))
    buffer_node_1 = fc.createNode("Buffer", pos=(150, -50))
    buffer_node_2 = fc.createNode("Buffer", pos=(150, 0))
    buffer_node_3 = fc.createNode("Buffer", pos=(150, 50))
    fft_node = fc.createNode("FFT", pos=(300, 0))
    svm_node = fc.createNode("SVM", pos=(450, 0))
    display_node = fc.createNode("PredictText", pos=(600, 0))

    # connect nodes
    fc.connectTerminals(dippid_node['accelX'], buffer_node_1['dataIn'])
    fc.connectTerminals(dippid_node['accelY'], buffer_node_2['dataIn'])
    fc.connectTerminals(dippid_node['accelZ'], buffer_node_3['dataIn'])
    fc.connectTerminals(buffer_node_1['dataOut'], fft_node['accelX'])
    fc.connectTerminals(buffer_node_2['dataOut'], fft_node['accelY'])
    fc.connectTerminals(buffer_node_3['dataOut'], fft_node['accelZ'])
    fc.connectTerminals(fft_node['frequency'], svm_node['dataIn'])
    fc.connectTerminals(svm_node['prediction'], display_node['dataIn'])

    # register callback on dippid_node connect button to get sensor data for svm_node
    dippid_node.connect_button.clicked.connect(lambda: callback(dippid_node, svm_node))

    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit(QtGui.QApplication.instance().exec_())


# callback function on connect button of dippid_node to get sensor for svm_node.
# wait 1 sec before getting sensor and register button in app on same sensor
def callback(d_node, s_node):
    time.sleep(1)
    sensor = d_node.get_sensor()
    sensor.register_callback('button_1', s_node.handle_button)


if __name__ == '__main__':
    create_flowcharts()
