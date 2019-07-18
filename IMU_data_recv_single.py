# Code by CWM 2019.7.2

import sys
import ekf
import imu
from PyQt5 import  QtWidgets, QtCore
from PyQt5.QtGui import *
from Ui_mainWindow import Ui_MainWindow

import multiprocessing
import threading
import socket
import queue
import math
import time
import csv
import transforms3d as tf3
import numpy as np
from datetime import datetime

import matplotlib as mpl
mpl.use("Qt5Agg")
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


###### Parameters ######
# Wi-Fi
Static_ip = '192.168.70.13'
WiFiPort = 2375

# IMU Setting
quanIMU = 1
quanAngles = quanIMU
plotFPS = 30
halfBody = 1



class MyMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        manager = multiprocessing.Manager()
        self.readDataBuffer = manager.Queue(1024)
        self.sockets = list()
        self.sDataHead = np.asarray([0.0, 0.0, 0.0, 0.0, 0]).reshape(1, -1)

        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('IMU Postures Monitor')

        self.btnCon.clicked.connect(self.StartConnection)
        self.btnDisCon.clicked.connect(self.Disconnection)
        self.btnCal.clicked.connect(self.Calibration)

        self.myFig = FigCanvas()
        self.gridlayout = QtWidgets.QGridLayout(self.gpb_Display)
        self.gridlayout.addWidget(self.myFig)

        self.dt = DataReceiveThreads()
        self.dt.postureSignal.newIMUData.connect(self.myFig.data_update)
        self.dt.postureSignal.postureAngle.connect(self.myFig.angle_update)
        self.dt.postureSignal.calibrateMatrix.connect(self.myFig.calibrate_update)
        self.dt.postureSignal.quatData.connect(self.dt.concatenate_data)

        self.multipDataRecv = multiprocessing.Process(target=self.dt.data_recv, args=(self.sockets, self.readDataBuffer,))
        self.threadQuatProcess = threading.Thread(target=self.dt.quat_process, args=(self.readDataBuffer, ))
        self.threadSaveProcess = threading.Thread(target=self.dt.concatenate_data, args=(self.sDataHead,)) 
        self.texbConStatus.append('****** Program is running ******')


    def StartConnection(self):
        numCon = 0
        self.texbConStatus.append("Waiting for Connections...")
        self.con = imu.WiFiConnection(Static_ip, WiFiPort)

        while (numCon < quanIMU):
            self.sockets.append(self.con.socket_connect())
            numCon += 1

        self.texbConStatus.append("All IMUs are Connected.")
        self.multipDataRecv.start()
        self.threadQuatProcess.start()
        self.threadSaveProcess.start()
        self.texbConStatus.append("Data Receiving...")


    def Calibration(self):
        self.dt.is_calibrate = True


    def Disconnection(self):
        self.multipDataRecv.terminate()
        self.con.recv_close(self.sockets)
        self.dt.endRecv = True
        self.texbConStatus.append("Data Receive Terminated")



class FigCanvas(FigureCanvas, FuncAnimation):
    def __init__(self):
        self.rotateMat = dict(zip(imu.IMU_List, [np.array([])] * quanIMU))
        self.calMat = dict(zip(imu.IMU_List, [np.identity(3)] * quanIMU))
        self.posAngles = [0] * quanAngles
        self.angleSignal = Communicate()
        
        # Plot parameters
        bodyLen = 2

        self.X_Body = [0, 0]
        self.Y_Body = [bodyLen / 2, -bodyLen / 2]
        self.Z_Body = [0, 0]

        # Figure setting
        self.labelX = [0] * 2
        self.labelY = [0] * 2
        self.lines = []
        self.angLabel = []
        self.fig = Figure(figsize=(10, 10), dpi=100, tight_layout=True)
        
        FigureCanvas.__init__(self, self.fig)

        self.axes = self.fig.add_subplot(111, projection='3d')
        self.lines.append(self.axes.plot(self.X_Body, self.Y_Body, self.Z_Body, color='seagreen', linewidth=4))

        self.angLabel.append(self.axes.text2D(0, 0, 'L'))
        self.angLabel.append(self.axes.text2D(0, 0, 'R'))

        self.axes.set_xlim3d([-3.0, 3.0])
        self.axes.set_xlabel('X')
        self.axes.set_ylim3d([-3.0, 3.0])
        self.axes.set_ylabel('Y')
        self.axes.set_zlim3d([-3.0, 3.0])
        self.axes.set_zlabel('Z')
        self.axes.view_init(elev=20, azim=-40)

        FuncAnimation.__init__(self, fig=self.fig, func=self.plot_update, fargs=(), interval=1000 / plotFPS)
        


    def plot_update(self, FuncAnimation):
        if (check_data_exist(self.rotateMat)):
            coorTorso = np.dot(np.dot(self.calMat['IMU_1'], self.rotateMat['IMU_1']), [self.X_Body, self.Y_Body, self.Z_Body])

            # Figure update
            self.lines[0][0].set_data(coorTorso[0], coorTorso[1])
            self.lines[0][0].set_3d_properties(coorTorso[2])

            # Angle update
            self.labelX[0], self.labelY[0], _ = proj3d.proj_transform(coorTorso[0, 0], coorTorso[1, 0], coorTorso[2, 0] + 0.5, self.axes.get_proj())
            self.labelX[1], self.labelY[1], _ = proj3d.proj_transform(coorTorso[0, 1], coorTorso[1, 1], coorTorso[2, 1] + 0.5, self.axes.get_proj())

            for i in range(2): # Display on figure
                self.angLabel[i].set_position((self.labelX[i], self.labelY[i]))

            return tuple(self.lines,)


    def data_update(self, sRoMat):
        self.rotateMat = sRoMat


    def angle_update(self, sAngle):
        self.posAngles = sAngle


    def calibrate_update(self, sCalMat):
        self.calMat = sCalMat



class AngleInfo(ekf.AngleInfo):
    def __init__(self):        
        self.calMat = dict(zip(imu.IMU_List, [np.identity(3)] * quanIMU))
        self.cal_initMat = np.array([[0, 0, -0.9999999999999999], [-0.9999999999999999, 0, 0], [0, 0.9999999999999999, 0]])


    def calculate_calibrate_matrix(self, roMat):
        for m in roMat.keys():
            if(len(roMat[m]) != 0):
                self.calMat[m] = np.dot(self.cal_initMat, np.linalg.inv(roMat[m]))
        return self.calMat


# IMU Data Receive
class DataReceiveThreads():
    def __init__(self):
        self.is_calibrate = False
        self.endRecv = False

        self.postureSignal = Communicate()
        self.sAngles = np.zeros((1, 8))
        self.sDataTerminate = np.asarray([0.0, 0.0, 0.0, 0.0, 0]).reshape(1, -1)
        self.sDataBuffer = self.sDataTerminate
        
        # Filename => Timestamp
        ts = time.time()
        dt = datetime.fromtimestamp(ts).strftime("%Y_%m_%d_%H%M")
        self.fileName = "{}.csv".format(dt)
        self.fileDir = "D:/Lab/IMU Exp/Data"


    def data_recv(self, sockets, readDataBuffer):
        time.sleep(0.5)
        imu.clear_buffer(sockets)

        while True:
            if(self.is_calibrate):
                imu.clear_buffer(sockets)
                self.is_calibrate = False
            for s in sockets:
                try:
                    readData = []
                    data = s.recv(imu.PacketSize).hex()
                    while (data[0:2] != '24' or data[-2:] != '41'):     # Check input data format
                        print('Recv Error')
                        while len(data) != imu.PacketSize*2:
                            data = data + s.recv(1).hex()
                        data = data[2:]
                        a = s.recv(1).hex()
                        data = data + a

                    for i in range(0, imu.PacketSize*2, 2):
                        readData.append(data[i:i + 2])
                    readDataBuffer.put(readData)

                except Exception:
                    continue
                if not data:
                    sockets.remove(s)
                    break


    def quat_process(self, readDataBuffer):
        roMat = dict(zip(imu.IMU_List, [np.array([])] * quanIMU))

        qp = imu.GetSensorsValue()
        kf = ekf.KalmanFilter()
        ai = AngleInfo()

        while True:
            if (self.endRecv):
                self.concatenate_data(self.sDataBuffer)
                break
            if (readDataBuffer.qsize() != 0):
                tempData = readDataBuffer.get(block=False)
                numIMU = 1
                numOfIMU = imu.IMU_case_dict[numIMU]      # Recognize which IMU (1-9)
                [gyro, accel, mag] = qp.get_sensor_value(tempData)

                # print(mag)
                # print('--'*10)

                sensorData = np.asarray([gyro.tolist(), accel.tolist(), mag.tolist()])
                quat = kf.EKF_Update(sensorData[0], sensorData[1], sensorData[2])
                # print(quat)
                roMat[numOfIMU] = tf3.quaternions.quat2mat(quat)
                quat = np.append(quat, numIMU)
                self.postureSignal.quatData.emit(quat)

                if (check_data_exist(roMat)):
                    if (self.is_calibrate):
                        self.is_calibrate = False   # Reset the calibrate flag
                        calMat = ai.calculate_calibrate_matrix(roMat)
                        self.postureSignal.calibrateMatrix.emit(calMat)

                    self.postureSignal.newIMUData.emit(roMat)



    def concatenate_data(self, data):
        data = np.asarray(data).reshape(1, -1)
        if (len(self.sDataBuffer) >= 50 * quanIMU *2 or self.endRecv == True):       # 50Hz*2s
            self.save_data(self.sDataBuffer)
            self.sDataBuffer = data
        self.sDataBuffer = np.concatenate((self.sDataBuffer, data), 0)
    

    def save_data(self, data):
        with open(self.fileDir + '/' + self.fileName, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')
            writer.writerows(data)



class Communicate(QtCore.QObject):
    newIMUData = QtCore.pyqtSignal(dict)
    postureAngle = QtCore.pyqtSignal(list)
    calibrateMatrix = QtCore.pyqtSignal(dict)
    quatData = QtCore.pyqtSignal(np.ndarray)
    sensorData = QtCore.pyqtSignal(list)



def check_data_exist(IMU_Data):
    return (len(IMU_Data.get('IMU_1', '')))

  
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
