# Code by CWM 2019.7.4

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

# IMU setting
quanIMU = 5
quanAngles = quanIMU-1
plotFPS = 30
halfBody = True



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
        self.threadSaveProcess = threading.Thread(target=self.dt.concatenate_data, args=(self.sDataHead, )) 
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


    def Calibration(self):  # if calibration button pressed
        self.dt.is_calibrate = True


    def Disconnection(self):
        self.multipDataRecv.terminate()
        self.con.recv_close(self.sockets)
        self.dt.endRecv = True
        self.texbConStatus.append("Data Receive Terminated.")
        


class FigCanvas(FigureCanvas, FuncAnimation):
    def __init__(self):
        self.rotateMat = dict(zip(imu.IMU_List, [np.array([])] * quanIMU))
        self.calMat = dict(zip(imu.IMU_List, [np.identity(3)] * quanIMU))
        self.posAngles = [0] * quanAngles
        self.angleSignal = Communicate()
        
        # Plot parameters
        bodyLen = 2
        shoulderLen = 1.2
        pelvisLen = 1
        armsLen = 1
        legLen = 1
        textInitPos = [0, 0]

        self.X_Body = [0, 0]
        self.Y_Body = [bodyLen / 2, -bodyLen / 2]
        self.Z_Body = [0, 0]

        self.seg_Shoulder = np.array([[-shoulderLen / 2, self.Y_Body[0], self.Z_Body[0]], [shoulderLen / 2, self.Y_Body[0], self.Z_Body[0]]])
        self.seg_Pelvis = np.array([[-pelvisLen / 2, self.Y_Body[1], self.Z_Body[1]], [pelvisLen / 2, self.Y_Body[1], self.Z_Body[1]]])

        self.seg_UpperArm_R = np.array([self.seg_Shoulder[1, 0], self.seg_Shoulder[1, 1] - armsLen, self.seg_Shoulder[1, 2]])
        self.seg_LowerArm_R = np.array([self.seg_UpperArm_R[0], self.seg_UpperArm_R[1] - armsLen, self.seg_UpperArm_R[2]])
        self.seg_UpperArm_L = np.array([self.seg_Shoulder[0, 0], self.seg_Shoulder[0, 1] - armsLen, self.seg_Shoulder[0, 2]])
        self.seg_LowerArm_L = np.array([self.seg_UpperArm_L[0], self.seg_UpperArm_L[1] - armsLen, self.seg_UpperArm_L[2]])

        if (halfBody == 0):                
            self.seg_Thigh_R = np.array([self.seg_Pelvis[1, 0], self.seg_Pelvis[1, 1] - legLen, self.seg_Pelvis[1, 2]])
            self.seg_Calf_R = np.array([self.seg_Thigh_R[0], self.seg_Thigh_R[1] - legLen, self.seg_Thigh_R[2]])
            self.seg_Thigh_L = np.array([self.seg_Pelvis[0, 0], self.seg_Pelvis[0, 1] - legLen, self.seg_Pelvis[0, 2]])
            self.seg_Calf_L = np.array([self.seg_Thigh_L[0], self.seg_Thigh_L[1] - legLen, self.seg_Thigh_L[2]])

        # Figure setting
        self.labelX = [0] * (quanAngles + 2)
        self.labelY = [0] * (quanAngles + 2)
        self.angLabel = []
        self.lines = []
        self.fig = Figure(figsize=(10, 10), dpi=100, tight_layout=True)
        
        FigureCanvas.__init__(self, self.fig)

        self.axes = self.fig.add_subplot(111, projection='3d')
        self.lines.append(self.axes.plot(self.X_Body, self.Y_Body, self.Z_Body, color='seagreen', linewidth=4))
        self.lines.append(self.axes.plot(self.X_Body, self.Y_Body, self.Z_Body, 'o-', color='red', linewidth=2))

        if (halfBody == 0): 
            self.lines.append(self.axes.plot(self.X_Body, self.Y_Body, self.Z_Body, 'o-', color='blueviolet', linewidth=2))

        for i in range(quanAngles):
            self.angLabel.append(self.axes.text2D(textInitPos[0], textInitPos[1], ''))
        self.angLabel.append(self.axes.text2D(textInitPos[0], textInitPos[1], 'L'))
        self.angLabel.append(self.axes.text2D(textInitPos[0], textInitPos[1], 'R'))

        self.axes.set_xlim3d([-3.0, 3.0])
        self.axes.set_xlabel('X')
        self.axes.set_ylim3d([-3.0, 3.0])
        self.axes.set_ylabel('Y')
        self.axes.set_zlim3d([-3.0, 3.0])
        self.axes.set_zlabel('Z')
        self.axes.view_init(elev=20, azim=-40)

        FuncAnimation.__init__(self, fig=self.fig, func=self.plot_update, fargs=(), interval=1000/plotFPS)


    def plot_update(self, FuncAnimation):
        if (check_data_exist(self.rotateMat)):
            vec_Shoulder = np.dot(np.dot(self.calMat['IMU_1'], self.rotateMat['IMU_1']), self.seg_Shoulder.T).T  #1
            vec_Pelvis = np.dot(np.dot(self.calMat['IMU_1'], self.rotateMat['IMU_1']), self.seg_Pelvis.T).T

            vec_UpperArm_R = np.dot(np.dot(self.calMat['IMU_2'], self.rotateMat['IMU_2']), np.array([self.seg_UpperArm_R - self.seg_Shoulder[1]]).T).T + vec_Shoulder[1] #2
            vec_LowerArm_R = np.dot(np.dot(self.calMat['IMU_4'], self.rotateMat['IMU_4']), np.array([self.seg_LowerArm_R - self.seg_UpperArm_R]).T).T + vec_UpperArm_R   #4
            vec_UpperArm_L = np.dot(np.dot(self.calMat['IMU_3'], self.rotateMat['IMU_3']), np.array([self.seg_UpperArm_L - self.seg_Shoulder[0]]).T).T + vec_Shoulder[0] #3
            vec_LowerArm_L = np.dot(np.dot(self.calMat['IMU_5'], self.rotateMat['IMU_5']), np.array([self.seg_LowerArm_L - self.seg_UpperArm_L]).T).T + vec_UpperArm_L  #5
            
            X_UpperBody = np.array([vec_LowerArm_L[0, 0], vec_UpperArm_L[0, 0], vec_Shoulder[0, 0], vec_Shoulder[1, 0], vec_UpperArm_R[0, 0], vec_LowerArm_R[0, 0]])
            Y_UpperBody = np.array([vec_LowerArm_L[0, 1], vec_UpperArm_L[0, 1], vec_Shoulder[0, 1], vec_Shoulder[1, 1], vec_UpperArm_R[0, 1], vec_LowerArm_R[0, 1]])            
            Z_UpperBody = np.array([vec_LowerArm_L[0, 2], vec_UpperArm_L[0, 2], vec_Shoulder[0, 2], vec_Shoulder[1, 2], vec_UpperArm_R[0, 2], vec_LowerArm_R[0, 2]])
            
            coorUpperBody = [X_UpperBody.T, Y_UpperBody.T, Z_UpperBody.T]

            if (halfBody == 0):
                vec_Thigh_R = np.dot(np.dot(self.calMat['IMU_8'], self.rotateMat['IMU_8']), np.array([self.seg_Thigh_R - self.seg_Pelvis[1]]).T).T + vec_Pelvis[1]  #8
                vec_Calf_R = np.dot(np.dot(self.calMat['IMU_6'], self.rotateMat['IMU_6']), np.array([self.seg_Calf_R - self.seg_Thigh_R]).T).T + vec_Thigh_R        #6
                vec_Thigh_L = np.dot(np.dot(self.calMat['IMU_9'], self.rotateMat['IMU_9']), np.array([self.seg_Thigh_L - self.seg_Pelvis[0]]).T).T + vec_Pelvis[0]  #9
                vec_Calf_L = np.dot(np.dot(self.calMat['IMU_7'], self.rotateMat['IMU_7']), np.array([self.seg_Calf_L - self.seg_Thigh_L]).T).T + vec_Thigh_L        #7
     
                X_LowerBody = np.array([vec_Calf_L[0, 0], vec_Thigh_L[0, 0], vec_Pelvis[0, 0], vec_Pelvis[1, 0], vec_Thigh_R[0, 0], vec_Calf_R[0, 0]])
                Y_LowerBody = np.array([vec_Calf_L[0, 1], vec_Thigh_L[0, 1], vec_Pelvis[0, 1], vec_Pelvis[1, 1], vec_Thigh_R[0, 1], vec_Calf_R[0, 1]])
                Z_LowerBody = np.array([vec_Calf_L[0, 2], vec_Thigh_L[0, 2], vec_Pelvis[0, 2], vec_Pelvis[1, 2], vec_Thigh_R[0, 2], vec_Calf_R[0, 2]])
        
                coorLowerBody = [X_LowerBody.T, Y_LowerBody.T, Z_LowerBody.T]

            coorTorso = np.dot(np.dot(self.calMat['IMU_1'], self.rotateMat['IMU_1']), [self.X_Body, self.Y_Body, self.Z_Body])

            # Figure update
            self.lines[0][0].set_data(coorTorso[0], coorTorso[1])
            self.lines[0][0].set_3d_properties(coorTorso[2])

            self.lines[1][0].set_data(coorUpperBody[0], coorUpperBody[1])
            self.lines[1][0].set_3d_properties(coorUpperBody[2])

            if (halfBody == 0):
                self.lines[2][0].set_data(coorLowerBody[0], coorLowerBody[1])
                self.lines[2][0].set_3d_properties(coorLowerBody[2])

            # Angle update
            self.labelX[0], self.labelY[0], _ = proj3d.proj_transform(vec_Shoulder[1, 0], vec_Shoulder[1, 1], vec_Shoulder[1, 2], self.axes.get_proj())
            self.labelX[1], self.labelY[1], _ = proj3d.proj_transform(vec_Shoulder[0, 0], vec_Shoulder[0, 1], vec_Shoulder[0, 2], self.axes.get_proj())
            self.labelX[2], self.labelY[2], _ = proj3d.proj_transform(vec_UpperArm_R[0, 0], vec_UpperArm_R[0, 1], vec_UpperArm_R[0, 2], self.axes.get_proj())
            self.labelX[3], self.labelY[3], _ = proj3d.proj_transform(vec_UpperArm_L[0, 0], vec_UpperArm_L[0, 1], vec_UpperArm_L[0, 2], self.axes.get_proj())

            self.labelX[-2], self.labelY[-2], _ = proj3d.proj_transform(vec_Shoulder[1, 0], vec_Shoulder[1, 1], vec_Shoulder[1, 2], self.axes.get_proj())
            self.labelX[-1], self.labelY[-1], _ = proj3d.proj_transform(vec_Shoulder[0, 0], vec_Shoulder[0, 1], vec_Shoulder[0, 2], self.axes.get_proj())
                
            if (halfBody == 0):
                self.labelX[4], self.labelY[4], _ = proj3d.proj_transform(vec_Pelvis[1, 0], vec_Pelvis[1, 1], vec_Pelvis[1, 2], self.axes.get_proj())
                self.labelX[5], self.labelY[5], _ = proj3d.proj_transform(vec_Pelvis[0, 0], vec_Pelvis[0, 1], vec_Pelvis[0, 2], self.axes.get_proj())
                self.labelX[6], self.labelY[6], _ = proj3d.proj_transform(vec_Thigh_R[0, 0], vec_Thigh_R[0, 1], vec_Thigh_R[0, 2], self.axes.get_proj())
                self.labelX[7], self.labelY[7], _ = proj3d.proj_transform(vec_Thigh_L[0, 0], vec_Thigh_L[0, 1], vec_Thigh_L[0, 2], self.axes.get_proj())
            
            for i in range(quanAngles + 2): # Display on figure
                self.angLabel[i].set_position((self.labelX[i], self.labelY[i]))
                self.angLabel[i].set_text(str(self.posAngles[i]))

            # self.angleSignal.postureAngle.emit(self.posAngles)
            return tuple(self.lines,)


    def data_update(self, sRoMat):
        self.rotateMat = sRoMat


    def angle_update(self, sAngle):
        self.posAngles = sAngle


    def calibrate_update(self, sCalMat):
        self.calMat = sCalMat



class AngleInfo(ekf.AngleInfo):
    def __init__(self):
        self.init_vec = [0, 1, 0]   # Rotate Vector (Y axis)
        self.bodyVec = dict(zip(imu.IMU_List, [list()] * quanIMU))
        self.posAngles = [0] * quanAngles
        self.vec_norm = np.linalg.norm(self.init_vec)** 2
        
        self.calMat = dict(zip(imu.IMU_List, [np.identity(3)] * quanIMU))
        self.cal_initMat = np.array([[0, 0, -0.9999999999999999], [-0.9999999999999999, 0, 0], [0, 0.9999999999999999, 0]])


    def calculate_calibrate_matrix(self, roMat):
        for m in roMat.keys():
            if(len(roMat[m]) != 0):
                self.calMat[m] = np.dot(self.cal_initMat, np.linalg.inv(roMat[m]))
        return self.calMat


    def get_joint_angles(self, roMat):
        for i in range(quanIMU):
            numOfIMU = imu.IMU_case_dict[i+1]
            self.bodyVec[numOfIMU] = np.dot(self.calMat[numOfIMU], np.dot(roMat[numOfIMU], self.init_vec))

        self.posAngles[0] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_1'], self.bodyVec['IMU_2'])/self.vec_norm)), 2)
        self.posAngles[1] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_1'], self.bodyVec['IMU_3'])/self.vec_norm)), 2)
        self.posAngles[2] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_2'], self.bodyVec['IMU_4'])/self.vec_norm)), 2)
        self.posAngles[3] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_3'], self.bodyVec['IMU_5'])/self.vec_norm)), 2)

        if (halfBody == 0):
            self.posAngles[4] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_1'], self.bodyVec['IMU_8'])/self.vec_norm)), 2)
            self.posAngles[5] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_1'], self.bodyVec['IMU_9'])/self.vec_norm)), 2)
            self.posAngles[6] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_8'], self.bodyVec['IMU_6'])/self.vec_norm)), 2)
            self.posAngles[7] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_9'], self.bodyVec['IMU_7']) / self.vec_norm)), 2)
            
        return self.posAngles



# IMU Data Receive
class DataReceiveThreads():
    def __init__(self):
        self.is_calibrate = False
        self.endRecv = False

        self.readData = np.empty((1, imu.PacketSize*2))
        self.postureSignal = Communicate()
        self.sAngles = np.zeros((1, 8))
        self.sDataTerminate = np.asarray([0.0, 0.0, 0.0, 0.0, 0]).reshape(1, -1)
        self.sDataBuffer = self.sDataTerminate
        
        # Filename => Timestamp
        ts = time.time()
        dt = datetime.fromtimestamp(ts).strftime("%Y_%m_%d_%H%M")
        self.fileName = "{}.csv".format(dt)
        self.fileDir = "D:/Lab/IMU_Test/IMU recv_Test/sensor_data"


    def data_recv(self, sockets, readDataBuffer):
        time.sleep(0.5)
        imu.clear_buffer(sockets)

        while True:
            time.sleep(0.01)
            if(self.is_calibrate):
                imu.clear_buffer(sockets)
                self.is_calibrate = False
            for s in sockets:
                # while(True):
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

        sv = imu.GetSensorsValue()
        ai = AngleInfo()
        self.kf1 = ekf.KalmanFilter()
        self.kf2 = ekf.KalmanFilter()
        self.kf3 = ekf.KalmanFilter()
        self.kf4 = ekf.KalmanFilter()
        self.kf5 = ekf.KalmanFilter()

        if (halfBody == False):
            self.kf6 = ekf.KalmanFilter()
            self.kf7 = ekf.KalmanFilter()
            self.kf8 = ekf.KalmanFilter()
            self.kf9 = ekf.KalmanFilter()

        while True:
            if (self.endRecv):
                self.concatenate_data(self.sDataTerminate)
                break
            if (readDataBuffer.qsize() != 0):
                tempData = readDataBuffer.get(block=False)
                numIMU = int(tempData[2], 16)
                numOfIMU = IMU_case_dict[numIMU]      # Recognize which IMU (1-9)
                [gyro, accel, mag] = sv.get_sensor_value(tempData)
                # print(gyro[0])
                sensorData = np.asarray([gyro.tolist(), accel.tolist(), mag.tolist()])
                quat = self.apply_kalman_filter(sensorData, numIMU)
                roMat[numOfIMU] = tf3.quaternions.quat2mat(quat)
                quat = np.append(quat, numIMU)
                self.postureSignal.quatData.emit(quat)

                if (check_data_exist(roMat)):
                    if (self.is_calibrate):
                        self.is_calibrate = False   # Reset the calibrate flag
                        calMat = ai.calculate_calibrate_matrix(roMat)
                        self.postureSignal.calibrateMatrix.emit(calMat)
                        
                    posAngle = ai.get_joint_angles(roMat) 
                    self.postureSignal.postureAngle.emit(posAngle)
                    self.postureSignal.newIMUData.emit(roMat)


    def apply_kalman_filter(self, sensorData, numIMU):
        if (numIMU == 1):
            quat = self.kf1.EKF_Update(sensorData[0], sensorData[1], sensorData[2])
        elif (numIMU == 2):
            quat = self.kf2.EKF_Update(sensorData[0], sensorData[1], sensorData[2])
        elif (numIMU == 3):
            quat = self.kf3.EKF_Update(sensorData[0], sensorData[1], sensorData[2])
        elif (numIMU == 4):
            quat = self.kf4.EKF_Update(sensorData[0], sensorData[1], sensorData[2])
        elif (numIMU == 5):
            quat = self.kf5.EKF_Update(sensorData[0], sensorData[1], sensorData[2])
        elif (numIMU == 6):
            quat = self.kf6.EKF_Update(sensorData[0], sensorData[1], sensorData[2])
        elif (numIMU == 7):
            quat = self.kf7.EKF_Update(sensorData[0], sensorData[1], sensorData[2])
        elif (numIMU == 8):
            quat = self.kf8.EKF_Update(sensorData[0], sensorData[1], sensorData[2])
        elif (numIMU == 9):
            quat = self.kf9.EKF_Update(sensorData[0], sensorData[1], sensorData[2])
        return quat
    

    def concatenate_data(self, data):
        data = np.asarray(data).reshape(1, -1)
        if (len(self.sDataBuffer) >= 50 * quanIMU *2 or self.endRecv == True):       # 50Hz*2s
            self.save_data(self.sDataBuffer)
            self.sDataBuffer = data
        # print(data)
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
    if (halfBody == 1):
        return (len(IMU_Data.get('IMU_1', '')) and len(IMU_Data.get('IMU_2', '')) and len(IMU_Data.get('IMU_3', '')) and len(IMU_Data.get('IMU_4', '')) and len(IMU_Data.get('IMU_5', '')) or
                len(IMU_Data.get('IMU_6', '')) or len(IMU_Data.get('IMU_7', '')) or len(IMU_Data.get('IMU_8', '')) or len(IMU_Data.get('IMU_9', '')))
    else:
        for i in IMU_Data.keys():
            if (len(IMU_Data.get(i, '')) == 0):
                return False
        return True


  
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
