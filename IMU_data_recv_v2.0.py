import sys
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

import matplotlib as mpl
mpl.use("Qt5Agg")
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

IMU_List = ['IMU_1', 'IMU_2', 'IMU_3', 'IMU_4', 'IMU_5', 'IMU_6', 'IMU_7', 'IMU_8', 'IMU_9']
IMU_case_dict = dict(zip(range(1, 10), IMU_List))


# Parameters
Static_ip = '192.168.70.13'
WiFiPort = 2375
PacketSize = 22
quanIMU = 1
quanAngles = quanIMU
plotFPS = 30
is_test_mode = 1



class MyMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        manager = multiprocessing.Manager()
        self.qData = manager.Queue(1024)
        self.readDataBuffer = manager.Queue(1024)
        self.posAngle = [0] * quanAngles
        self.sockets = list()

        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('IMU Postures Monitor')

        self.btnCon.clicked.connect(self.StartConnection)
        self.btnDisCon.clicked.connect(self.Disconnection)
        self.btnCal.clicked.connect(self.Calibration)

        # self.myFig = FigCanvas()
        # self.gridlayout = QtWidgets.QGridLayout(self.gpb_Display)
        # self.gridlayout.addWidget(self.myFig)

        self.dt = DataReceiveThreads()
        # self.dt.postureSignal.newIMUData.connect(self.myFig.data_update)
        # self.dt.postureSignal.postureAngle.connect(self.myFig.angle_update)
        self.dt.postureSignal.sensorData.connect(self.dt.concatenate_data)
        # self.dt.postureSignal.calibrateMatrix.connect(self.myFig.calibrate_update)

        self.multipDataRecv = multiprocessing.Process(target=self.dt.data_recv, args=(self.sockets, self.readDataBuffer,))
        self.threadQuatProcess = threading.Thread(target=self.dt.quat_process, args=(self.readDataBuffer, ))
        self.threadSaveProcess = threading.Thread(target=self.dt.concatenate_data, args=()) 

        self.texbConStatus.append('****** Program is running ******')


    def StartConnection(self):
        connections = 0
        self.texbConStatus.append("Waiting for Connection...")
        self.con = WiFiConnection(msgList=self.texbConStatus)

        while (connections < quanIMU):
            self.sockets.append(self.con.socket_connect())
            connections += 1

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
        self.texbConStatus.append("Data Saving...")
        # self.dt.save_data()
        # self.texbConStatus.append("Data Saved")
        


class FigCanvas(FigureCanvas, FuncAnimation):
    def __init__(self):
        self.rotateMat = dict(zip(IMU_List, [np.array([])] * quanIMU))
        self.calMat = dict(zip(IMU_List, [np.identity(3)] * quanIMU))
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
        if (is_test_mode == 0):    
            

            self.seg_UpperArm_R = np.array([self.seg_Shoulder[1, 0], self.seg_Shoulder[1, 1] - armsLen, self.seg_Shoulder[1, 2]])
            self.seg_LowerArm_R = np.array([self.seg_UpperArm_R[0], self.seg_UpperArm_R[1] - armsLen, self.seg_UpperArm_R[2]])
            self.seg_UpperArm_L = np.array([self.seg_Shoulder[0, 0], self.seg_Shoulder[0, 1] - armsLen, self.seg_Shoulder[0, 2]])
            self.seg_LowerArm_L = np.array([self.seg_UpperArm_L[0], self.seg_UpperArm_L[1] - armsLen, self.seg_UpperArm_L[2]])

            self.seg_Thigh_R = np.array([self.seg_Pelvis[1, 0], self.seg_Pelvis[1, 1] - legLen, self.seg_Pelvis[1, 2]])
            self.seg_Calf_R = np.array([self.seg_Thigh_R[0], self.seg_Thigh_R[1] - legLen, self.seg_Thigh_R[2]])
            self.seg_Thigh_L = np.array([self.seg_Pelvis[0, 0], self.seg_Pelvis[0, 1] - legLen, self.seg_Pelvis[0, 2]])
            self.seg_Calf_L = np.array([self.seg_Thigh_L[0], self.seg_Thigh_L[1] - legLen, self.seg_Thigh_L[2]])

        # Figure setting
        self.labelX = [0] * quanAngles
        self.labelY = [0] * quanAngles
        self.angLabel = []
        self.lines = []
        self.fig = Figure(figsize=(10, 10), dpi=100, tight_layout=True)
        
        FigureCanvas.__init__(self, self.fig)

        self.axes = self.fig.add_subplot(111, projection='3d')
        self.lines.append(self.axes.plot(self.X_Body, self.Y_Body, self.Z_Body, color='seagreen', linewidth=4))
        self.lines.append(self.axes.plot(self.X_Body, self.Y_Body, self.Z_Body, 'o-', color='red', linewidth=2))
        self.lines.append(self.axes.plot(self.X_Body, self.Y_Body, self.Z_Body, 'o-', color='blueviolet', linewidth=2))

        for i in range(quanAngles):
            self.angLabel.append(self.axes.text2D(textInitPos[0], textInitPos[1], ''))

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

            if (is_test_mode == 0):
                vec_UpperArm_R = np.dot(np.dot(self.calMat['IMU_2'], self.rotateMat['IMU_2']), np.array([self.seg_UpperArm_R - self.seg_Shoulder[1]]).T).T + vec_Shoulder[1] #2
                vec_LowerArm_R = np.dot(np.dot(self.calMat['IMU_4'], self.rotateMat['IMU_4']), np.array([self.seg_LowerArm_R - self.seg_UpperArm_R]).T).T + vec_UpperArm_R   #4
                vec_UpperArm_L = np.dot(np.dot(self.calMat['IMU_3'], self.rotateMat['IMU_3']), np.array([self.seg_UpperArm_L - self.seg_Shoulder[0]]).T).T + vec_Shoulder[0] #3
                vec_LowerArm_L = np.dot(np.dot(self.calMat['IMU_5'], self.rotateMat['IMU_5']), np.array([self.seg_LowerArm_L - self.seg_UpperArm_L]).T).T + vec_UpperArm_L   #5

                vec_Thigh_R = np.dot(np.dot(self.calMat['IMU_8'], self.rotateMat['IMU_8']), np.array([self.seg_Thigh_R - self.seg_Pelvis[1]]).T).T + vec_Pelvis[1]   #8
                vec_Calf_R = np.dot(np.dot(self.calMat['IMU_6'], self.rotateMat['IMU_6']), np.array([self.seg_Calf_R - self.seg_Thigh_R]).T).T + vec_Thigh_R         #6
                vec_Thigh_L = np.dot(np.dot(self.calMat['IMU_9'], self.rotateMat['IMU_9']), np.array([self.seg_Thigh_L - self.seg_Pelvis[0]]).T).T + vec_Pelvis[0]   #9
                vec_Calf_L = np.dot(np.dot(self.calMat['IMU_7'], self.rotateMat['IMU_7']), np.array([self.seg_Calf_L - self.seg_Thigh_L]).T).T + vec_Thigh_L         #7

                X_UpperBody = np.array([vec_LowerArm_L[0,0], vec_UpperArm_L[0,0], vec_Shoulder[0,0], vec_Shoulder[1,0], vec_UpperArm_R[0,0], vec_LowerArm_R[0,0]])
                Y_UpperBody = np.array([vec_LowerArm_L[0,1], vec_UpperArm_L[0,1], vec_Shoulder[0,1], vec_Shoulder[1,1], vec_UpperArm_R[0,1], vec_LowerArm_R[0,1]])
                Z_UpperBody = np.array([vec_LowerArm_L[0,2], vec_UpperArm_L[0,2], vec_Shoulder[0,2], vec_Shoulder[1,2], vec_UpperArm_R[0,2], vec_LowerArm_R[0,2]])
                
                X_LowerBody = np.array([vec_Calf_L[0,0], vec_Thigh_L[0,0], vec_Pelvis[0,0], vec_Pelvis[1,0], vec_Thigh_R[0,0], vec_Calf_R[0,0]])
                Y_LowerBody = np.array([vec_Calf_L[0,1], vec_Thigh_L[0,1], vec_Pelvis[0,1], vec_Pelvis[1,1], vec_Thigh_R[0,1], vec_Calf_R[0,1]])
                Z_LowerBody = np.array([vec_Calf_L[0,2], vec_Thigh_L[0,2], vec_Pelvis[0,2], vec_Pelvis[1,2], vec_Thigh_R[0,2], vec_Calf_R[0,2]])

                coorUpperBody = [X_UpperBody.T, Y_UpperBody.T, Z_UpperBody.T]
                coorLowerBody = [X_LowerBody.T, Y_LowerBody.T, Z_LowerBody.T]

            coorTorso = np.dot(np.dot(self.calMat['IMU_1'], self.rotateMat['IMU_1']), [self.X_Body, self.Y_Body, self.Z_Body])

            # Figure update
            self.lines[0][0].set_data(coorTorso[0], coorTorso[1])
            self.lines[0][0].set_3d_properties(coorTorso[2])

            if (is_test_mode == 0):
                self.lines[1][0].set_data(coorUpperBody[0], coorUpperBody[1])
                self.lines[1][0].set_3d_properties(coorUpperBody[2])
                self.lines[2][0].set_data(coorLowerBody[0], coorLowerBody[1])
                self.lines[2][0].set_3d_properties(coorLowerBody[2])

            if (is_test_mode == 0):
                # Angle update
                self.labelX[0], self.labelY[0], _ = proj3d.proj_transform(vec_Shoulder[1, 0], vec_Shoulder[1, 1], vec_Shoulder[1, 2], self.axes.get_proj())
                self.labelX[1], self.labelY[1], _ = proj3d.proj_transform(vec_Shoulder[0, 0], vec_Shoulder[0, 1], vec_Shoulder[0, 2], self.axes.get_proj())  # Right
                self.labelX[2], self.labelY[2], _ = proj3d.proj_transform(vec_UpperArm_R[0, 0], vec_UpperArm_R[0, 1], vec_UpperArm_R[0, 2], self.axes.get_proj())
                self.labelX[3], self.labelY[3], _ = proj3d.proj_transform(vec_UpperArm_L[0, 0], vec_UpperArm_L[0, 1], vec_UpperArm_L[0, 2], self.axes.get_proj())
                self.labelX[4], self.labelY[4], _ = proj3d.proj_transform(vec_Pelvis[1, 0], vec_Pelvis[1, 1], vec_Pelvis[1, 2], self.axes.get_proj())
                self.labelX[5], self.labelY[5], _ = proj3d.proj_transform(vec_Pelvis[0, 0], vec_Pelvis[0, 1], vec_Pelvis[0, 2], self.axes.get_proj()) # Right
                self.labelX[6], self.labelY[6], _ = proj3d.proj_transform(vec_Thigh_R[0, 0], vec_Thigh_R[0, 1], vec_Thigh_R[0, 2], self.axes.get_proj())
                self.labelX[7], self.labelY[7], _ = proj3d.proj_transform(vec_Thigh_L[0, 0], vec_Thigh_L[0, 1], vec_Thigh_L[0, 2], self.axes.get_proj())
            
            for i in range(quanAngles):
                self.angLabel[i].set_position((self.labelX[i], self.labelY[i]))
                self.angLabel[i].set_text(str(self.posAngles[i]))

            self.angleSignal.postureAngle.emit(self.posAngles)
            return tuple(self.lines,)


    def data_update(self, sRoMat):
        self.rotateMat = sRoMat


    def angle_update(self, sAngle):
        self.posAngles = sAngle


    def calibrate_update(self, sCalMat):
        self.calMat = sCalMat



class WiFiConnection():
    def __init__(self, msgList):
        self.msg = msgList
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serverSocket.bind((Static_ip, WiFiPort))
        self.serverSocket.listen(9)


    def get_lan_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('8.8.8.8', 80))
            self.ip = s.getsockname()[0]
        finally:
            s.close()
        return self.ip

    
    def send_to_IMU(self, sockets, text):
        for s in sockets:
            s.send(text)


    def socket_connect(self):
        (clientSocket, addr) = self.serverSocket.accept()
        id_IMU = clientSocket.recv(PacketSize).hex()
        print("Connected from: IMU" + str(int(id_IMU[4:6], 16)))
        clientSocket.setblocking(0)
        return clientSocket


    def recv_close(self, sockets):
        for s in sockets:
            s.close()



# Quaternion Process Functions
class QuaternionProcessFunc():
    def get_sensor_value(self, rawData):
        gryo_data = np.array([0, 0, 0])
        accel_data = np.array([0, 0, 0])
        mag_data = np.array([0, 0, 0])

        try:
            if (rawData[0] == '24' and rawData[-1] == '41'):
                if (int(rawData[1], 16) == 3):
                    for i in [3, 5, 7]:
                        gryo_data[int((i - 3) / 2)] = self.two_byte_to_int(int(rawData[i], 16), int(rawData[i + 1], 16))
                    for i in [9, 11, 13]:
                        accel_data[int((i - 9) / 2)] = self.two_byte_to_int(int(rawData[i], 16), int(rawData[i + 1], 16))
                    for i in [15, 17, 19]:
                        mag_data[int((i - 15) / 2)] = self.two_byte_to_int(int(rawData[i], 16), int(rawData[i + 1], 16))
                    print(rawData)
                    return [gryo_data, accel_data, mag_data]
        except Exception:
            pass


    def two_byte_to_int(self, byte1, byte2):
        num = byte1 * 256 + byte2
        if (num > 32767):
            num = num - 65536
        return num


    def quat_normalized(self, q_raw):
        d = np.linalg.norm(q_raw)
        q_norm = q_raw / d
        return q_norm


    def rotate_matrix(self, quat):
        xx = quat[1]**2
        xy = quat[1]*quat[2]
        xz = quat[1]*quat[3]
        xw = quat[1]*quat[0]
        yy = quat[2]**2
        yz = quat[2]*quat[3]
        yw = quat[2]*quat[0]
        zz = quat[3]**2
        zw = quat[3]*quat[0]
        # ww = quat[0]** 2
        
        rotateMat = np.array(  [[1-2*(yy+zz), 2*(xy - zw), 2*(xz + yw)],
                                [2*(xy + zw), 1-2*(xx+zz), 2*(yz - xw)],
                                [2*(xz - yw), 2*(yz + xw), 1-2*(xx+yy)]])
        return rotateMat



class AngleInfo():
    def __init__(self):
        self.PI = 3.1415926536

        self.init_vec = [0, 1, 0]   # Rotate Vector (Y axis)
        self.bodyVec = dict(zip(IMU_List, [list()] * quanIMU))
        self.posAngles = [0] * quanAngles
        self.vec_norm = np.linalg.norm(self.init_vec)** 2
        self.saveAngles = list()
        
        self.calMat = dict(zip(IMU_List, [np.identity(3)] * quanIMU))
        self.cal_initMat = np.array([[0, 0, -0.9999999999999999], [-0.9999999999999999, 0, 0], [0, 0.9999999999999999, 0]])

        self.posture = [0.0, 0.0, 0.0]
        self.Cb = [0, 0, 0, 0, 0]


    def get_IMU_eular_angle(self, quat):
        
        # r = qp.rotate_matrix(quat)
        # posture[0] = np.rad2deg(np.arctan2(np.sqrt(r[0,2]** 2 + r[1,2]** 2), r[2,2]))   # Pitch
        # posture[1] = -np.rad2deg(np.arctan2(r[2,0],r[2,1]))                             # Roll
        # posture[2] = np.rad2deg(np.arctan2(r[0,2], -r[1,2]))                            # Yaw

        self.Cb[0] = 2 * (quat[0]** 2 + quat[1]** 2) - 1
        self.Cb[1] = 2 * (quat[1] * quat[2] + quat[0] * quat[3])
        self.Cb[2] = 2 * (quat[1] * quat[3] - quat[0] * quat[2])
        self.Cb[3] = 2 * (quat[2] * quat[3] + quat[0] * quat[1])
        self.Cb[4] = 2 * (quat[0]** 2 + quat[3]** 2) - 1
        
        # Pitch
        self.posture[0] = np.arctan2(self.Cb[3], self.Cb[4])
        if(self.posture[0] == self.PI):
            self.posture[0] = -self.PI

        # Roll
        if (self.Cb[2] >= 1):
            self.posture[1] = self.PI/2
        elif (self.Cb[2] <= -1):
            self.posture[1] = self.PI/2
        else:
            self.posture[1] = np.arcsin(-self.Cb[2])

        # Yaw
        self.posture[2] = np.arctan2(self.Cb[1], self.Cb[0])
        if (self.posture[2] < 0):
            self.posture[2] = self.posture[2] + self.PI*2
        elif (self.posture[2] >= self.PI*2):
            self.posture[2] = 0
        
        self.posture[0] = round(np.rad2deg(self.posture[0]), 3)
        self.posture[1] = round(np.rad2deg(self.posture[1]), 3)
        self.posture[2] = round(np.rad2deg(self.posture[2]), 3)
                                 
        return self.posture


    def calculate_calibrate_matrix(self, roMat):
        for m in roMat.keys():
            if(len(roMat[m]) != 0):
                self.calMat[m] = np.dot(self.cal_initMat, np.linalg.inv(roMat[m]))
        return self.calMat


    def get_angle_by_roMat(self, roMat):
        for i in range(quanIMU):
            numOfIMU = IMU_case_dict[i+1]
            self.bodyVec[numOfIMU] = np.dot(self.calMat[numOfIMU], np.dot(roMat[numOfIMU], self.init_vec))

        if (is_test_mode == 0):
            self.posAngles[0] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_1'], self.bodyVec['IMU_2'])/self.vec_norm)), 2)
            self.posAngles[1] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_1'], self.bodyVec['IMU_3'])/self.vec_norm)), 2)
            self.posAngles[2] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_2'], self.bodyVec['IMU_4'])/self.vec_norm)), 2)
            self.posAngles[3] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_3'], self.bodyVec['IMU_5'])/self.vec_norm)), 2)
            self.posAngles[4] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_1'], self.bodyVec['IMU_8'])/self.vec_norm)), 2)
            self.posAngles[5] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_1'], self.bodyVec['IMU_9'])/self.vec_norm)), 2)
            self.posAngles[6] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_8'], self.bodyVec['IMU_6'])/self.vec_norm)), 2)
            self.posAngles[7] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_9'], self.bodyVec['IMU_7'])/self.vec_norm)), 2)
            return self.posAngles



# IMU Data Receive
class DataReceiveThreads():
    def __init__(self):
        self.is_calibrate = False
        self.endRecv = False

        self.postureSignal = Communicate()
        self.savedData_head = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(1, -1)
        self.savedData = self.savedData_head
        # self.saveAngles = np.zeros((1, 8))
        
        # Filename => Timestamp
        ts = time.time()
        dt = datetime.fromtimestamp(ts).strftime("%Y_%m_%d_%H%M")
        self.fileName = "{}.csv".format(dt)
        self.fileDir = "D:/Lab/IMU_Test/IMU recv_Test/sensor_data"


    def data_recv(self, sockets, readDataBuffer):
        time.sleep(0.5)
        self.clear_buffer(sockets)

        while True:
            time.sleep(0.01)
            if(self.is_calibrate):
                self.clear_buffer(sockets)
                self.is_calibrate = False
            for s in sockets:
                readData = []
                try:
                    data = s.recv(PacketSize*2).hex()
                    while (data[0:2] != '24' or data[-2:] != '41'):     # Check input data format
                        print('Recv Error')
                        while len(data) != PacketSize*2:
                            data = data + s.recv(1).hex()
                        data = data[2:]
                        a = s.recv(1).hex()
                        data = data + a

                    for i in range(0, PacketSize*2, 2):
                        readData.append(data[i:i + 2])
                    readDataBuffer.put(readData)
                    self.getData = True

                except Exception:
                    continue
                if not data:
                    sockets.remove(s)
                    break


    def quat_process(self, readDataBuffer):
        # roMat = dict(zip(IMU_List, [np.array([])] * quanIMU))
        # rawData = dict(zip(IMU_List, [list()] * quanIMU))
        # sensorData = dict(zip(IMU_List, [np.array([])] * quanIMU))
        # quatData = dict(zip(IMU_List, [np.array([])] * quanIMU))

        qp = QuaternionProcessFunc()
        # ai = AngleInfo()

        while True:
            if (self.endRecv):
                self.concatenate_data(self.savedData_head)
                break
            if (readDataBuffer.qsize() != 0):
                tempData = readDataBuffer.get(block=False)
                # self.getData = False
                # numOfIMU = IMU_case_dict[int(tempData[2], 16)]      # Recognize which IMU (1-9)
                # rawData[numOfIMU].append(tempData)
                [gyro, accel, mag] = qp.get_sensor_value(tempData)
                # roMat[numOfIMU] = qp.rotate_matrix(quatData[numOfIMU])
                # pry = ai.get_IMU_eular_angle(quatData[numOfIMU])
                # print(pry)
                sensorData= [gyro.tolist(), accel.tolist(), mag.tolist()]
                # print(sensorData)
                print('-' * 10)
                self.postureSignal.sensorData.emit(sensorData)
                # self.concatenate_data(sensorData)
                                    
            # if (check_data_exist(roMat)):
                # if (self.is_calibrate):
                    # calMat = ai.calculate_calibrate_matrix(roMat)
                    # self.is_calibrate = False   # Reset the calibrate flag
                    # self.postureSignal.calibrateMatrix.emit(calMat)
                # posAngle = ai.get_angle_by_roMat(roMat) 
                # self.postureSignal.postureAngle.emit(posAngle)
                # self.postureSignal.newIMUData.emit(roMat)


    def clear_buffer(self, sockets):
        for s in sockets:
            try:
                s.recv(PacketSize*1000).hex()
            except Exception:
                print('Clear Error')
                continue
    

    def concatenate_data(self, data):
        data = np.asarray(data).reshape(1, -1)
        if (len(self.savedData) >= 50 * 2 or self.endRecv == True):       # 50Hz*10s
            self.save_data(self.savedData)
            self.savedData = data
        self.savedData = np.concatenate((self.savedData, data), 0)
    

    def save_data(self, data):
        with open(self.fileDir + '/' + self.fileName, 'a', newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')
            writer.writerows(data)



class Communicate(QtCore.QObject):
    newIMUData = QtCore.pyqtSignal(dict)
    postureAngle = QtCore.pyqtSignal(list)
    calibrateMatrix = QtCore.pyqtSignal(dict)
    sensorData = QtCore.pyqtSignal(list)



def check_data_exist(IMU_Data):
    return (len(IMU_Data.get('IMU_1', '')) or len(IMU_Data.get('IMU_2', '')) or len(IMU_Data.get('IMU_3', '')) or len(IMU_Data.get('IMU_4', '')) or len(IMU_Data.get('IMU_5', '')) or
            len(IMU_Data.get('IMU_6', '')) or len(IMU_Data.get('IMU_7', '')) or len(IMU_Data.get('IMU_8', '')) or len(IMU_Data.get('IMU_9', '')))


  
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
