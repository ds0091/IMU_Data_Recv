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
import csv

# Parameters
Static_ip = '192.168.70.13'
WiFiPort = 2375
BufferSize = 12
quaIMU = 5
quaAngles = quaIMU-1
plotFPS = 30

IMU_List = ['IMU_1', 'IMU_2', 'IMU_3', 'IMU_4', 'IMU_5', 'IMU_6', 'IMU_7', 'IMU_8', 'IMU_9']
IMU_case_dict = dict(zip(range(1,10), IMU_List))

class MyMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        manager = multiprocessing.Manager()
        self.qData = manager.Queue(1024)
        self.readDataBuffer = manager.Queue(1024)
        self.sockets = list()

        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('IMU Postures Monitor')

        self.btnCon.clicked.connect(self.StartConnection)
        self.btnDisCon.clicked.connect(self.Disconnection)
        self.btnCal.clicked.connect(self.Calibration)

        self.myFig = FigCanvas()
        self.gridlayout = QtWidgets.QGridLayout(self.gpb_Display)
        self.gridlayout.addWidget(self.myFig)
        self.texbConStatus.append('****** Program is running ******')


    def StartConnection(self):
        self.connections = 0
        self.texbConStatus.append("Waiting for Connection...")
        self.con = WiFiConnection(msgList=self.texbConStatus)
        # self.texbConStatus.append("Current IP: ")
        # self.texbConStatus.insertPlainText(con.get_lan_ip())

        while (self.connections < quaIMU):
            self.sockets.append(self.con.socket_connect())
            self.connections += 1

        self.dt = DataReceiveThreads()
        self.dt.postureSignal.newIMUData.connect(self.myFig.data_update)
        self.dt.postureSignal.postureAngle.connect(self.myFig.angle_update)
        self.dt.postureSignal.calibrateMatrix.connect(self.myFig.calibrate_update)

        self.multipDataRecv = multiprocessing.Process(target=self.dt.data_recv, args=(self.sockets, self.qData, ))
        self.multipReadData = multiprocessing.Process(target=self.dt.read_IMU_data, args=(self.qData, self.readDataBuffer, ))
        self.threadQuatProcess = threading.Thread(target=self.dt.quat_process, args=(self.readDataBuffer, ))
    
        self.multipDataRecv.start()
        self.multipReadData.start()
        self.threadQuatProcess.start()
        self.texbConStatus.append("Data Receiving...")


    def Calibration(self):
        self.dt.is_calibrate = 1


    def Disconnection(self):
        self.multipDataRecv.terminate()
        self.multipReadData.terminate()
        self.con.recv_close(self.sockets)
        self.texbConStatus.append("Data Receive Terminated")



class FigCanvas(FigureCanvas, FuncAnimation):
    def __init__(self):
        self.rotateMat = dict(zip(IMU_List, [np.array([])] * quaIMU))
        self.calMat = dict(zip(IMU_List, [np.identity(3)] * quaIMU))
        self.posAngles = [0] * quaAngles
        
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

        self.seg_Thigh_R = np.array([self.seg_Pelvis[1, 0], self.seg_Pelvis[1, 1] - legLen, self.seg_Pelvis[1, 2]])
        self.seg_Calf_R = np.array([self.seg_Thigh_R[0], self.seg_Thigh_R[1] - legLen, self.seg_Thigh_R[2]])
        self.seg_Thigh_L = np.array([self.seg_Pelvis[0, 0], self.seg_Pelvis[0, 1] - legLen, self.seg_Pelvis[0, 2]])
        self.seg_Calf_L = np.array([self.seg_Thigh_L[0], self.seg_Thigh_L[1] - legLen, self.seg_Thigh_L[2]])

        # Figure setting
        self.labelX = [0] * quaAngles
        self.labelY = [0] * quaAngles
        self.angLabel = []
        self.lines = []
        self.fig = Figure(figsize=(10, 10), dpi=100, tight_layout=True)
        
        FigureCanvas.__init__(self, self.fig)

        self.axes = self.fig.add_subplot(111, projection='3d')
        self.lines.append(self.axes.plot(self.X_Body, self.Y_Body, self.Z_Body, color='seagreen', linewidth=4))
        self.lines.append(self.axes.plot(self.X_Body, self.Y_Body, self.Z_Body, 'o-', color='red', linewidth=2))
        self.lines.append(self.axes.plot(self.X_Body, self.Y_Body, self.Z_Body, 'o-', color='blueviolet', linewidth=2))

        for i in range(quaAngles):
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

            vec_UpperArm_R = np.dot(np.dot(self.calMat['IMU_2'], self.rotateMat['IMU_2']), np.array([self.seg_UpperArm_R - self.seg_Shoulder[1]]).T).T + vec_Shoulder[1] #2
            vec_LowerArm_R = np.dot(np.dot(self.calMat['IMU_4'], self.rotateMat['IMU_4']), np.array([self.seg_LowerArm_R - self.seg_UpperArm_R]).T).T + vec_UpperArm_R   #4
            vec_UpperArm_L = np.dot(np.dot(self.calMat['IMU_3'], self.rotateMat['IMU_3']), np.array([self.seg_UpperArm_L - self.seg_Shoulder[0]]).T).T + vec_Shoulder[0] #3
            vec_LowerArm_L = np.dot(np.dot(self.calMat['IMU_5'], self.rotateMat['IMU_5']), np.array([self.seg_LowerArm_L - self.seg_UpperArm_L]).T).T + vec_UpperArm_L   #5

            vec_Thigh_R = np.dot(np.dot(self.calMat['IMU_1'], self.rotateMat['IMU_1']), np.array([self.seg_Thigh_R - self.seg_Pelvis[1]]).T).T + vec_Pelvis[1]   #6
            vec_Calf_R = np.dot(np.dot(self.calMat['IMU_1'], self.rotateMat['IMU_1']), np.array([self.seg_Calf_R - self.seg_Thigh_R]).T).T + vec_Thigh_R         #8
            vec_Thigh_L = np.dot(np.dot(self.calMat['IMU_1'], self.rotateMat['IMU_1']), np.array([self.seg_Thigh_L - self.seg_Pelvis[0]]).T).T + vec_Pelvis[0]   #7
            vec_Calf_L = np.dot(np.dot(self.calMat['IMU_1'], self.rotateMat['IMU_1']), np.array([self.seg_Calf_L - self.seg_Thigh_L]).T).T + vec_Thigh_L         #9

            X_UpperBody = np.array([vec_LowerArm_L[0,0], vec_UpperArm_L[0,0], vec_Shoulder[0,0], vec_Shoulder[1,0], vec_UpperArm_R[0,0], vec_LowerArm_R[0,0]])
            Y_UpperBody = np.array([vec_LowerArm_L[0,1], vec_UpperArm_L[0,1], vec_Shoulder[0,1], vec_Shoulder[1,1], vec_UpperArm_R[0,1], vec_LowerArm_R[0,1]])
            Z_UpperBody = np.array([vec_LowerArm_L[0,2], vec_UpperArm_L[0,2], vec_Shoulder[0,2], vec_Shoulder[1,2], vec_UpperArm_R[0,2], vec_LowerArm_R[0,2]])
            
            X_LowerBody = np.array([vec_Calf_L[0,0], vec_Thigh_L[0,0], vec_Pelvis[0,0], vec_Pelvis[1,0], vec_Thigh_R[0,0], vec_Calf_R[0,0]])
            Y_LowerBody = np.array([vec_Calf_L[0,1], vec_Thigh_L[0,1], vec_Pelvis[0,1], vec_Pelvis[1,1], vec_Thigh_R[0,1], vec_Calf_R[0,1]])
            Z_LowerBody = np.array([vec_Calf_L[0,2], vec_Thigh_L[0,2], vec_Pelvis[0,2], vec_Pelvis[1,2], vec_Thigh_R[0,2], vec_Calf_R[0,2]])

            coorTorso = np.dot(np.dot(self.calMat['IMU_1'], self.rotateMat['IMU_1']), [self.X_Body, self.Y_Body, self.Z_Body])
            coorUpperBody = [X_UpperBody.T, Y_UpperBody.T, Z_UpperBody.T]
            coorLowerBody = [X_LowerBody.T, Y_LowerBody.T, Z_LowerBody.T]

            # Figure update
            self.lines[0][0].set_data(coorTorso[0], coorTorso[1])
            self.lines[0][0].set_3d_properties(coorTorso[2])
            self.lines[1][0].set_data(coorUpperBody[0], coorUpperBody[1])
            self.lines[1][0].set_3d_properties(coorUpperBody[2])
            self.lines[2][0].set_data(coorLowerBody[0], coorLowerBody[1])
            self.lines[2][0].set_3d_properties(coorLowerBody[2])

            # Angle update
            self.labelX[0], self.labelY[0], _ = proj3d.proj_transform(vec_Shoulder[1, 0], vec_Shoulder[1, 1], vec_Shoulder[1, 2], self.axes.get_proj())
            self.labelX[1], self.labelY[1], _ = proj3d.proj_transform(vec_Shoulder[0, 0], vec_Shoulder[0, 1], vec_Shoulder[0, 2], self.axes.get_proj()) # Right
            self.labelX[2], self.labelY[2], _ = proj3d.proj_transform(vec_UpperArm_R[0, 0], vec_UpperArm_R[0, 1], vec_UpperArm_R[0, 2], self.axes.get_proj())
            self.labelX[3], self.labelY[3], _ = proj3d.proj_transform(vec_UpperArm_L[0, 0], vec_UpperArm_L[0, 1], vec_UpperArm_L[0, 2], self.axes.get_proj())
            
            for i in range(quaAngles):
                self.angLabel[i].set_position((self.labelX[i], self.labelY[i]))
                self.angLabel[i].set_text(str(self.posAngles[i]) + ' deg')
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
        id_IMU = clientSocket.recv(BufferSize).hex()
        print("Connected from: IMU" + str(int(id_IMU[4:6], 16)))
        # self.msg.append("Connected from: IMU")
        # self.msg.insertPlainText(str(int(id_IMU[4:6], 16)))
        clientSocket.setblocking(0)
        return clientSocket


    def recv_close(self, sockets):
        for s in sockets:
            s.close()



# Quaternion Process Functions
class QuaternionProcessFunc():
    def get_quat(self, rawData):
        q_raw = np.array([2, 2, 2, 2])
        if (rawData[0] == '24' and rawData[11] == '00'):
            if (int(rawData[1], 16) == 3):
                for i in [3, 5, 7, 9]:
                    q_raw[int((i-3)/2)] = self.two_byte_to_int(int(rawData[i], 16), int(rawData[i+1], 16))
                return self.quat_normalized(q_raw)
            elif (int(rawData[1], 16) == 1):
                q_raw[1] = self.IMU_correction(int(rawData[3], 16), int(rawData[4], 16))
                return q_raw


    def two_byte_to_int(self, byte1, byte2):
        num = byte1 * 256 + byte2
        if (num > 32767):
            num = num - 65536
        return num


    def IMU_correction(self, byte1, byte2):
        a = self.two_byte_to_int(byte1, byte2)/10
        if (a > 0):
            num = a - 180
        else:
            num = a + 180
        return num


    def quat_normalized(self, q_raw):
        d = np.linalg.norm(q_raw)
        q = q_raw / d
        return q


    def rotate_matrix(self, q1):
        xx = q1[1]**2
        xy = q1[1]*q1[2]
        xz = q1[1]*q1[3]
        xw = q1[1]*q1[0]
        yy = q1[2]**2
        yz = q1[2]*q1[3]
        yw = q1[2]*q1[0]
        zz = q1[3]**2
        zw = q1[3]*q1[0]
        ww = q1[0]**2
        rotateMat = np.array(  [[ww+xx-yy-zz, 2*(xy - zw), 2*(xz + yw)],
                                [2*(xy + zw), ww-xx+yy-zz, 2*(yz - xw)],
                                [2*(xz - yw), 2*(yz + xw), ww-xx-yy+zz]])
        return rotateMat



class AngleInfo():
    def __init__(self):
        self.init_vec = [0, 1, 0]   # Rotate Vector (Y axis)
        self.bodyVec = dict(zip(IMU_List, [list()] * quaIMU))
        self.posAngles = [0] * quaAngles
        self.vec_norm = np.linalg.norm(self.init_vec)** 2
        
        self.calMat = dict(zip(IMU_List, [np.identity(3)] * quaIMU))
        self.cal_initMat = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])


    def get_IMU_eular_angle(self, quat, qp):
        posture = np.array([0.0, 0.0, 0.0])
        r = qp.rotate_matrix(quat)
        posture[0] = np.rad2deg(np.arctan2(np.sqrt(r[0,2]** 2 + r[1,2]** 2), r[2,2]))   # Pitch
        posture[1] = -np.rad2deg(np.arctan2(r[2,0],r[2,1]))                             # Roll
        posture[2] = np.rad2deg(np.arctan2(r[0,2], -r[1,2]))                            # Yaw
        return posture


    def calculate_calibrate_matrix(self, roMat):
        for m in roMat.keys():
            if(len(roMat[m]) != 0):
                self.calMat[m] = np.dot(self.cal_initMat, np.linalg.inv(roMat[m]))
        return self.calMat


    def get_angle_by_roMat(self, roMat):
        for i in range(quaIMU):
            numOfIMU = IMU_case_dict[i+1]
            self.bodyVec[numOfIMU] = np.dot(self.calMat[numOfIMU], np.dot(roMat[numOfIMU], self.init_vec))

        # print(np.inner(self.bodyVec['IMU_3'], self.bodyVec['IMU_5'])/self.vec_norm)
        self.posAngles[0] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_1'], self.bodyVec['IMU_2'])/self.vec_norm)), 2)
        self.posAngles[1] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_1'], self.bodyVec['IMU_3'])/self.vec_norm)), 2)
        self.posAngles[2] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_2'], self.bodyVec['IMU_4'])/self.vec_norm)), 2)
        self.posAngles[3] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_3'], self.bodyVec['IMU_5'])/self.vec_norm)), 2)
        # self.posAngles[4] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_1'], self.bodyVec['IMU_6'])/self.vec_norm)), 2)
        # self.posAngles[5] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_1'], self.bodyVec['IMU_7'])/self.vec_norm)), 2)
        # self.posAngles[6] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_6'], self.bodyVec['IMU_8'])/self.vec_norm)), 2)
        # self.posAngles[7] = round(math.degrees(math.acos(np.inner(self.bodyVec['IMU_7'], self.bodyVec['IMU_9'])/self.vec_norm)), 2)
        return self.posAngles



# IMU Data Receive
class DataReceiveThreads():
    def __init__(self):
        self.postureSignal = Communicate()
        self.is_calibrate = 0

        ts = time.time()
        dt = datetime.fromtimestamp(ts).strftime("%Y_%m_%d_%H%M")
        filename = "{}.csv".format(dt)
        self.fileName = filename
        self.fileDir = "D:\\Lab\\IMU recv"


    def data_recv(self, sockets, qData):
        time.sleep(1)
        for s in sockets:
            try:
                data = s.recv(32767).hex()
            except Exception:
                print('Clear Error')
                continue

        while True:
            # time.sleep(0.01)
            for s in sockets:
                try:
                    data = s.recv(BufferSize).hex()
                    while (data[0:2] != '24' or data[-2:] != '00'):     # Check input data format
                        print('Recv Error')
                        while len(data) != BufferSize*2:
                            data = data + s.recv(1).hex()
                        data = data[2:]
                        a = s.recv(1).hex()
                        data = data + a
                    qData.put(data)

                except Exception:
                    # print(str(s)+'Err')
                    continue
                if not data:
                    sockets.remove(s)
                    break


    def read_IMU_data(self, qData, readDataBuffer):
        while True:
            readData = []
            if (qData.qsize() != 0):
                tempData = qData.get(block=False)
                for i in range(0, BufferSize*2, 2):
                    readData.append(tempData[i:i + 2])
                readDataBuffer.put(readData)


    def quat_process(self, readDataBuffer):
        roMat = dict(zip(IMU_List, [np.array([])] * quaIMU))
        rawData = dict(zip(IMU_List, [list()] * quaIMU))
        quatData = dict(zip(IMU_List, [np.array([])] * quaIMU))
        posAngle = [0]*quaAngles

        qp = QuaternionProcessFunc()
        ai = AngleInfo()
        

        while True:
            if (readDataBuffer.qsize() != 0):
                tempData = readDataBuffer.get(block=False)
                numOfIMU = IMU_case_dict[int(tempData[2], 16)]      # Recognize which IMU (1-9)
                # print(numOfIMU)
                rawData[numOfIMU].append(tempData)
                if (len(rawData.get(numOfIMU)) != 0):
                    quatData[numOfIMU] = qp.get_quat(rawData[numOfIMU].pop())
                    roMat[numOfIMU] = qp.rotate_matrix(quatData[numOfIMU])
                    
            if (check_data_exist(roMat)):
                if (self.is_calibrate):
                    self.is_calibrate = 0   # Reset the calibrate flag
                    calMat = ai.calculate_calibrate_matrix(roMat)
                    self.postureSignal.calibrateMatrix.emit(calMat)
                posAngle = ai.get_angle_by_roMat(roMat)
                # with open(self.fileDir+'/'+self.fileName, 'a') as csvFile:
                #     writer = csv.writer(csvFile, delimiter=',')
                #     writer.writerow(posAngle)
                self.postureSignal.postureAngle.emit(posAngle)
                self.postureSignal.newIMUData.emit(roMat)



class Communicate(QtCore.QObject):
    newIMUData = QtCore.pyqtSignal(dict)
    postureAngle = QtCore.pyqtSignal(list)
    calibrateMatrix = QtCore.pyqtSignal(dict)



def check_data_exist(IMU_Data):
    # for i in IMU_Data.keys():
    #     if (len(IMU_Data.get(i, '')) == 0):
    #         return False
    # return True
    return (len(IMU_Data.get('IMU_1', '')) and len(IMU_Data.get('IMU_2', '')) and len(IMU_Data.get('IMU_3', '')) and len(IMU_Data.get('IMU_4', '')) and len(IMU_Data.get('IMU_5', '')) or
            len(IMU_Data.get('IMU_6', '')) or len(IMU_Data.get('IMU_7', '')) or len(IMU_Data.get('IMU_8', '')) or len(IMU_Data.get('IMU_9', '')))


  
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
