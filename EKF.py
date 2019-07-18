import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import transforms3d as tf3

class ReadCsv(object):
    def __init__(self):
        self.fileDir = "D:/Lab/IMU_Test/IMU recv_Test/sensor_data"
        self.fileName = "2019_06_03_1131.csv"


    def read_file(self):
        with open(self.fileDir + '/' + self.fileName, 'r') as csvFile:
            rows = csv.reader(csvFile)
            data = np.asarray(list(rows))
            csvFile.close()
        return data



class DataProcess():
    def __init__(self):
        self.gyroRes = 1000.0 / 32768.0
        self.accelRes = 2.0 / 32768.0
        self.magRes = 4912.0 / 32760.0
        self.magScale  = np.asarray([1.01, 0.98, 0.92])
        self.magBias = np.asarray([71.04, 72.43, -5.90])
        self.gyroBias = np.asarray([0.56, -0.21, 0.12])
        self.accelBias = np.asarray([0.00299, -0.00916, 0.00952])


    def get_sensors(self, inputData):
        self.gyroRaw = np.asarray(inputData[:, 0:3]).astype(np.float64)
        self.accelRaw = np.asarray(inputData[:, 3:6]).astype(np.float64)
        self.magRaw = np.asarray(inputData[:, 6:9]).astype(np.float64)
        self.get_sensor_value()
        return [self.gyroVal, self.accelVal, self.magVal]


    def get_sensor_value(self):
        
        self.gyroVal = self.gyroRaw * self.gyroRes# - self.gyroBias
        self.accelVal = self.accelRaw * self.accelRes - self.accelBias
        self.magVal = (self.magRaw * self.magRes - self.magBias) * self.magScale



class AngleInfo():
    def __init__(self):
        self.PI = 3.1415926536
        self.Threshold = 0.5 - 0.009765625
        self.posture = np.asarray([0.0, 0.0, 0.0])


    def rotate_matrix(self, q1):
        xx = q1[1]** 2
        xy = q1[1] * q1[2]
        xz = q1[1] * q1[3]
        xw = q1[1] * q1[0]
        yy = q1[2]** 2
        yz = q1[2] * q1[3]
        yw = q1[2] * q1[0]
        zz = q1[3]** 2
        zw = q1[3] * q1[0]
        
        rotateMat = np.array(2 * [  [0.5 - yy - zz, xy + zw, xz - yw],
                                    [xy - zw, 0.5 - xx - zz, yz + xw],
                                    [xz + yw, yz - xw, 0.5 - xx + yy]])
        return rotateMat


    def get_IMU_eular_angle(self, quat):
        Pitch = np.arctan2(2 * (quat[2] * quat[3] + quat[0] * quat[1]), (1 - 2 * (quat[1]** 2 + quat[2]** 2)))        
        
        if ((abs(Pitch) <= 45) or (abs(Pitch) > 135)):
            [Yaw, Roll, Pitch] = tf3.euler.quat2euler(quat, axes='szyx')
        else:
            [Yaw, Pitch, Roll] = tf3.euler.quat2euler(quat, axes='szxy')
            
        self.posture[0] = round(np.rad2deg(Pitch), 3)
        self.posture[1] = round(np.rad2deg(Roll), 3)
        self.posture[2] = round(np.rad2deg(Yaw), 3)
                                 
        return self.posture



class TestFunc():
    def __init__(self):
        self.datalen = 180
    

    def quat_generator(self, axes):
        quat = np.empty(shape=[self.datalen, 4])
        
        
        if (axes == 'x'):
            for i in range(1, self.datalen + 1):
                quat[i - 1] = self.euler_to_quat([i / 2, 0, 0])
        elif (axes == 'y'):
            for i in range(1, self.datalen + 1):
                quat[i - 1] = self.euler_to_quat([0, i / 2, 0])
        elif (axes == 'z'):
            for i in range(1, self.datalen + 1):
                quat[i - 1] = self.euler_to_quat([0, 0, i / 2])
        return quat


    def euler_to_quat(self, euler):
        q = [0, 0, 0, 0]
        p = math.radians(euler[0] / 2)
        r = math.radians(euler[1] / 2)
        y = math.radians(euler[2] / 2)
        q[0] = math.cos(p) * math.cos(r) * math.cos(y) + math.sin(p) * math.sin(r) * math.sin(y)
        q[1] = math.sin(p) * math.cos(r) * math.cos(y) - math.cos(p) * math.sin(r) * math.sin(y)
        q[2] = math.cos(p) * math.sin(r) * math.cos(y) + math.sin(p) * math.cos(r) * math.sin(y)
        q[3] = math.cos(p) * math.cos(r) * math.sin(y) - math.sin(p) * math.sin(r) * math.cos(y)
        q = np.asarray(q) + np.asarray([random.random() / 100, random.random() / 100, random.random() / 100, random.random() / 100])
        return self.Data_Normalization(q)
    

    def Data_Normalization(self, data_raw):
        d = np.linalg.norm(data_raw)
        if(d != 0):
            q = data_raw / d
            return q
        else:
            return data_raw



class KalmanFilter():
    def __init__(self):
        self.quat = np.asarray([1.0, 0.0, 0.0, 0.0])
        self.deltaT = 0.02
        
        # self.P = 0.001 * np.identity(4)
        self.P = np.asarray([[0.125, 0.0003, 0.0003, 0.0003],
                             [0.0003, 0.125, 0.0003, 0.0003],
                             [0.0003, 0.0003, 0.125, 0.0003],
                             [0.0003, 0.0003, 0.0003, 0.125]])                             
        self.Q = 0.000009 * np.identity(4)
        self.R_accel = 1.96 * np.identity(3)
        self.R_mag = 0.0256 * np.identity(3)  


    def Data_Normalization(self, data_raw):
        d = np.linalg.norm(data_raw)
        if(d != 0):
            q = data_raw / d
            return q
        else:
            return data_raw


    def EKF_Update(self, gyro, accel, mag):
        gyro = gyro* 0.0174533
        quat_est = self.EKF_Estimation(gyro)
        [P1, quat1] = self.EKF_Correction_Stage1(quat_est, accel)
        self.quat = self.Data_Normalization(quat1)
        # self.EKF_Correction_Stage2_ver2(quat_est, quat1, P1, mag)

        return self.quat
    

    def EKF_Estimation(self, gyro):
        A = np.asarray([[0,       -gyro[0], -gyro[1],   -gyro[2]],
                        [gyro[0],  0,       gyro[2],    -gyro[1]],
                        [gyro[1], -gyro[2], 0,          gyro[0]],
                        [gyro[2],  gyro[1], -gyro[0],   0]])
        A = np.identity(4) + 0.5 * A * self.deltaT

        quat_est = np.dot(A, self.quat)
        quat_est = self.Data_Normalization(quat_est)
        self.P = np.dot(np.dot(A, self.P), A.T) + self.Q

        return quat_est

    
    def EKF_Correction_Stage1(self, quat_est, accel):
        # accel = self.Data_Normalization(accel)
        quat = self.quat
        H_accel = 2 * np.asarray([  [-quat[2], quat[3], -quat[0], quat[1]],
                                    [quat[1], quat[0], quat[3], quat[2]],
                                    [quat[0], -quat[1], -quat[2], quat[3]]])

        PH_accel = np.dot(self.P, H_accel.T)
        K_accel = np.dot(PH_accel, np.linalg.inv(np.dot(H_accel, PH_accel) + self.R_accel))
        
        quat = quat_est
        quat[3] = 0
        h1 = np.asarray([[2 * (quat[1] * quat[3] - quat[0] * quat[2])],
                         [2 * (quat[0] * quat[1] + quat[2] * quat[3])],
                         [quat[0]** 2 - quat[1]** 2 - quat[2]** 2 + quat[3]** 2]])
                         
        quat = np.dot(K_accel, (accel.reshape(3, 1) - h1))
        quat1 = quat_est + quat.T[0]
        quat1 = self.Data_Normalization(quat1)

        P1 = np.dot((np.identity(4) - np.dot(K_accel, H_accel)), self.P)
        return [P1, quat1]
                        

    def EKF_Correction_Stage2(self, quat_est, quat1, P1, mag):
        mag = self.Data_Normalization(mag)
        quat = self.quat
        H_mag = 2 * np.asarray([[quat[3], quat[2], quat[1], quat[0]],
                                [quat[0], -quat[1], -quat[2], -quat[3]],
                                [-quat[1], -quat[0], quat[3], quat[2]]])

        PH_mag = np.dot(self.P, H_mag.T)
        K_mag = np.dot(PH_mag, np.linalg.inv(np.dot(H_mag, PH_mag) + self.R_mag))

        quat = quat_est
        quat[1] = 0
        quat[2] = 0
        h2 = np.asarray([[2 * (quat[1] * quat[2] + quat[0] * quat[3])],
                         [quat[0]** 2 - quat[1]** 2 - quat[2]** 2 - quat[3]** 2],
                         [2 * (quat[2] * quat[3] - quat[0] * quat[1])]])

        quat = np.dot(K_mag, (mag.reshape(3, 1) - h2))
        self.quat = quat1 + quat.T[0]
        self.quat = self.Data_Normalization(self.quat)

        self.P = np.dot((np.identity(4) - np.dot(K_mag, H_mag)), P1)


    def EKF_Correction_Stage2_ver2(self, quat_est, quat1, P1, mag):
        mag = self.Data_Normalization(mag)
        quat = self.quat
        
        hx = 2 * mag[0] * (0.5- quat[2]** 2 - quat[3]** 2) + 2 * mag[1] * (quat[1] * quat[2] - quat[0] * quat[3]) + 2 * mag[2] * (quat[1] * quat[3] + quat[0] * quat[2])
        hy = 2 * mag[0] * (quat[1] * quat[2] + quat[0] * quat[3]) + 2 * mag[1] * (0.5- quat[1]** 2 - quat[3]** 2) + 2 * mag[2] * (quat[2] * quat[3] - quat[0] * quat[1])
        hz = 2 * mag[0] * (quat[1] * quat[3] - quat[0] * quat[2]) + 2 * mag[1] * (quat[2] * quat[3] + quat[0] * quat[1]) + 2 * mag[2] * (0.5- quat[1]** 2 - quat[2]** 2)
        bx = math.sqrt(hx ** 2 + hy ** 2)
        bz = hz

        H_mag = 2 * np.asarray([[bx * quat[0] - bz * quat[2], bx * quat[1] + bz * quat[3], -bx * quat[2] - bz * quat[0], bz * quat[1] - bx * quat[3]],
                                [bz * quat[1] - bx * quat[3], bx * quat[2] + bz * quat[0], bx * quat[1] + bz * quat[3], bz * quat[2] - bx * quat[0]],
                                [bx*quat[2] + bz*quat[0], bx*quat[3] - bz*quat[1], bx*quat[0] - bz*quat[2], bx*quat[1] + bz*quat[3]]])
                                
        quat = quat_est
        quat[1] = 0
        quat[2] = 0
                               
        PH_mag = np.dot(self.P, H_mag.T)
        K_mag = np.dot(PH_mag, np.linalg.inv(np.dot(H_mag, PH_mag) + self.R_mag))

        h2 = np.asarray([[2 * bx * (0.5 - (quat[2]** 2 + quat[3]** 2)) + 2 * bz * (quat[1] * quat[3] - quat[0] * quat[2])],
                         [2 * bx * (quat[1] * quat[2] - quat[0] * quat[3]) + 2 * bz * (quat[0] * quat[1] + quat[2] * quat[3])],
                         [2 * bx * (quat[1] * quat[3] + quat[0] * quat[2]) + 2 * bz * (0.5 - (quat[1]** 2 + quat[2]** 2))]])

        quat = np.dot(K_mag, (mag.reshape(3, 1) - h2))
        self.quat = quat1 + quat.T[0]
        self.quat = self.Data_Normalization(self.quat)

        self.P = np.dot((np.identity(4) - np.dot(K_mag, H_mag)), P1)



if __name__ == '__main__':
    rc = ReadCsv()
    dp = DataProcess()
    ai = AngleInfo() 
    kf = KalmanFilter()
    tc = TestFunc()


    quat_x = tc.quat_generator('x')
    quat_x2 = quat_x[::-1]
    quat_y = tc.quat_generator('y')
    quat_y2 = quat_y[::-1]
    quat_z = tc.quat_generator('z')
    quat_z2 = quat_z[::-1]

    # test_quat = [0.7071, 0.7070, 0.0093, 0.0031]
    test_quat_x = [0.7071, 0.7071, 0, 0]
    test_quat_y = [0.7071, 0, 0.7071, 0]
    test_quat_z = [0.7071, 0, 0, 0.7071]
    test_quat_x = np.tile(test_quat_x, (50, 1))
    test_quat_y = np.tile(test_quat_y, (50, 1))
    test_quat_z = np.tile(test_quat_z, (50, 1))

    quat_save = np.concatenate((quat_x, test_quat_x, quat_x2, quat_y, test_quat_y, quat_y2, quat_z, test_quat_z, quat_z2), 0)

    test_len = quat_save.shape[0]
    pry = np.empty(shape=[test_len, 3])
    
    for i in range(test_len):
        pry[i] = ai.get_IMU_eular_angle(quat_save[i])

    # inputData = rc.read_file()
    # [gyro, accel, mag] = dp.get_sensors(inputData)

    # len = gyro.shape[0]

    # pry = np.empty(shape=[len, 3])
    # quat_save = np.empty(shape=[len, 4])
    
    # for i in range(len):
    #     quat = kf.EKF_Update(gyro[i], accel[i], mag[i])
    #     pry[i] = ai.get_IMU_eular_angle(quat)
    #     quat_save[i] = quat


    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(pry[2:-1, 0], color='red', label='Pitch')
    ax1.plot(pry[2:-1, 1], color='blue', label='Roll')
    ax1.plot(pry[2:-1, 2], color='lime', label='Yaw')
    ax1.legend()

    ax2.plot(quat_save[2:-1, 0], color='brown', label='q0')
    ax2.plot(quat_save[2:-1, 1], color='red', label='q1')
    ax2.plot(quat_save[2:-1, 2], color='blue', label='q2')
    ax2.plot(quat_save[2:-1, 3], color='lime', label='q3')
    ax2.legend()
    plt.show()
        