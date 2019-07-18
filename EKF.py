import csv
import numpy as np
import math
import random
import transforms3d as tf3


IMU_List = ['IMU_1', 'IMU_2', 'IMU_3', 'IMU_4', 'IMU_5', 'IMU_6', 'IMU_7', 'IMU_8', 'IMU_9']
IMU_case_dict = dict(zip(range(1, 10), IMU_List))


class AngleInfo():
    def __init__(self):
        self.PI = 3.1415926536
        self.posture = np.asarray([0.0, 0.0, 0.0])


    def Data_Normalization(self, data_raw):
        d = np.linalg.norm(data_raw)
        if(d != 0):
            q = data_raw / d
            return q
        else:
            return data_raw

    
    def euler_to_quat(self, a1, a2, a3):    # rad
        q = tf3.euler.euler2quat(a1, a2, a3, axes='sxyz')
        return self.Data_Normalization(q)


    def quat_to_eular(self, quat):
        r = tf3.quaternions.quat2mat(quat)

        t = r[2, 0]

        if (t < 0.99):
            if (t > -0.99):
                Pitch = np.arctan2(r[2, 1], r[2, 2])
                Roll = np.arctan2(-t, np.sqrt(r[2, 1]**2 + r[2, 2]**2))
                Yaw = np.arctan2(r[1, 0], r[0, 0])
            else:
                Pitch = 0
                Roll = self.PI / 2
                Yaw = Pitch - np.arctan2(-r[1, 2], r[1, 1])
        else:
            Pitch = 0
            Roll = -self.PI / 2
            Yaw = np.arctan2(-r[1, 2], r[1, 1]) - Pitch
        
        self.posture[0] = round(np.rad2deg(Pitch), 3)
        self.posture[1] = round(np.rad2deg(Roll), 3)
        self.posture[2] = round(np.rad2deg(Yaw), 3)
                                 
        return self.posture




class KalmanFilter():
    def __init__(self):
        self.quat = np.asarray([1.0, 0.0, 0.0, 0.0])
        self.deltaT = 0.02
        
        self.P = np.asarray([[0.125, 0.0003, 0.0003, 0.0003],
                             [0.0003, 0.125, 0.0003, 0.0003],
                             [0.0003, 0.0003, 0.125, 0.0003],
                             [0.0003, 0.0003, 0.0003, 0.125]])                             
        self.Q = 0.00009 * np.identity(4) # 0.000009 0.00009
        self.R_accel = 0.856 * np.identity(3) # 1.96  0.00096
        self.R_mag = 0.556 * np.identity(3)    #0.0556


    def Data_Normalization(self, data_raw):
        d = np.linalg.norm(data_raw)
        if(d != 0):
            q = data_raw / d
            return q
        else:
            return data_raw


    def EKF_Update(self, gyro, accel, mag):
        gyro = np.deg2rad(gyro)
        [P_est, quat_est] = self.EKF_Estimation(gyro)
        [P1, quat1] = self.EKF_Correction_Stage1(quat_est, P_est, accel)
        quat2 = self.EKF_Correction_Stage2(quat_est, quat1, P_est, P1, mag)
        # quat2 = self.EKF_Correction_Stage2_ver2(quat_est, quat1, P_est, P1, mag)
        # self.P = P1
        self.quat = self.Data_Normalization(quat2)

        return self.quat
    

    def EKF_Estimation(self, gyro):
        Om = np.asarray([[0,       -gyro[0], -gyro[1],   -gyro[2]],
                         [gyro[0],  0,       gyro[2],    -gyro[1]],
                         [gyro[1], -gyro[2], 0,          gyro[0]],
                         [gyro[2],  gyro[1], -gyro[0],   0]])
        A = np.identity(4) + 0.5 * Om * self.deltaT

        quat_est = np.dot(A, self.quat)
        quat_est = self.Data_Normalization(quat_est)
        P_est = np.dot(np.dot(A, self.P), A.T) + self.Q

        return [P_est, quat_est]

    
    def EKF_Correction_Stage1(self, quat_est, P_est, accel):
        quat = self.quat
        H_accel = 2 * np.asarray([  [-quat[2], quat[3], -quat[0], quat[1]],
                                    [quat[1], quat[0], quat[3], quat[2]],
                                    [quat[0], -quat[1], -quat[2], quat[3]]])

        PH_accel = np.dot(P_est, H_accel.T)
        K_accel = np.dot(PH_accel, np.linalg.inv(np.dot(H_accel, PH_accel) + self.R_accel))
        
        quat = quat_est
        h1 = np.asarray([[2 * (quat[1] * quat[3] - quat[0] * quat[2])],
                         [2 * (quat[0] * quat[1] + quat[2] * quat[3])],
                         [quat[0]** 2 - quat[1]** 2 - quat[2]** 2 + quat[3]** 2]])
                         
        quat = np.dot(K_accel, (accel.reshape(3, 1) - h1))
        # quat[3] = 0
        quat1 = quat_est + quat.T[0]

        P1 = np.dot((np.identity(4) - np.dot(K_accel, H_accel)), P_est)
        return [P1, quat1]
                        

    def EKF_Correction_Stage2(self, quat_est, quat1, P_est, P1, mag):
        mag = self.Data_Normalization(mag)
        
        quat = self.quat
        H_mag = 2 * np.asarray([[quat[3], quat[2], quat[1], quat[0]],
                                [quat[0], -quat[1], -quat[2], -quat[3]],
                                [-quat[1], -quat[0], quat[3], quat[2]]])

        PH_mag = np.dot(P_est, H_mag.T)
        K_mag = np.dot(PH_mag, np.linalg.inv(np.dot(H_mag, PH_mag) + self.R_mag))

        quat = quat_est
        h2 = np.asarray([[2 * (quat[1] * quat[2] + quat[0] * quat[3])],
                         [quat[0]** 2 - quat[1]** 2 - quat[2]** 2 - quat[3]** 2],
                         [2 * (quat[2] * quat[3] - quat[0] * quat[1])]])

        quat = np.dot(K_mag, (mag.reshape(3, 1) - h2))
        
        quat[1] = 0
        quat[2] = 0
        quat2 = quat1 + quat.T[0]

        self.P = np.dot((np.identity(4) - np.dot(K_mag, H_mag)), P1)
        return quat2


    def EKF_Correction_Stage2_ver2(self, quat_est, quat1, P_est, P1, mag):
        mag = self.Data_Normalization(mag) * 2.0
        quat = self.quat
        
        hx = mag[0] * (0.5 - quat[2]** 2 - quat[3]** 2) + mag[1] * (quat[1] * quat[2] - quat[0] * quat[3]) + mag[2] * (quat[1] * quat[3] + quat[0] * quat[2])
        hy = mag[0] * (quat[1] * quat[2] + quat[0] * quat[3]) + mag[1] * (0.5 - quat[1]** 2 - quat[3]** 2) + mag[2] * (quat[2] * quat[3] - quat[0] * quat[1])
        hz = mag[0] * (quat[1] * quat[3] - quat[0] * quat[2]) + mag[1] * (quat[2] * quat[3] + quat[0] * quat[1]) + mag[2] * (0.5 - quat[1]** 2 - quat[2]** 2)
        bx = math.sqrt(hx** 2 + hy** 2)
        bz = hz

        H_mag = 2 * np.asarray([[bx * quat[0] - bz * quat[2], bx * quat[1] + bz * quat[3], -bx * quat[2] - bz * quat[0], bz * quat[1] - bx * quat[3]],
                                [bz * quat[1] - bx * quat[3], bx * quat[2] + bz * quat[0], bx * quat[1] + bz * quat[3], bz * quat[2] - bx * quat[0]],
                                [bx * quat[2] + bz * quat[0], bx * quat[3] - bz * quat[1], bx * quat[0] - bz * quat[2], bx * quat[1] + bz * quat[3]]])
                                  
        PH_mag = np.dot(P_est, H_mag.T)
        K_mag = np.dot(PH_mag, np.linalg.inv(np.dot(H_mag, PH_mag) + self.R_mag))

        quat = quat_est
        h2 = 2 * np.asarray([[bx * (0.5 - (quat[2]** 2 + quat[3]** 2)) + bz * (quat[1] * quat[3] - quat[0] * quat[2])],
                             [bx * (quat[1] * quat[2] - quat[0] * quat[3]) + bz * (quat[0] * quat[1] + quat[2] * quat[3])],
                             [bx * (quat[1] * quat[3] + quat[0] * quat[2]) + bz * (0.5 - (quat[1]** 2 + quat[2]** 2))]])
     
        quat = np.dot(K_mag, (mag.reshape(3, 1) / 2.0 - h2))
        quat[1] = 0
        quat[2] = 0
        quat2 = quat1 + quat.T[0]

        self.P = np.dot((np.identity(4) - np.dot(K_mag, H_mag)), P1)
        return quat2
