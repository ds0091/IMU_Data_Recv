import imu
import numpy as np
import socket

PacketSize = 22

IMU_List = ['IMU_1', 'IMU_2', 'IMU_3', 'IMU_4', 'IMU_5', 'IMU_6', 'IMU_7', 'IMU_8', 'IMU_9']
IMU_case_dict = dict(zip(range(1, 10), IMU_List))

class GetSensorsValue():
    def __init__(self):
        self.gyroRes = 1000.0 / 32768.0
        self.accelRes = 2.0 / 32768.0
        self.magRes = 10 * 4912.0 / 32760.0
        self.magScale  = np.asarray([1.01, 0.91, 0.82])
        self.magBias = np.asarray([71.04, 82.43, 15.90])
        self.gyroBias = np.asarray([0.56, -0.21, 0.12])
        self.accelBias = np.asarray([0.00299, -0.00916, 0.00952])


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
                    
                    gryo_data = gryo_data * self.gyroRes - self.gyroBias
                    accel_data = accel_data * self.accelRes - self.accelBias
                    mag_data = (mag_data * self.magRes - self.magBias) * self.magScale
                    return [gryo_data, accel_data, mag_data]

        except Exception:
            pass


    def two_byte_to_int(self, byte1, byte2):
        num = byte1 * 256 + byte2
        if (num > 32767):
            num = num - 65536
        return num



class WiFiConnection():
    def __init__(self, Static_ip, WiFiPort):
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
        id_IMU = clientSocket.recv(imu.PacketSize).hex()
        print("Connected from: IMU" + str(int(id_IMU[4:6], 16)))
        clientSocket.setblocking(0)
        return clientSocket


    def recv_close(self, sockets):
        for s in sockets:
            s.close()



def clear_buffer(sockets):
    for s in sockets:
        try:
            s.recv(PacketSize*1000).hex()
        except Exception:
            print('Clear Error')
            continue