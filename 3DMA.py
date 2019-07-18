import csv
import numpy as np
import math
import matplotlib.pyplot as plt

class ReadCsv(object):
    def __init__(self):
        self.fileDir = "D:/Lab/IMU Exp"
        self.fileName = "1080712.csv"  # 10 13
        

    def read_file(self):
        with open(self.fileDir + '/' + self.fileName, 'r') as csvFile:
            rows = csv.reader(csvFile)
            data = np.asarray(list(rows))
            csvFile.close()
        return data



if __name__ == '__main__':
    rc = ReadCsv()
    

    inputData = rc.read_file()
    inputData = np.asarray(inputData[5:, 2:]).astype('float64')
    # print(inputData[:, 0:3])
    # print(inputData[:, 9:12])
    # print(inputData[:, 12:15])
    vecUpArmL = inputData[:, 0:3] - inputData[:, 9:12]
    vecLowArmL = inputData[:, 9:12] - inputData[:, 12:15]
    len = vecUpArmL.shape[0]
    angle = np.empty((len,1))
    for i in range(len):
        angle[i] = math.degrees(np.arccos(np.inner(vecUpArmL[i], vecLowArmL[i]) / (np.linalg.norm(vecUpArmL[i]) * np.linalg.norm(vecLowArmL[i]))))
    
    plt.figure()
    plt.plot(angle[1200:2500])
    plt.show()
