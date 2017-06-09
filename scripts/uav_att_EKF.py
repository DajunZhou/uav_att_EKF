from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Vector3Stamped, PoseStamped, Quaternion
import numpy as np
import math


class att_est_EKF():
    def __init__(self):

        #init parameter
        self.Cov_acc = 10  ## gravity acceleration
        self.Cov_gyro = 0.05
        self.Cov_mag = 10

        self.X = np.zeros([12, 1], dtype=float)
        self.P = np.zeros([12, 12], dtype=float)


        self.Q = np.identity(12, dtype=float) * 0.01

        self.R_acc = np.identity(3, dtype=float)*self.Cov_acc
        self.R_gyro = np.identity(3, dtype=float)*self.Cov_gyro
        self.R_mag = np.identity(3, dtype=float)*self.Cov_mag

        self.R_imu = np.zeros([6, 6], dtype=float)
        self.R_imu[0:3, 0:3] = self.R_gyro
        self.R_imu[3: , 3: ] = self.R_acc

        self.P[0:3, 0:3] = self.R_gyro
        self.P[3:6, 3:6] = self.R_gyro
        self.P[6:9, 6:9] = self.R_acc
        self.P[9: , 9: ] = self.R_mag


        self.curr_t = 0.0
        self.imu_init = False
        self.mag_init = False


    def skew_symmetric(self, w):

        # S = np.array([[0, -w[2], w[1]],
        #               [w[2], 0, -w[0]],
        #               [-w[1], w[0], 0]])

        S = np.array([[0, -w[2][0], w[1][0]],
                      [w[2][0], 0, -w[0][0]],
                      [-w[1][0], w[0][0], 0]])

        return S

    def predict(self, t):

        dt = t - self.curr_t

        w = np.array(self.X[0:3])
        dw = np.array(self.X[3:6])
        rg = np.array(self.X[6:9])
        rm = np.array(self.X[9: ])

        #priori estimation
        ##################
        self.X[0:3] += dw*dt
        self.X[6:9] += -np.dot(self.skew_symmetric(w), rg)*dt ## Because the vectors of gravitaion and magnetic field is stationary relative to the world coordinate system
        self.X[9: ] += -np.dot(self.skew_symmetric(w), rm)*dt

        A = np.identity(12 ,dtype=float)
        A[0:3, 3:6] = np.identity(3, float)*dt
        A[6:9, 0:3] = self.skew_symmetric(rg)*dt
        A[6:9, 6:9] += -self.skew_symmetric(w)*dt
        A[9: , 0:3] = self.skew_symmetric(rm)*dt
        A[9: , 9: ] += -self.skew_symmetric(w)*dt

        self.P = np.dot((np.dot(A, self.P)), np.transpose(A)) + self.Q

        self.curr_t = t
        ##################
        # print self.X[3:6]

    def update_by_mag(self, mag, t):

        if not (self.mag_init and self.imu_init):
            self.X[9: ] = mag
            self.mag_init = True
            self.curr_t = t
            return

        if t < self.curr_t:
            print "t is smaller than curr_t"
            return

        self.predict(t)

        H = np.zeros([3, 12], dtype=float)
        z = np.array(mag)
        H[0:, 9: ] = np.identity(3, float)

        # P_inv = np.linalg.inv(self.P)
        P_inv = self.P

        Item = np.linalg.inv(np.dot(np.dot(H, P_inv), np.transpose(H)) + self.R_mag)
        K = np.dot(np.dot(P_inv, np.transpose(H)), Item)
        I_12 = np.identity(12, float)

        # posteriori estimaiton
        self.X = self.X + np.dot(K, z-np.dot(H, self.X))
        self.P = np.dot((I_12 - np.dot(K, H)), P_inv)


    def update_by_imu(self, gyro, acc, t):

        if not (self.mag_init and self.imu_init):
            self.X[0:3] = gyro
            self.X[6:9] = acc

            self.imu_init = True
            print "imu initialize: ", np.transpose(self.X)[0,:]
            self.curr_t = t
            return

        if t < self.curr_t:
            print "t is smaller than curr_t"
            return

        self.predict(t)

        H = np.zeros([6, 12], dtype=float)
        z = np.append(gyro, acc, axis=0)
        H[0:3, 0:3] = np.identity(3, float)
        H[3: , 6:9] = np.identity(3, float)

        # P_inv = np.linalg.inv(self.P)
        P_inv = self.P

        Item = np.linalg.inv(np.dot(np.dot(H, P_inv), np.transpose(H)) + self.R_imu)
        K = np.dot(np.dot(P_inv, np.transpose(H)), Item)
        I_12 = np.identity(12, float)
        # xx = z - np.dot(H, self.X)

        # posteriori estimaiton
        self.X = self.X + np.dot(K, z - np.dot(H, self.X))
        self.P = np.dot((I_12 - np.dot(K, H)), P_inv)


    def get_Rotation_Matrix(self):
        if not (self.mag_init and self.imu_init):
            return np.identity(3, float)

        rg = self.X[6:9]
        rm = self.X[9:]
        # print rg[2]
        bIn_z = np.array(rg)
        bIn_y = np.dot(self.skew_symmetric(bIn_z), rm)
        bIn_x = np.dot(self.skew_symmetric(bIn_y), bIn_z)
        bIn_x /= np.linalg.norm(bIn_x)
        bIn_y /= np.linalg.norm(bIn_y)
        bIn_z /= np.linalg.norm(bIn_z)
        bRn = np.transpose(np.array([bIn_x, bIn_y, bIn_z]))
        bRn = bRn.reshape(3,3)
        return np.transpose(bRn)

        # rg = self.X[6:9]
        # rm = self.X[9: ]
        # rm[1] = -rm[1]
        # rm[2] = -rm[2]
        #
        # bIn_z = np.array(rg)
        # bIn_y = np.dot(self.skew_symmetric(bIn_z),rm)
        # bIn_x = np.dot(self.skew_symmetric(bIn_y),bIn_z)
        #
        # bIn_x /= np.linalg.norm(bIn_x)
        # bIn_y /= np.linalg.norm(bIn_y)
        # bIn_z /= np.linalg.norm(bIn_z)
        #
        # bRn = np.transpose(np.array([bIn_x, bIn_y, bIn_z]))
        # return bRn.reshape(3,3)

    def get_time(self):
        return self.curr_t

