import numpy as np
from rotation_trans import *
from geometry_msgs.msg import Quaternion

class att_2stageEKF():

    def __init__(self):
        # init parameter / How to init?
        self.Cov_acc = 10  ## gravity acceleration
        self.Cov_gyro = 0.05
        self.Cov_mag = 10

        self.X = np.zeros([4, 1], dtype=float)
        # self.P = np.zeros([4, 4], dtype=float)

        self.Q = np.identity(4, dtype=float) * 0.01

        self.R_acc = np.identity(3, float) * self.Cov_acc
        self.R_gyro = np.identity(3, float) * self.Cov_gyro
        self.R_mag = np.identity(3, float) * self.Cov_mag

        p0 = 0.2
        p1 = 0.1

        self.P = np.array([[p0, p1, p1, p1],
                           [p1, p0, p1, p1],
                           [p1, p1, p0, p1],
                           [p1, p1, p1, p0]])


        # self.curr_t = t
        # self.imu_init = False
        # self.mag_init = False

    def state_init(self, acc, mag, t):
        bIn_z = np.array(acc)
        bIn_y = np.dot(self.skew_symmetric(bIn_z), mag)
        bIn_x = np.dot(self.skew_symmetric(bIn_y), bIn_z)
        bIn_x /= np.linalg.norm(bIn_x)
        bIn_y /= np.linalg.norm(bIn_y)
        bIn_z /= np.linalg.norm(bIn_z)
        bRn = np.transpose(np.array([bIn_x, bIn_y, bIn_z]))
        bRn = bRn.reshape(3, 3)
        nRb = np.transpose(bRn)

        q = rr2quaternion(nRb)
        self.X[0][0] = q.w
        self.X[1][0] = q.x
        self.X[2][0] = q.y
        self.X[3][0] = q.z

        self.curr_t = t

    def skew_symmetric(self, w):

        # S = np.array([[0, -w[2], w[1]],
        #               [w[2], 0, -w[0]],
        #               [-w[1], w[0], 0]])

        S = np.array([[0, -w[2][0], w[1][0]],
                      [w[2][0], 0, -w[0][0]],
                      [-w[1][0], w[0][0], 0]])

        return S

    # priori estimation
    def predit(self, gyro, t):
        dt = t - self.curr_t

        wx = gyro[0][0]
        wy = gyro[1][0]
        wz = gyro[2][0]


        A_tc = 1/2 * np.array([[0, -wx, -wy, -wz],
                               [wx,  0,  wz, -wy],
                               [wy, -wz,  0,  wx],
                               [wz,  wy, -wx,  0]])

        I_4 = np.identity(4, float)
        A = I_4 + A_tc*dt

        self.X = np.dot(A, self.X)
        self.P = np.dot(np.dot(A, self.P), np.transpose(A)) + self.Q


    def correct_by_acc(self, acc, t):

        if t < self.curr_t:
            print "t is smaller than curr_t"
            return

        q0 = self.X[0][0]
        q1 = self.X[1][0]
        q2 = self.X[2][0]
        q3 = self.X[3][0]

        h1_x = np.array([[2*q1*q3 - 2*q0*q2],
                         [2*q0*q1 + 2*q2*q3],
                         [q0*q0 - q1*q1 - q2*q2 + q3*q3]])

        H_k1 = np.array([[-2*q2, 2*q3, -2*q0, 2*q1],
                         [ 2*q1, 2*q0,  2*q3, 2*q2],
                         [ 2*q0,-2*q1, -2*q2, 2*q3]])

        P_inv = self.P


        Item = np.linalg.inv(np.dot(np.dot(H_k1, P_inv), np.transpose(H_k1)) + self.R_acc)
        K_k1 = np.dot(np.dot(P_inv, np.transpose(H_k1)), Item)

        z = acc / np.linalg.norm(acc)
        qe1 = np.dot(K_k1, z - h1_x)
        qe1[3][0] = 0.0
        self.X += qe1
        self.X /= np.linalg.norm(self.X)

        self.P = np.dot((np.identity(4,float) - np.dot(K_k1, H_k1)), P_inv)

    def correct_by_mag(self, mag, t):

        if t < self.curr_t:
            print "t is smaller than curr_t"
            return

        q0 = self.X[0][0]
        q1 = self.X[1][0]
        q2 = self.X[2][0]
        q3 = self.X[3][0]

        h2_x = np.array([[q0*q0 + q1*q1 - q2*q2 - q3*q3],
                         [2 * q1 * q2 - 2 * q0 * q3],
                         [2 * q1 * q3 + 2 * q0 * q2]])


        # h2_x = np.array([[2*q1*q2 + 2*q0*q3],
        #                  [q0*q0 - q1*q1 + q2*q2 - q3*q3],
        #                  [2*q2*q3 - 2*q0*q1]])
        #
        # hz_x = np.array([[2 * q1 * q3 - 2 * q0 * q2],
        #                  [2 * q0 * q1 + 2 * q2 * q3],
        #                  [q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]])

        # H_k2 = np.array([[2*q3,  2*q2,  2*q1,  2*q0],
        #                  [2*q0, -2*q1,  2*q2, -2*q3],
        #                  [-2*q1,-2*q0,  2*q3,  2*q2]])

        H_k2 = np.array([[2*q0,  2*q1, -2*q2, -2*q3],
                         [-2*q3, 2*q2,  2*q1, -2*q0],
                         [2*q2,  2*q3,  2*q0, 2*q1]])

        P_inv = self.P
        Item = np.linalg.inv(np.dot(np.dot(H_k2, P_inv), np.transpose(H_k2)) + self.R_mag)
        K_k2 = np.dot(np.dot(P_inv, np.transpose(H_k2)), Item)

        z = mag
        # z = mag / np.linalg.norm(mag)
        # z[0][0] = 0.0
        # z[2][0] = 0.0
        qe2 = np.dot(K_k2, z - h2_x)
        qe2[1][0] = 0.0
        qe2[2][0] = 0.0
        self.X += qe2
        self.X /= np.linalg.norm(self.X)
        # print qe2

        self.P = np.dot((np.identity(4,float) - np.dot(K_k2, H_k2)), P_inv)

    def get_quaternion(self):
        q = Quaternion()
        q.w = self.X[0][0]
        q.x = self.X[1][0]
        q.y = self.X[2][0]
        q.z = self.X[3][0]
        return q

    def get_time(self):
        return self.curr_t