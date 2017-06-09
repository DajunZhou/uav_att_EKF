from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Vector3Stamped, PoseStamped, Quaternion
import numpy as np
import math


def eul2r(rpy):

    sr = math.sin(rpy.x)
    cr = math.cos(rpy.x)
    sp = math.sin(rpy.y)
    cp = math.cos(rpy.y)
    sy = math.sin(rpy.z)
    cy = math.cos(rpy.z)

    R = np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                  [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                  [  -sp,            cp*sr,            cp*cr]])

    return R

def r2eul(R):
    rpy = Vector3()
    rpy.x = math.atan2(R[2][1], R[2][2])
    rpy.y = math.asin(-R[2][0])
    rpy.z = math.atan2(R[1][0], R[0][0])

    return

def eul2quaternion(rpy):

    sr = math.sin(rpy.x/2)
    cr = math.cos(rpy.x/2)
    sp = math.sin(rpy.y/2)
    cp = math.cos(rpy.y/2)
    sy = math.sin(rpy.z/2)
    cy = math.cos(rpy.z/2)

    q = Quaternion()
    q.w = cr*cp*cy + sr*sp*sy
    q.x = sr*cp*cy - cr*sp*sy
    q.y = cr*sp*cy + sr*cp*sy
    q.z = cr*cp*sy - sr*sp*cy

    return q

def quaternion2rr(q): # The right matrix, rotate around its own coordinate system
    q0 = q.w
    q1 = q.x
    q2 = q.y
    q3 = q.z

    R = np.array([[q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2)],
                  [2*(q1*q2+q0*q3), q0*q0-q1*q1+q2*q2-q3*q3, 2*(q2*q3-q0*q1)],
                  [2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), q0*q0-q1*q1-q2*q2+q3*q3]])

    return R

def rr2quaternion(R):
    q = Quaternion()
    q.w = math.sqrt(1 + R[0][0] + R[1][1] + R[2][2]) / 2
    q.x = (R[2][1] - R[1][2]) / (4 * q.w)
    q.y = (R[0][2] - R[2][0]) / (4 * q.w)
    q.z = (R[1][0] - R[0][1]) / (4 * q.w)

    return q



def quaternion2lr(q):  # The left matrix, rotate around a fixed coordinate system
    q0 = q.w
    q1 = q.x
    q2 = q.y
    q3 = q.z

    R = np.array([[q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
                  [2 * (q1 * q2 - q0 * q3), q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3, 2 * (q2 * q3 + q0 * q1)],
                  [2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3]])

    return R

def lr2quaternion(R):
    q = Quaternion()
    q.w = math.sqrt(1 + R[0][0] + R[1][1] + R[2][2]) / 2
    q.x = -(R[2][1] - R[1][2]) / (4 * q.w)
    q.y = -(R[0][2] - R[2][0]) / (4 * q.w)
    q.z = -(R[1][0] - R[0][1]) / (4 * q.w)

    return q




def quaternion2eul(q):
    return r2eul(quaternion2rr(q))