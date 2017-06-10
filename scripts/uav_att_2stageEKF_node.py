import rospy
# from sensor_msgs.msg import Imu
# from geometry_msgs.msg import Vector3, Vector3Stamped, PoseStamped, Quaternion
# import numpy as np
# import math
from collections import deque
from uav_att_2stageEKF import att_2stageEKF
from rotation_trans import *

uav_att = att_2stageEKF()

imu_que = deque()
mag_que = deque()

gyro_bias = np.zeros([3,1], float)
acc_bias = np.zeros([3,1], float)
curr_Pos = np.zeros(3, float)
imu_bias_init = False

# err = Vector3()

def imuCallback(msg):
    global imu_que
    t = msg.header.stamp.to_sec()
    imu_msg = msg
    imu_que.append((t, imu_msg))


def imubiasCallback(msg):
    global gyro_bias, acc_bias, imu_bias_init
    gyro_bias[0] = msg.angular_velocity.x
    gyro_bias[1] = msg.angular_velocity.y
    gyro_bias[2] = msg.angular_velocity.z
    acc_bias[0] = msg.linear_acceleration.x
    acc_bias[1] = msg.linear_acceleration.y
    acc_bias[2] = msg.linear_acceleration.z

    imu_bias_init = True


def magCallback(msg):
    global mag_que
    t = msg.header.stamp.to_sec()
    mag_msg = msg
    mag_que.append((t, mag_msg))


def gt_pose_Callback(msg):
    global curr_Pos
    curr_Pos[0] = msg.pose.position.x
    curr_Pos[1] = msg.pose.position.y
    curr_Pos[2] = msg.pose.position.z
    # print curr_Pos

def get_att_pose():
    global uav_att
    q = uav_att.get_quaternion()
    pose = PoseStamped()

    pose.header.stamp = rospy.Time(uav_att.get_time())
    pose.header.frame_id = "/world"
    pose.pose.position.x = curr_Pos[0]
    pose.pose.position.y = curr_Pos[1]
    pose.pose.position.z = curr_Pos[2]
    pose.pose.orientation.w = q.w
    pose.pose.orientation.x = q.x
    pose.pose.orientation.y = q.y
    pose.pose.orientation.z = q.z

    return pose


rospy.init_node("uav_att_EKF", anonymous=True)
rospy.Subscriber("/raw_imu", Imu, imuCallback, queue_size=100)
rospy.Subscriber("/raw_imu/bias", Imu, imubiasCallback, queue_size=100)
rospy.Subscriber("/magnetic", Vector3Stamped, magCallback, queue_size=100)
rospy.Subscriber("/ground_truth_to_tf/pose", PoseStamped, gt_pose_Callback, queue_size=100)

pose_pub = rospy.Publisher("/pose", PoseStamped, queue_size=10)

r = rospy.Rate(100)

while len(imu_que) == 0 or len(mag_que) == 0 :
    continue

acc = np.zeros([3,1], float)
mag = np.zeros([3,1], float)

t_1, imu_msg = imu_que.popleft()
acc[0] = imu_msg.linear_acceleration.x
acc[1] = imu_msg.linear_acceleration.y
acc[2] = imu_msg.linear_acceleration.z

t_2, mag_msg = mag_que.popleft()
mag[0] = mag_msg.vector.x
mag[1] = mag_msg.vector.y
mag[2] = mag_msg.vector.z

if t_1 < t_2:
    t = t_2
else:
    t = t_1

uav_att.state_init(acc, mag, t)


while not rospy.is_shutdown():

    if len(imu_que) == 0 or len(mag_que) == 0 :
        continue

    while imu_que[-1][0] - imu_que[0][0] > 0.05:

        if imu_que[0][0] < mag_que[0][0]:
            gyro = np.zeros([3,1], float)
            acc = np.zeros([3,1], float)

            t, imu_msg = imu_que.popleft()
            gyro[0] = imu_msg.angular_velocity.x
            gyro[1] = imu_msg.angular_velocity.y
            gyro[2] = imu_msg.angular_velocity.z
            acc[0] = imu_msg.linear_acceleration.x
            acc[1] = imu_msg.linear_acceleration.y
            acc[2] = imu_msg.linear_acceleration.z

            if imu_bias_init:
                gyro -= gyro_bias
                acc -= acc_bias

            uav_att.predit(gyro, t)
            uav_att.correct_by_acc(acc, t)

            att_pose = get_att_pose()
            pose_pub.publish(att_pose)
            # print 1

        else:
            mag = np.zeros([3,1], float)
            Acc = np.zeros([3, 1], float)

            t, mag_msg = mag_que.popleft()

            Acc[0] = imu_que[0][1].linear_acceleration.x
            Acc[1] = imu_que[0][1].linear_acceleration.y
            Acc[2] = imu_que[0][1].linear_acceleration.z
            Acc /= np.linalg.norm(Acc)

            mag[0] = mag_msg.vector.x
            mag[1] = mag_msg.vector.y
            mag[2] = mag_msg.vector.z
            mag /= np.linalg.norm(mag)

            mag_z = sum(mag*Acc)[0]
            magz = mag_z*Acc

            magxy = mag-magz
            magE = magxy / np.linalg.norm(magxy)


            # MAG = mag+Acc
            # MAG /= np.linalg.norm(MAG)

            uav_att.correct_by_mag(magE, t)

            att_pose = get_att_pose()
            pose_pub.publish(att_pose)

            # print 2

        if len(imu_que) == 0 or len(mag_que) == 0:
            break

    r.sleep()





