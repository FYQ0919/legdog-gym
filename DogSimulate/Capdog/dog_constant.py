"""
~~~~~~~~~~~~~~~~~~~~~~~~~~
Copyright @ Changda Tian
2023.6
SJTU RL2 LAB
~~~~~~~~~~~~~~~~~~~~~~~~~~

Define constants and kinematics for Littledog
"""

import math
import mujoco
import mujoco_viewer
import os
import numpy as np
from math import sin,cos
from enum import Enum, auto
from numpy import deg2rad
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt

DTYPE = np.float32

def quaternion_2_euler(q):
    w,x,y,z = q
    eps = 0.0009765625
    thres = 0.5 - eps

    test = w * y - x * z
    
    if test < -thres or test > thres:
        sign = 1 if test > 0 else -1
        gamma = -2 * sign * math.atan2(x, w)
        beta = sign * (math.pi / 2)
        alpha = 0
    else:
        alpha = math.atan2(2 * (y*z + w*x), w*w - x*x - y*y + z*z)
        beta = math.asin(-2 * (x*z - w*y))
        gamma = math.atan2(2 * (x*y + w*z), w*w + x*x - y*y - z*z)
    return alpha,beta,gamma

def euler_2_quaternion(eu):
    rm = Rot.from_euler('xyz',eu,degrees=True)
    q = rm.as_quat()
    return q

class RobotType(Enum):
    MINI_CHEETAH = auto()
    LITTLE_DOG = auto()
    QINGZHUI = auto()

class Robot:

    def __init__(self, robotype:RobotType):

        if robotype is RobotType.LITTLE_DOG:
            self._abadLinkLength = 0.0802
            self._hipLinkLength = 0.249
            self._kneeLinkLength = 0.24532
            self._bodyMass = 22
            self._legNum = 6
            # self._bodyInertia = np.array([0.078016, 0, 0,
			# 	 0, 0.42789, 0,
			# 	 0, 0, 0.48331])
            self._bodyInertia = np.array([2.8097, 0, 0,
				 0, 2.5396, 0,
				 0, 0, 0.5977])            
            self._bodyHeight = 0.41
            self._foot_ball_rad = 0.02
            # robot 里的逆运动学： 负、正
            self._nominal_jpos = np.tile(np.array([0,-0.5,1.0]), (self._legNum,1))
            self.nomi_tip_array_in_hip = np.array([
                [-0.00, -0.08, -self._bodyHeight],
                [0, -0.08, -self._bodyHeight],
                [-0.05, -0.08, -self._bodyHeight],
                [-0.05, 0.08, -self._bodyHeight],
                [0, 0.08, -self._bodyHeight],
                [-0.00, 0.08, -self._bodyHeight]
            ])
            for i in range(self._legNum):
                isRight = 1 if i<3 else -1
                self._nominal_jpos[i,:], _ = self.inverseKinematics(
                    self.nomi_tip_array_in_hip[i,:],
                    isRight,
                    bendInfo=-1
                )

            self._kneeLinkY_offset = 0.0 
            self._abadLocation = np.array([
                [0.33, -0.05, 0.0],   # rf
                [0.0, -0.19025, 0.0], # rm
                [-0.33, -0.05, 0.0],  # rb
                [-0.33, 0.05, 0.0],   # lb
                [0.0, 0.19025, 0.0],  # lm
                [0.33, 0.05, 0.0]     # lf
            ], dtype=DTYPE).reshape((6,3))

            self._nominal_tippos = self._abadLocation + self.nomi_tip_array_in_hip


            self._bodyName = "base"

            self._friction_coeffs = np.ones(4, dtype=DTYPE) * 0.4
            # (roll_pitch_yaw, position, angular_velocity, velocity, gravity_place_holder)
            self._mpc_weights = [1, 1, 0.0,
                                 0.1, 0.1, 1.0,
                                 0.1, 0.1, 0.1,
                                 1.0, 1.0, 0.1,
                                 0.0]

            # for kinematic constraints
            self._kinematicBounds = []
            for i in range(self._legNum):
                if i < 3:
                    self._kinematicBounds.append(
                        np.array([
                            [-0.2,0.4],  # x lower-upper in hip frame
                            [-0.4,0.1],  # y
                            [-0.5,-0.2]  # z
                        ])
                    )
                else:
                    self._kinematicBounds.append(
                        np.array([
                            [-0.2,0.4],
                            [-0.1,0.4],
                            [-0.5,-0.2]
                        ])                        
                    )
        
        else:
            raise Exception("Invalid RobotType")
            
        self._robotType = robotype

    def getHipLocation(self, leg:int):
        """
        Get location of the hip for the given leg in robot base frame
        leg index: rf, rm, rb, lb, lm, lf
        return: shape (3,1)
        """
        assert leg >= 0 and leg < self._legNum
        pHip = self._abadLocation[leg,:].reshape((3,1))
        return pHip.copy()
    
    def get_hip_nominal(self):
        hn = np.zeros((self._legNum,3))
        for i in range(self._legNum):
            hn[i] = self.getHipLocation(i).flatten()
        return hn
    
    def forwardKinematics(self, leg:int, jpos:np.ndarray):
        '''
            forward kinematics
            return: tipPos (3,)
        '''
        l0 = self._abadLinkLength
        l1 = self._hipLinkLength
        l2 = self._kneeLinkLength
        s0 = sin(jpos[0])
        c0 = cos(jpos[0])
        s1 = sin(jpos[1])
        c1 = cos(jpos[1])
        s2 = sin(jpos[1]+jpos[2])
        c2 = cos(jpos[1]+jpos[2])

        ls0 = l0 * s0
        lc0 = l0 * c0
        ls1 = l1 * s1
        lc1 = l1 * c1
        ls2 = l2 * s2
        lc2 = l2 * c2

        tipPos = np.zeros(3)
        tipPos[0] = -ls1 - ls2
        isRightLeg = 1 if leg < 3 else -1

        tipPos[1] = - isRightLeg * lc0 + (lc1+lc2)*s0
        tipPos[2] = - isRightLeg * ls0 - (lc1+lc2)*c0

        return tipPos
    
    def forward_kinematics(self,jpos):
        fk = np.zeros((self._legNum,3))
        for i in range(self._legNum):
            fk[i] = self.forwardKinematics(i,jpos[i])
        return fk

    def getHipJacobian(self, leg:int, qpos:np.ndarray) -> np.ndarray:
        '''
            get linear velocity Jacobian in the hip frame
        '''
        s = np.zeros(3)
        c = np.zeros(3)
        ls = np.zeros(3)
        lc = np.zeros(3)
        actJ = qpos

        l0 = self._abadLinkLength
        l1 = self._hipLinkLength
        l2 = self._kneeLinkLength


        s[0]=sin(actJ[0])
        c[0]=cos(actJ[0])
        s[1]=sin(actJ[1])
        c[1]=cos(actJ[1])
        s[2]=sin(actJ[1]+actJ[2])
        c[2]=cos(actJ[1]+actJ[2])
        ls[0]=l0*s[0]
        lc[0]=l0*c[0]
        ls[1]=l1*s[1]
        lc[1]=l1*c[1]
        ls[2]=l2*s[2]
        lc[2]=l2*c[2]

        isRightLeg = 1 if leg < 3 else -1


        return np.array([
            [0, -lc[1]-lc[2], -lc[2]],
            [isRightLeg*ls[0]+c[0]*(lc[1]+lc[2]), -s[0]*(ls[1]+ls[2]), -s[0]*ls[2]],
            [-isRightLeg*lc[0]+s[0]*(lc[1]+lc[2]), c[0]*(ls[1]+ls[2]), c[0]*ls[2]]
        ])
    
    def get_jacobian_hip(self,qpos):
        j = np.zeros((self._legNum,3,3))
        for i in range(self._legNum):
            j[i] = self.getHipJacobian(i,qpos[i])
        return j
    
    def inverseKinematics(self, tip:np.ndarray, rightFlag:int, bendInfo:int):
        '''
            计算逆运动学
            tip: ndarray (3,) in the hip frame
            rightFlag: 1 右腿 or -1 左腿
            bendInfo: 1 负正 or -1 正负

            return: joint: ndarray (3,) 
                    normal: bool
        '''
        l0 = self._abadLinkLength
        l1 = self._hipLinkLength
        l2 = self._kneeLinkLength    

        normal = True
        # firstSolution = True

        p = tip.copy()
        legLength = np.linalg.norm(p)
        legLengthMax = math.sqrt((l1+l2)*(l1+l2) + l0*l0)
        if legLength > legLengthMax:
            p = p * legLengthMax/legLength
            normal = False    
        
        legLengthMin = math.sqrt(l0*l0)
        if legLength < legLengthMin:
            p = p * legLengthMin/legLength
            normal = False

        yz_square = p[1]*p[1] + p[2]*p[2] - l0*l0
        if yz_square < 0:
            p[1] = p[1]/math.sqrt(p[1]*p[1]+p[2]*p[2])*l0 + 1e-5
            p[2] = p[2]/math.sqrt(p[1]*p[1]+p[2]*p[2])*l0 + 1e-5
            yz_square = 0
            
        
        length_square = yz_square + p[0]*p[0]

        yz = math.sqrt(yz_square)           # 腿长在yz平面上的投影长度
        length = math.sqrt(length_square)   # 腿长

        if rightFlag == 1:
            q0 = - (math.atan2(p[2], -p[1]) + math.atan2(yz, l0))
        else:
            q0 =  math.atan2(p[2], p[1]) + math.atan2(yz,l0)

        if bendInfo == 1:
            # 负正
            tmp = (l1*l1+length_square-l2*l2)/2/l1/length
            tmp = max(min(tmp,1),-1)

            q1 = -math.atan2(p[0], yz) - math.acos(tmp)

            
            tmp2 = (l1*l1+l2*l2-length_square)/2/l1/l2
            tmp2 = max(min(tmp2,1),-1)
            q2 = math.pi - math.acos(tmp2)

        else:
            # 正负
            tmp = (l1*l1+length_square-l2*l2)/2/l1/length
            tmp = max(min(tmp,1),-1)

            q1 = -math.atan2(p[0], yz) + math.acos(tmp)

            tmp2 = (l1*l1+l2*l2-length_square)/2/l1/l2
            tmp2 = max(min(tmp2,1),-1)
            q2 = -math.pi + math.acos(tmp2)

        return np.array([q0,q1,q2]), normal


    def getNominalFoot(self, leg:int):
        '''
            得到标准足端位置在base坐标系下的坐标
        '''
        return self._nominal_tippos[leg,:].reshape(3,1).copy()
    
    def get_feet_nominal(self):
        fn = np.zeros((self._legNum,3))
        for i in range(self._legNum):
            fn[i] = self.getNominalFoot(i).flatten()
        return fn
    
    def inv_kine_hip(self, tip_pos, bend_info):
        """inverse kinematics in leg hip frame"""
        # t1 > 0 , t2 < 0 for bend in
        # t1 < 0 , t2 > 0 for bend out
        """inverse kinematics in leg hip frame
            tip_pos: ndarray (6,3)
            bend_info: ndarray (6,)
        """
        # t1 > 0 , t2 < 0 for bend in. bend_info=1
        # t1 < 0 , t2 > 0 for bend out. bend_info=0
        
        joint_pos = np.zeros((self._legNum,3))
        for i in range(len(joint_pos)):
            joint_pos[i],suc = self.inverseKinematics(tip_pos[i], 1 if i<self._legNum/2 else -1, bend_info[i])
        return joint_pos
    
    def inv_kine_bcs(self, tip_pos, bend_info):
        """inverse kinematics in body frame"""
        tp_hip = tip_pos - self.get_hip_nominal()
        return self.inv_kine_hip(tp_hip,bend_info)
    
    def inv_kine_gcs(self,bp,br,tp,bend_info):
        '''
        @brief inverse kinematics in GCS that attached to current BCS.
        @param bp : body coordinate (x,y,z) in gcs
        @param br : body rotation (alpha,beta,gamma) in gcs
        @param tp : footend positions of 6 legs in gcs
        @param bend_info: a (6,1) nparray that contains leg bend direction. 0-out, 1-in
        @return a (6,3) nparray that contains joint target position of each leg
        '''
        alpha, beta, gamma = br
        Rx = np.array([[1,0,0],[0,math.cos(alpha),-math.sin(alpha)],[0,math.sin(alpha),math.cos(alpha)]],dtype=DTYPE)
        Ry = np.array([[math.cos(beta),0,math.sin(beta)],[0,1,0],[-math.sin(beta),0,math.cos(beta)]],dtype=DTYPE)
        Rz = np.array([[math.cos(gamma),-math.sin(gamma),0],[math.sin(gamma),math.cos(gamma),0],[0,0,1]],dtype=DTYPE)
        W2B_R = Rz@Ry@Rx
        B2W_R = np.transpose(W2B_R)
        foot_pos_b = []
        for i in range(self._legNum):
            foot_pos_b_one_leg = tp[i] - bp
            tmp = B2W_R @ foot_pos_b_one_leg
            foot_pos_b.append(tmp)
        foot_pos_b = np.array(foot_pos_b)
        return self.inv_kine_bcs(foot_pos_b,bend_info)
    
    def joint_pd_controller(self,qpos,qpos_tar,qvel):
        kp = 5.8
        kd = 0.04
        tau = kp*(qpos_tar - qpos) - kd * qvel
        return tau
    