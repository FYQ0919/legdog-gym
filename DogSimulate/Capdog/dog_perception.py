"""
~~~~~~~~~~~~~~~~~~~~~~~~~~
Copyright @ Changda Tian, Ziqi MA
2023.6
SJTU RL2 LAB
~~~~~~~~~~~~~~~~~~~~~~~~~~

Littledog perception model

Including terrain map info, local map requirement, and sensor data fetching.

"""
import math
from pdb import post_mortem
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import cv2
from dog_constant import *

xml_name = "littledog_pd.xml"

class Point:
    def __init__(self,x,y,z) -> None:
        self.x = x
        self.y = y
        self.z = z


class Gridmap:
    def __init__(self,len=0,wid=0,res=10) -> None:
        self.Length = len
        self.Width = wid
        self.Resolution = res # how many sample points in 1m.

        self.map = np.zeros((self.Length*self.Resolution+1,self.Width*self.Resolution+1))

        self.max_height = 0
        self.map_name = "gm"
        self.xml_file = xml_name
        self.rbt_pos = np.zeros(3)
        self.rbt_dir = np.zeros(3)
        self.rbt_dir_quat = np.array([1,0,0,0])
        
    def show_map(self):
        print("\n Global map shape: ",self.map.shape)
        print(self.map)

    def set_map_name(self,map_name):
        self.map_name = map_name

    def set_xml_name(self,xml_name):
        self.xml_file = xml_name

    def __get_plane(self,p1,p2,p3):
        a = (p2.y-p1.y)*(p3.z-p1.z)-(p2.z-p1.z)*(p3.y-p1.y) 
        b = (p2.z-p1.z)*(p3.x-p1.x)-(p2.x-p1.x)*(p3.z-p1.z)
        c = (p2.x-p1.x)*(p3.y-p1.y)-(p2.y-p1.y)*(p3.x-p1.x)
        d = 0-(a*p1.x+b*p1.y+c*p1.z)
        return a,b,c,d

    def height(self,x,y):
        row_point = (x + self.Length/2) * self.Resolution
        col_point = (y + self.Width/2) * self.Resolution
        left_row_point = math.floor(row_point)
        right_row_point = left_row_point + 1
        left_col_point = math.floor(col_point)
        right_col_point = left_col_point +1
        hlu = self.map[left_row_point,left_col_point]
        hld = self.map[right_row_point,left_col_point]
        hru = self.map[left_row_point,right_col_point]
        hrd = self.map[right_row_point,right_col_point]
        # p1 = Point(left_row_point,left_col_point,hlu)
        # p2 = Point(right_row_point,left_col_point,hld)
        # p3 = Point(left_row_point,right_col_point,hru)
        # a,b,c,d = self.__get_plane(p1,p2,p3)
        # return -(d+a*row_point+b*col_point) / c
        return (hlu + hld + hru + hrd)/4
    def visualize(self):
        
        x = np.linspace(-self.Length/2,self.Length/2, self.Resolution*self.Length+1)
        y = np.linspace(-self.Width/2,self.Width/2, self.Resolution*self.Width+1)
        X,Y = np.meshgrid(x, y)
        X = X.T
        Y = Y.T
        Z = np.copy(self.map)

        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(12,10))
        ls = LightSource(270, 20)
        rgb = ls.shade(Z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
                                linewidth=0, antialiased=False, shade=False)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def gen_rand_tough_terrain(self, variance):
        tm = np.zeros((self.Length*self.Resolution+1,self.Width*self.Resolution+1))
        for i in range(self.Length*self.Resolution+1):
            tmp = np.random.random(self.Width*self.Resolution+1)
            tm[i] = tmp
        tm = tm * variance
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                self.map[i][j] = tm[i][j]
                if tm[i][j] > self.max_height:
                    self.max_height = tm[i][j]
        
    def set_block_height(self,x_range,y_range,height):
        x_left = math.floor((x_range[0]+self.Length/2)*self.Resolution)
        x_right = math.ceil((x_range[1]+self.Length/2)*self.Resolution)
        y_left = math.floor((y_range[0]+self.Width/2)*self.Resolution)
        y_right = math.ceil((y_range[1]+self.Width/2)*self.Resolution)

        for i in range(x_left,x_right+1):
            for j in range(y_left,y_right+1):
                self.map[i,j] = height
        
        if height > self.max_height:
            self.max_height = height

    def gen_random_block(self, block_num, height = 1, block_size=0.1):

        for _ in range(block_num):
            tmp_x = np.random.uniform(-0.45,0.45,1)
            tmp_y = np.random.uniform(-0.45,0.45,1)
            tmp_x *= self.Length
            tmp_y *= self.Width

            self.set_block_height([tmp_x, tmp_x + block_size], [tmp_y, tmp_y + block_size], np.random.uniform(height, height + 10, 1))

    def put_robot(self,x,y,z,alpha,beta,gamma):
        # put the robot to world.
        self.rbt_pos = np.array([x,y,z])
        self.rbt_dir = np.array([alpha,beta,gamma])
        q = euler_2_quaternion(self.rbt_dir)
        self.rbt_dir_quat = np.array([q[3],q[0],q[1],q[2]])
        print("rbt_pos: ",self.rbt_pos,"rbt_dir_quat: ",self.rbt_dir_quat)


    def parse_xml(self):
        template_xml = []
        with open("./model/"+xml_name,'r') as f:
            for i in f:
                template_xml.append(i)

        hf_pos = []
        for i in range(len(template_xml)):
            if "hfield" in template_xml[i]:
                hf_pos.append(i)
        change_string = template_xml[hf_pos[0]]
        file_index = change_string.find("file=")
        png_index = change_string.find(".png") 
        change_string = change_string[:file_index+6] + self.map_name + change_string[png_index:]

        size_index = change_string.find("size=")
        end_index = change_string.find(" />")
        size_str = f"{self.Length/2} {self.Width/2} {self.max_height} 0.001"
        change_string = change_string[:size_index+6] + size_str + change_string[end_index-1:]

        template_xml[hf_pos[0]] = change_string

        body_pos_dir_index = 0
        for i in range(len(template_xml)):
            if '''body name="base"''' in template_xml[i]:
                body_pos_dir_index = i
                break
        pos_string = template_xml[body_pos_dir_index]
        pos_index = pos_string.find("pos=")
        pos_end = pos_string.find(">")
        pos_string = pos_string[:pos_index+4] + f'''"{self.rbt_pos[0]} {self.rbt_pos[1]} {self.rbt_pos[2]+0.2}"''' + pos_string[pos_end:]

        q = euler_2_quaternion(self.rbt_dir)
        pos_end = pos_string.find(">")
        pos_string = pos_string[:pos_end] + f''' quat="{q[3]} {q[0]} {q[1]} {q[2]}">\n'''
        template_xml[body_pos_dir_index] = pos_string

        with open(f"./model/{self.xml_file}",'w') as f:
            for i in template_xml:
                f.writelines(i)

    def __min_max_normalize(self,v,min_v,max_v):
        if max_v-min_v ==0:
            return 0
        return (v-min_v) / (max_v-min_v)

    def map_to_img(self,res=100):
        # res must larger than self.Resolution and be its mutiple.
        img = np.zeros((self.Length*res+1,self.Width*res+1))
        max_h = np.max(self.map)
        min_h = np.min(self.map)
        for i in range(self.Length*res+1):
            for j in range(self.Width*res+1):
                tmp_h = self.map[math.floor(i*(self.Resolution/res)),math.floor(j*(self.Resolution/res))]
                img[i,j] = self.__min_max_normalize(tmp_h,min_h,max_h)
        img = img*255
        # rotate because in mujoco sim, the robot is horizontal view.
        img = np.rot90(img,1)
        cv2.imwrite(f'./model/{self.map_name}.png',img)
        cv2.imwrite(f'./model/meshes/{self.map_name}.png',img)

    def __min_max_denormalize(self,v,min_v,max_v):
        return min_v+v*(max_v-min_v) 

    def img_to_map(self,img_file,min_h,max_h,res=10):
        img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
        img = np.rot90(img,-1)
        for i in range(self.Length*self.Resolution+1):
            for j in range(self.Width*self.Resolution+1):
                kernel_h = int(res / self.Resolution)
                avg_mat = np.ones((kernel_h,kernel_h))
                avg_mat = avg_mat / kernel_h**2 / 255
                self.map[i,j] = self.__min_max_denormalize(np.sum(img[i*kernel_h:(i+1)*kernel_h,j*kernel_h:(j+1)*kernel_h] * avg_mat),min_h,max_h)

class Localmap:
    def __init__(self,side_len,peb,res=10) -> None:
        self.side_len = side_len
        self.peb = peb
        self.resolution = res  # how many sample points in 1m.
        self.map = np.zeros((int(self.side_len*self.resolution)+1,int(self.side_len*self.resolution)+1))

    def show_map(self):
        print("\n Local map shape: ",self.map.shape)
        print(self.map)

    def visualize(self):
        x = np.linspace(-self.side_len/2,self.side_len/2, self.resolution*self.side_len+1)
        y = np.linspace(-self.side_len/2,self.side_len/2, self.resolution*self.side_len+1)
        X,Y = np.meshgrid(x, y)
        X = X.T
        Y = Y.T
        Z = np.copy(self.map)

        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(12,10))
        ls = LightSource(270, 20)
        rgb = ls.shade(Z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
                                linewidth=0, antialiased=False, shade=False)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        plt.show()

    def __get_plane(self,p1,p2,p3):
        a = (p2.y-p1.y)*(p3.z-p1.z)-(p2.z-p1.z)*(p3.y-p1.y) 
        b = (p2.z-p1.z)*(p3.x-p1.x)-(p2.x-p1.x)*(p3.z-p1.z)
        c = (p2.x-p1.x)*(p3.y-p1.y)-(p2.y-p1.y)*(p3.x-p1.x)
        d = 0-(a*p1.x+b*p1.y+c*p1.z)
        return a,b,c,d

    def height(self,x,y):
        row_point = (x + self.side_len/2) * self.resolution
        col_point = (y + self.side_len/2) * self.resolution
        left_row_point = math.floor(row_point)
        right_row_point = left_row_point + 1
        left_col_point = math.floor(col_point)
        right_col_point = left_col_point +1
        hlu = self.map[left_row_point,left_col_point]
        hld = self.map[right_row_point,left_col_point]
        hru = self.map[left_row_point,right_col_point]
        # hrd = self.map[right_row_point,right_col_point]
        p1 = Point(left_row_point,left_col_point,hlu)
        p2 = Point(right_row_point,left_col_point,hld)
        p3 = Point(left_row_point,right_col_point,hru)
        a,b,c,d = self.__get_plane(p1,p2,p3)
        return -(d+a*row_point+b*col_point) / c

    def get_local_map(self,g_map):
        x_g = self.peb[0]
        y_g = self.peb[1]
        gamma = self.peb[-1]
        R_b2w = np.array([[math.cos(gamma), -math.sin(gamma)],
                            [math.sin(gamma),  math.cos(gamma)]])
        x = np.linspace(-self.side_len/2,self.side_len/2, int(self.resolution*self.side_len)+1)
        y = np.linspace(-self.side_len/2,self.side_len/2, int(self.resolution*self.side_len)+1)
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                temp_x,temp_y = R_b2w@np.array([x[i],y[j]]).T + np.array([x_g, y_g]).T
                self.map[i,j] = g_map.height(temp_x,temp_y)



class Littledog_Perception:
    def __init__(self,g_map,mujoco_model,mujoco_data) -> None:
        self.global_map = g_map
        self.local_map = None
        self.local_map_side_len = 3.1

        self.model = mujoco_model
        self.data = mujoco_data

        self.body_pos = np.zeros(6)
        self.bend = np.zeros(6)
        self.feet_pos_gcs = np.zeros((6,3))
        self.feet_pos_bcs = np.zeros((6,3))

        self.feet_force = np.zeros(6)
        
        self.body_pos = self.get_body_pos()
        self.feet_pos_gcs = self.gcs_get_pee()
        self.feet_pos_bcs = self.bcs_get_pee()
        self.get_local_map()

    def get_body_pos(self):
        tp = np.copy(self.data.qpos[:3])
        tq = np.copy(self.data.qpos[3:7])
        te = quaternion_2_euler(tq)
        return np.concatenate((tp,te),axis=0)

    def get_jp(self):
        return np.copy(self.data.qpos[7:].reshape((6,3)))

    def gcs_get_pee(self):
        pee = np.zeros((6,3))
        for i in range(6):
            pee[i] = np.array(self.data.sensor(f"L{i}_tip_pos_gcs").data)
        return pee

    def bcs_get_pee(self):
        pee = np.zeros((6,3))
        for i in range(6):
            pee[i] = np.array(self.data.sensor(f"L{i}_tip_pos_bcs").data)
        return pee
    
    def bcs_get_pee_no_rotate(self):
        pee = self.gcs_get_pee()
        bp = self.get_body_pos()
        bp_xyz = bp[:3]
        bp_z_rotate = bp[5]
        bp_y_rotate = bp[4]
        bp_x_rotate = bp[3]
        R_z = np.array([[math.cos(bp_z_rotate),-math.sin(bp_z_rotate),0],[math.sin(bp_z_rotate),math.cos(bp_z_rotate),0],[0,0,1]])
        R_y = np.array([[math.cos(bp_y_rotate),0,math.sin(bp_y_rotate)],[0,1,0],[-math.sin(bp_y_rotate),0,math.cos(bp_y_rotate)]])
        R_x = np.array([[1,0,0],[0,math.cos(bp_x_rotate),-math.sin(bp_x_rotate)],[0,math.sin(bp_x_rotate),math.cos(bp_x_rotate)]])
        R = R_z@R_y@R_x
        res_pee = np.copy(pee)
        for i in range(6):
            res_pee[i] = R.T @ (pee[i] - bp_xyz)
        return res_pee
    
    def bcs_get_pee_no_rotate_z_in_gcs(self):
        pee = self.bcs_get_pee_no_rotate()
        gcs_pee = self.gcs_get_pee()
        for i in range(len(pee)):
            pee[i][2] = gcs_pee[i][2]
        return pee
    
    def bcs_get_pee_xy_no_rotate(self):
        pee = self.bcs_get_pee_no_rotate()
        pee = pee[:,:-1]
        return pee

    def bcs_get_hip(self):
        hip = np.zeros((6,3))
        for i in range(6):
            hip[i] = np.array(self.data.sensor(f"L{i}_hip_pos_bcs").data)
        return hip

    def gcs_get_hip(self):
        hip = np.zeros((6,3))
        for i in range(6):
            hip[i] = np.array(self.data.sensor(f"L{i}_hip_pos_gcs").data)
        return hip

    def get_body_acc(self):
        acc = np.array(self.data.sensor("body_acc").data)
        return acc
    
    def get_body_vel(self):
        velo = np.array(self.data.sensor("body_velo").data)
        return velo

    def get_tip_force(self):
        fc = np.zeros(6)
        for i in range(6):
            fc[i] = self.data.sensor(f"L{i}_touch_force").data
        return fc

    def bcs_xy_proj_get_peb(self):
        peb = self.get_body_pos()
        peb[0] = 0
        peb[1] = 0
        peb[5] = 0
        return peb

    def bcs_get_pee_z_in_gcs(self):
        pee = self.bcs_get_pee()
        gcs_pee = self.gcs_get_pee()
        for i in range(len(pee)):
            pee[i][2] = gcs_pee[i][2]
        return pee

    def bcs_get_pee_xy(self):
        pee = self.bcs_get_pee()
        pee = pee[:,:-1]
        return pee

    def get_bend(self):
        res = np.zeros(6)
        res = self.bend[:]
        jp = self.get_jp()
        for i in range(6):
            if jp[i][2] < -0.01:
                res[i] = 1
            if jp[i][2] > 0.01:
                res[i] = 0
        return res

    def get_local_map(self):
        self.local_map = Localmap(3.1,self.get_body_pos())
        self.local_map.get_local_map(self.global_map)
    
