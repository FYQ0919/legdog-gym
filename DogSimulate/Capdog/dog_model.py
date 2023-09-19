"""
~~~~~~~~~~~~~~~~~~~~~~~~~~
Copyright @ Changda Tian
2023.6
SJTU RL2 LAB
~~~~~~~~~~~~~~~~~~~~~~~~~~

Little dog control model

"""
import time

from dog_perception import *
np.set_printoptions(threshold=np.inf)

class Littledog:
    def __init__(self,g_map,viewer) -> None:
        # define mujoco items 
        self.model = mujoco.MjModel.from_xml_path("./model/littledog_pd.xml")
        # self.model = mujoco.MjModel.from_xml_path(f"./model/{g_map.xml_file}")
        self.data = mujoco.MjData(self.model)
        if viewer == True:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        else:
            self.viewer = None
        # define simulation frames of an action
        self.steps_of_an_action = 2000
        
        # define qz2 perception model
        self.perception = Littledog_Perception(g_map,self.model,self.data)

        self.rbt = Robot(RobotType.LITTLE_DOG)

        # define robot state items
        self.jp = np.zeros((6,3))
        self.step_height = 0.15
        self.step_len = 0
        
        # define JS controller items

        # self.delta_t = 0.005
        # self.js = XboxController()
        
    def step(self,ctrl):
        ct = np.zeros(len(self.data.ctrl))
        ct[:len(self.data.ctrl)//2] = ctrl
        self.data.ctrl[:] = ct
        mujoco.mj_step(self.model,self.data)

    def sim_render(self):
        if self.viewer and self.viewer.is_alive:
            self.viewer.render()

    def update_jp(self):
        self.jp = np.copy(self.perception.get_jp())

    def close_viewer(self):
        if self.viewer:
            self.viewer.close()

    def joint_position_control(self,target_jp):
        """control the jp to your target jp."""
        t_jp = target_jp.flatten()
        self.step(t_jp)

    def current_joint_keep_control(self):
        """keep current jp."""
        t_jp = self.jp.flatten()
        self.step(t_jp)

    def steady(self,frame_num):
        """keep current jp and steady the robot for given frame num."""
        for i in range(frame_num):
            self.current_joint_keep_control()
            self.sim_render()

    def set_pee_bcs(self,tip_tar_pos):
        """set feet pos in body frame."""
        jp = self.rbt.inv_kine_bcs(tip_tar_pos,self.perception.bend)
        self.joint_position_control(jp)
        self.update_jp()

    def set_pee_bcs_activate_leg(self,bcs_tip_tar_pos,activate_leg,cur_jp):
        jp = np.copy(cur_jp)
        tar_jp = self.rbt.inv_kine_bcs(bcs_tip_tar_pos,self.perception.get_bend())
        for i in activate_leg:
            jp[i] = tar_jp[i]
        self.joint_position_control(jp)
        self.update_jp()

    def set_pee_gcs(self,body_pr,tip_tar_pos):
        """set feet pos in world frame oriented from current bcs."""
        bp = body_pr[:3]
        br = body_pr[3:]
        
        jp = self.rbt.inv_kine_gcs(bp,br,tip_tar_pos,self.perception.bend)
        # print(f"target jp {jp}")
        self.joint_position_control(jp)
        self.update_jp()

        # print(f"current jp {self.jp}")

    def walk_one_step_to(self, body_xy_target, tip_xy_target, side=0):
        '''
        @brief Walk one step to target body_pos xy and tip xy pos in this time bcs-attached gcs frame.
        @brief Body z , alpha, beta and tip z are concluded by local terrain. Swing traj are sine shaped.
        
        @param body_xy_target : (1,2) target body pos.
        @param tip_xy_target : (6,2) target tip pos.

        '''
        print(tip_xy_target)
        step_h = self.step_height
        interval_points_num = self.steps_of_an_action

        target_bp = np.zeros(6)
        target_tp = np.zeros((6,3))
        
        # set tip pos target
        for i in range(6):
            target_tp[i][0] = tip_xy_target[i][0]
            target_tp[i][1] = tip_xy_target[i][1]
            target_tp[i][2] = self.perception.local_map.height(tip_xy_target[i][0],tip_xy_target[i][1]) + self.rbt._foot_ball_rad

        # alpha : x-axis rotation
        delta_z = target_tp[4][2] - target_tp[1][2]
        delta_y = target_tp[4][1]  - target_tp[1][1]
        alpha = math.atan2(delta_z,delta_y)

        # beta : y-axis rotation
        delta_z = (target_tp[0][2] + target_tp[5][2]) - (target_tp[3][2] + target_tp[2][2])
        delta_x = (target_tp[0][0] + target_tp[5][0]) - (target_tp[3][0] + target_tp[2][0])
        beta = - math.atan2(delta_z,delta_x)

        # set body pos target
        target_bp[0] = body_xy_target[0]
        target_bp[1] = body_xy_target[1]
        target_bp[2] = self.rbt._bodyHeight + self.perception.local_map.height(body_xy_target[0],body_xy_target[1])
        target_bp[3] = alpha
        target_bp[4] = beta

        current_bp = self.perception.get_body_pos()
        current_bp[0] = 0
        current_bp[1] = 0

        current_tp = self.perception.bcs_get_pee_no_rotate_z_in_gcs()

        if side == 0:
            first_move_leg = [0,2,4]
            second_move_leg = [1,3,5]
        else:
            second_move_leg = [0,2,4]
            first_move_leg = [1,3,5]

        # body move list
        body_x = np.linspace(current_bp[0],target_bp[0],2*interval_points_num+1)[1:]
        body_y = np.linspace(current_bp[1],target_bp[1],2*interval_points_num+1)[1:]
        body_z = np.linspace(current_bp[2],target_bp[2],2*interval_points_num+1)[1:]
        body_a = np.linspace(current_bp[3],target_bp[3],2*interval_points_num+1)[1:]
        body_b = np.linspace(current_bp[4],target_bp[4],2*interval_points_num+1)[1:]
        
        # tip z traj
        tip_z = np.sin(np.linspace(0,math.pi,interval_points_num+1)[1:])*step_h
        # leg move list
        leg_traj = np.zeros((6,interval_points_num,3))
        for i in range(6):
            leg_traj[i] = np.linspace(current_tp[i],target_tp[i],interval_points_num+1)[1:]

        command_bp = np.zeros((2*interval_points_num,6))
        command_tp = np.zeros((2*interval_points_num,6,3))
        command_tp[:] = np.copy(current_tp)
        for i in range(2*interval_points_num):
            command_bp[i] = np.array([body_x[i],body_y[i],body_z[i],body_a[i],body_b[i],0])
        
        for leg in first_move_leg:
            for i in range(interval_points_num):
                command_tp[i][leg][0] = leg_traj[leg][i][0]
                command_tp[i][leg][1] = leg_traj[leg][i][1]
                command_tp[i][leg][2] = leg_traj[leg][i][2] + tip_z[i]
        command_tp[interval_points_num:] = np.copy(command_tp[interval_points_num-1])
        for leg in second_move_leg:
            for i in range(interval_points_num,2*interval_points_num):
                command_tp[i][leg][0] = leg_traj[leg][i-interval_points_num][0]
                command_tp[i][leg][1] = leg_traj[leg][i-interval_points_num][1]
                command_tp[i][leg][2] = leg_traj[leg][i-interval_points_num][2] + tip_z[i-interval_points_num]
        # walk
        # print(command_bp[-1],command_tp[-1,0])
        for i in range(2*interval_points_num):
            self.set_pee_gcs(command_bp[i],command_tp[i])
            self.sim_render()
        self.perception.get_local_map()

    def walk_to(self, target_xy, step_len):

        while np.linalg.norm(self.perception.get_body_pos()[:2] - target_xy) > 0.2:
            current_tp_xy = self.perception.bcs_get_pee_xy_no_rotate()
            current_bp = self.perception.get_body_pos()

            last_pos = self.perception.get_body_pos()[:2]

            delta_x = target_xy[0] - current_bp[0]
            delta_y = target_xy[1] - current_bp[1]

            step_num = int(abs(delta_x) // step_len)

            delta_x_each_step = (1 if delta_x > 0 else -1) * step_len
            delta_y_each_step = (1 if delta_x > 0 else -1) * step_len * abs(delta_y / delta_x)

            target_tp_xy = np.copy(current_tp_xy)
            for i in range(6):
                target_tp_xy[i][0] = current_tp_xy[i][0] + delta_x_each_step
                target_tp_xy[i][1] = current_tp_xy[i][1] + delta_y_each_step

            for _ in range(step_num):
                try:
                    self.walk_one_step_to([delta_x_each_step, delta_y_each_step], target_tp_xy)
                    self.perception.local_map.show_map()
                    if np.linalg.norm(self.perception.get_body_pos()[:2] - last_pos) > 1:
                        break
                except IndexError:
                    return False

        # last step walk precisely to the target xy
        current_bp = self.perception.get_body_pos()
        delta_x = target_xy[0] - current_bp[0]
        delta_y = target_xy[1] - current_bp[1]

        target_tp_xy = self.perception.bcs_get_pee_xy_no_rotate()
        for i in range(6):
            target_tp_xy[i][0] += delta_x
            target_tp_xy[i][1] += delta_y
        try:
            self.walk_one_step_to([delta_x, delta_y], target_tp_xy)
            print(f"final pos: {self.perception.get_body_pos()[0:2]}")
        except IndexError:
            return False
        return True
    
    def recover_stand_nominal(self):
        s0 = self.rbt.get_feet_nominal()[:,:-1]
        bp_t = [0,0]
        self.walk_one_step_to(bp_t,s0)

    def tn_once(self,dir,angle=10,side=0):
        tn_target_theta = dir * deg2rad(angle)
        current_tp = self.perception.bcs_get_pee_no_rotate_z_in_gcs()
        
        first_leg_group = [0,2,4] if side == 0 else [3,1,5]
        second_leg_group = [3,1,5] if side == 0 else [0,2,4]

        # get rotation radius for each foot
        current_r = np.zeros(6)
        for leg_id in range(6):
            current_r[leg_id] = math.sqrt(current_tp[leg_id][0]**2+current_tp[leg_id][1]**2)

        # get init angle for each foot
        current_angle = np.zeros(6)
        for leg_id in range(6):
            current_angle[leg_id] = math.atan2(current_tp[leg_id][1],current_tp[leg_id][0])

        phase_1_frame = self.steps_of_an_action     # for first group of legs, say 0,2,4
        phase_2_frame = self.steps_of_an_action     # for second group of legs, 1,3,5
        this_step_frame = phase_1_frame + phase_2_frame

        # target pee of each frame
        target_tp = np.zeros((this_step_frame,6,3))

        # phase 1 target pee
        for leg in first_leg_group:

            # update x by adding r*cos(wt) 
            dx = current_r[leg]*(math.cos(tn_target_theta+current_angle[leg]) - math.cos(current_angle[leg]))
            x = np.linspace(0,dx,phase_1_frame+1)[1:]

            # update y by adding r*sin(wt)
            dy = current_r[leg]*(math.sin(tn_target_theta+current_angle[leg]) - math.sin(current_angle[leg]))
            y = np.linspace(0,dy,phase_1_frame+1)[1:]

            # update z by step_height 
            dz = self.perception.local_map.height(current_tp[leg][0]+dx,current_tp[leg][1]+dy) + self.rbt._foot_ball_rad - current_tp[leg][2]
            z_1 = np.linspace(0,dz,phase_1_frame+1)[1:]
            z_2 = np.sin(np.linspace(0,math.pi,phase_1_frame+1)[1:])*self.step_height
            z = z_1 + z_2
            # z = np.sin(np.linspace(0,math.pi,phase_1_frame+1)[1:])*self.step_height

            for i in range(phase_1_frame):
                target_tp[i][leg] = current_tp[leg] + np.array([x[i],y[i],z[i]])
        

        for leg in second_leg_group:
            for i in range(phase_1_frame):
                target_tp[i][leg] = current_tp[leg]

        # phase 2 target pee
        for leg in second_leg_group:

            # update x by adding r*cos(wt) 
            dx = current_r[leg]*(math.cos(tn_target_theta+current_angle[leg]) - math.cos(current_angle[leg]))
            x = np.linspace(0,dx,phase_2_frame+1)[1:]

            # update y by adding r*sin(wt)
            dy = current_r[leg]*(math.sin(tn_target_theta+current_angle[leg]) - math.sin(current_angle[leg]))
            y = np.linspace(0,dy,phase_2_frame+1)[1:]

            # update z by step_height 
            dz = self.perception.local_map.height(current_tp[leg][0]+dx,current_tp[leg][1]+dy) + self.rbt._foot_ball_rad - current_tp[leg][2]
            z_1 = np.linspace(0,dz,phase_2_frame+1)[1:]
            z_2 = np.sin(np.linspace(0,math.pi,phase_2_frame+1)[1:])*self.step_height
            z = z_1 + z_2
            # z = np.sin(np.linspace(0,math.pi,phase_2_frame+1)[1:])*self.step_height

            for i in range(phase_1_frame,this_step_frame):
                target_tp[i][leg] = current_tp[leg] + np.array([x[i-phase_1_frame],y[i-phase_1_frame],z[i-phase_1_frame]])

        for leg in first_leg_group:
            for i in range(phase_1_frame,this_step_frame):
                target_tp[i][leg] = target_tp[phase_1_frame-1][leg]


        # calculate target peb of each frame
        target_peb = np.zeros((this_step_frame,6))
        init_peb = np.zeros(6)
        init_peb[2] = self.rbt._bodyHeight

        z1 = np.sin(np.linspace(0,math.pi,phase_1_frame+1)[1:])*self.step_height/6
        z2 = np.sin(np.linspace(0,math.pi,phase_2_frame+1)[1:])*self.step_height/6

        dc = tn_target_theta
        c = np.linspace(0,dc,this_step_frame+1)[1:]

        for i in range(phase_1_frame):
            target_peb[i] = init_peb + np.array([0,0,z1[i],0,0,c[i]])
        for i in range(phase_1_frame,this_step_frame):
            target_peb[i] = init_peb + np.array([0,0,z2[i-phase_1_frame],0,0,c[i]])

        # take action
        for st in range(this_step_frame):
            self.set_pee_gcs(target_peb[st],target_tp[st])
            self.sim_render()
        
        self.perception.get_local_map()


    def pre_tn(self):
        """
        make body parallel to the ground.
        """
        current_pee = self.perception.gcs_get_pee()
        current_peb = self.perception.get_body_pos()
        target_peb = np.copy(current_peb)
        target_peb[3] = 0
        target_peb[4] = 0
        
        delta_peb = target_peb - current_peb
        delta_peb_per_frame = delta_peb / self.steps_of_an_action

        for i in range(self.steps_of_an_action):
            current_peb = current_peb + delta_peb_per_frame
            self.set_pee_gcs(current_peb,current_pee)
            self.sim_render()

    def after_tn(self):
        """
        let footend to nominal pos.
        """
        target_pee = self.perception.bcs_get_pee_z_in_gcs()

        for i in range(len(target_pee)):
            target_pee[i][1] = self.rbt.get_feet_nominal()[i][1]

        self.walk_one_step_to([0,0],target_pee[:,:-1])

    def tn(self,dir,angle,side=0):
        """
        feedback control to target angle.
        """
        self.pre_tn()

        current_peb = self.perception.get_body_pos()
        delta_angle = dir * angle
        current_angle = current_peb[5]*180/math.pi + 4*360
        target_angle = delta_angle + current_angle
        delta_a = target_angle - current_angle

        while abs(delta_a) > 3:
            if abs(delta_a) > 10:
                self.tn_once(dir,10,side)
            else:
                self.tn_once(1,delta_a,side)
            current_angle = self.perception.get_body_pos()[5]*180/math.pi + 4*360
            delta_a = target_angle - current_angle
            if delta_a<-180:
                delta_a += 360
            if delta_a>180:
                delta_a -= 360
            
        self.after_tn()

    def simulate(self):
        self.steady(100)
        target_pee = np.copy(self.rbt.get_feet_nominal())
        # target_peb = np.zeros(6)
        # target_peb[0] = 0.05
        print("bend",self.perception.get_bend())
        print("current_pee ", self.perception.bcs_get_pee())
        print("target_pee ",target_pee)

        for i in range(300):
            self.set_pee_bcs(target_pee)
            self.sim_render()
        self.steady(100)
        # self.tn(1,7)
        # for i in range(300):
        #     self.set_pee_gcs(target_peb,target_pee)
        #     self.sim_render()
        self.walk_to((-5,0),0.15)
        print("walk finished")
        self.recover_stand_nominal()

        self.steady(1000)
        self.close_viewer()

if __name__ == "__main__":
    # img = cv2.imread("./model/gm.png",cv2.IMREAD_GRAYSCALE)
    # print((np.max(img)))
    g_map = Gridmap(20,8)
    # g_map.gen_rand_tough_terrain(0)
    g_map.gen_random_block(100, height=1)
    g_map.map_to_img(res=10)
    # g_map.img_to_map("./model/gm.png", 0, 1)

    g_map.set_map_name("gm")
    g_map.set_xml_name("three_pd.xml")

    g_map.parse_xml()
    # g_map.img_to_map("./model/gm.png", min_h=0, max_h=0.1)
    #
    #
    #
    # # g_map.visualize()
    #
    qz2 = Littledog(g_map,viewer=True)
    qz2.simulate()

    