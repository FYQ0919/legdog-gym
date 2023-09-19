"""
~~~~~~~~~~~~~~~~~~~~~~~~~~
Copyright @ Changda Tian
2023.6
SJTU RL2 LAB
~~~~~~~~~~~~~~~~~~~~~~~~~~

The mujoco virtual simulation environment for Littledog.
"""
from dog_model import *

class QZ_GYM:
    '''Virtual env for training dog robot'''
    def __init__(self,env_type,env_roughness,env_size,rand_seed) -> None:
        self.env_type = env_type
        self.env_roughness = env_roughness
        self.env_size = env_size
        self.rand_seed = rand_seed
        np.random.seed(self.rand_seed)

        self.world = Gridmap(env_size,env_size)
        self.world.set_map_name(self.env_type)
        self.world.set_xml_name(f'{self.env_type}_qz.xml')

        self.world.gen_rand_tough_terrain(self.env_roughness)

        self.action_space = np.array([0,1,2,3,4,5,6,7])
        self.action_space_shape = 8

        self.observation_space_shape = 32**2 + 18
        self.world.map_to_img()
        self.world.put_robot(0,0,body_h_normal+self.world.height(0,0),0,0,0)
        self.world.parse_xml()
        

    def reset(self):
        rbt_x = np.random.random()*2 - 1
        rbt_y = np.random.random()*2 - 1
        rbt_gamma = np.random.random()*360 - 180
        self.world.put_robot(rbt_x,rbt_y,body_h_normal+self.world.height(rbt_x,rbt_y),0,0,rbt_gamma)
        self.world.parse_xml()
        self.qz_rbt = Littledog(self.world,viewer=False)
        self.qz_rbt.set_init_state()
        self.qz_rbt.set_init_state_with_terrain()
        self.qz_rbt.steady(10)
        terrain_info = np.copy(self.qz_rbt.perception.local_map.map)
        terrain_info = terrain_info.flatten()
        pos_ee = self.qz_rbt.perception.bcs_get_pee().flatten()
        state = np.concatenate((terrain_info,pos_ee))
        return state

    def _success(self,old_pos,old_gamma):
        d = 1
        cur_x, cur_y = self.qz_rbt.perception.get_body_pos()[:2]
        old_x, old_y = old_pos[:2]
        nx = old_x + d*math.cos(old_gamma)
        ny = old_y + d*math.sin(old_gamma)
        dis = math.sqrt((nx-cur_x)**2 + (ny-cur_y)**2)
        if dis<=0.2:
            return True
        else:
            return False


    def step(self,action):
        """define your own step, let the robot do your action in simulation and set your own reward, done, other_info."""
        next_state = None
        reward = None
        done = 0
        other_info = None

        return next_state,reward,done,other_info

