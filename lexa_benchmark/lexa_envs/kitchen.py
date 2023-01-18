import numpy as np
import gym
import random
import itertools
from itertools import combinations
from lexa_envs.base_envs import BenchEnv
from d4rl.kitchen.kitchen_envs import KitchenMicrowaveKettleLightTopLeftBurnerV0

def goal_to_task_element(goal_idx: int):
  if goal_idx == 0:
    return 'bottom left burner' #'bottom_burner'
  elif goal_idx == 1:
    return 'light switch' #'light_switch'
  elif goal_idx == 2:
    return 'slide cabinet' #'slide_cabinet'
  elif goal_idx == 3:
    return 'hinge cabinet' #'hinge_cabinet'
  elif goal_idx == 4:
    return 'microwave'
  elif goal_idx == 5:
    return 'kettle'
  else:
    assert False

def goal_to_site_str(goal_idx: int):
  if goal_idx == 0:
    return 'burner_knob' #'bottom_burner'
  elif goal_idx == 1:
    return 'lightswitch' #'light_switch'
  elif goal_idx == 2:
    return 'slidecab' #'slide_cabinet'
  elif goal_idx == 3:
    return 'hingedoor' #'hinge_cabinet'
  elif goal_idx == 4:
    return 'microwave_door'
  elif goal_idx == 5:
    return 'kettle_obj'
  else:
    assert False

class KitchenEnv(BenchEnv):
  def __init__(self, action_repeat=1, task_num=0, use_goal_idx=False, log_per_goal=False,  control_mode='end_effector', width=64):

    super().__init__(action_repeat, width)
    self.use_goal_idx = use_goal_idx
    self.log_per_goal = log_per_goal
    with self.LOCK:
      self._env =  KitchenMicrowaveKettleLightTopLeftBurnerV0(frame_skip=16, control_mode = control_mode, imwidth=width, imheight=width)
      self._env.TASK_ELEMENTS = [goal_to_task_element(task_num)]
      self._env.sim_robot.renderer._camera_settings = dict(
        distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)

    self.rendered_goal = False
    self._env.reset()
    self.init_qpos = self._env.sim.data.qpos.copy()
    self.goal_idx = task_num
    self.obs_element_goals, self.obs_element_indices, self.goal_configs = get_kitchen_benchmark_goals()
    self.goals = list(range(len(self.obs_element_goals)))

  def set_goal_idx(self, idx):
    assert False
    self.goal_idx = idx

  def get_goal_idx(self):
    return self.goal_idx

  def get_goals(self):
    return self.goals

  def _get_obs(self, state, ppc):
    image = self._env.render('rgb_array', width=self._env.imwidth, height =self._env.imheight)
    obs = {'image': image, 'state': state, 'image_goal': self.render_goal(), 'goal': self.goal, 'ppc': ppc}

    if self.log_per_goal:
      for i, goal_idx in enumerate(self.goals):
        # add rewards for all goals
        task_rel_success, all_obj_success = self.compute_success(goal_idx)
        obs['metric_success_task_relevant/goal_'+str(goal_idx)] = task_rel_success
        obs['metric_success_all_objects/goal_'+str(goal_idx)]   = all_obj_success
    if self.use_goal_idx:
      task_rel_success, all_obj_success = self.compute_success(self.goal_idx)
      obs['metric_success_task_relevant/goal_'+str(self.goal_idx)] = task_rel_success
      obs['metric_success_all_objects/goal_'+str(self.goal_idx)]   = all_obj_success

    return obs

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(action)
      ppc = self._env._get_proprioception_obs()
      reward = self.compute_reward()
      total_reward += reward
      if done:
        break
    obs = self._get_obs(state, ppc)
    for k, v in obs.items():
      if 'metric_' in k:
        info[k] = v
    return obs, total_reward, done, info

  def compute_reward(self, goal=None, add_eef_to_obj_term: bool = True):
    if goal is None:
      goal = self.goal
    qpos = self._env.sim.data.qpos.copy()

    if len(self.obs_element_indices[goal]) > 9 :
      import pdb; pdb.set_trace() # why would this happen?
      return  -np.linalg.norm(qpos[self.obs_element_indices[goal]][9:] - self.obs_element_goals[goal][9:])
    else:
      add_distance_to_reward = False

      if add_distance_to_reward:
        # norm b/w indices of interest
        element_reward = -np.linalg.norm(qpos[self.obs_element_indices[goal]] - self.obs_element_goals[goal])

        # move hand to object area
        eef_pos = self._env.get_ee_pose()
        obj_site_str = goal_to_site_str(self.goal_idx)
        obj_pos = self._env.get_site_xpos(obj_site_str)
        obj_dist = -np.linalg.norm(obj_pos - eef_pos)

        reward = obj_dist + 10 * element_reward
        # print(f"{obj_dist}, {element_reward}")
      else:
        reward = element_reward
      return reward

  def compute_success(self, goal = None):

    if goal is None:
      goal = self.goal
    qpos = self._env.sim.data.qpos.copy()

    goal_qpos = self.init_qpos.copy()
    goal_qpos[self.obs_element_indices[goal]] = self.obs_element_goals[goal]

    per_obj_success = {
    'bottom_burner' : ((qpos[9]<-0.38) and (goal_qpos[9]<-0.38)) or ((qpos[9]>-0.38) and (goal_qpos[9]>-0.38)),
    'top_burner':    ((qpos[13]<-0.38) and (goal_qpos[13]<-0.38)) or ((qpos[13]>-0.38) and (goal_qpos[13]>-0.38)),
    'light_switch':  ((qpos[17]<-0.25) and (goal_qpos[17]<-0.25)) or ((qpos[17]>-0.25) and (goal_qpos[17]>-0.25)),
    'slide_cabinet' :  abs(qpos[19] - goal_qpos[19])<0.1,
    'hinge_cabinet' :  abs(qpos[21] - goal_qpos[21])<0.2,
    'microwave' :      abs(qpos[22] - goal_qpos[22])<0.2,
    'kettle' : np.linalg.norm(qpos[23:25] - goal_qpos[23:25]) < 0.2
    }
    task_objects = self.goal_configs[goal]

    task_rel_success = 1
    for _obj in task_objects:
      task_rel_success *= per_obj_success[_obj]

    all_obj_success = 1
    for _obj in per_obj_success:
      all_obj_success *= per_obj_success[_obj]

    return int(task_rel_success), int(all_obj_success)

  def render_goal(self):
    if self.rendered_goal:
      return self.rendered_goal_obj

    # random.sample(list(obs_element_goals), 1)[0]
    backup_qpos = self._env.sim.data.qpos.copy()
    backup_qvel = self._env.sim.data.qvel.copy()

    qpos = self.init_qpos.copy()
    qpos[self.obs_element_indices[self.goal]] = self.obs_element_goals[self.goal]

    self._env.set_state(qpos, np.zeros(len(self._env.init_qvel)))

    goal_obs = self._env.render('rgb_array', width=self._env.imwidth, height=self._env.imheight)

    self._env.set_state(backup_qpos, backup_qvel)

    self.rendered_goal = True
    self.rendered_goal_obj = goal_obs
    return goal_obs

  def reset(self):
    if not self.use_goal_idx:
      self.goal_idx = np.random.randint(len(self.goals))
      # print(f"new_goal: {self.goal_idx}")
    self.goal = self.goals[self.goal_idx]

    with self.LOCK:
      self._env.TASK_ELEMENTS = [goal_to_task_element(self.goal)]
      state = self._env.reset()
      ppc = self._env._get_proprioception_obs()
    self.rendered_goal = False
    return self._get_obs(state, ppc)

def get_kitchen_benchmark_goals():
    '''
        self.init_qpos = np.array(
            [
            0    1.48388023e-01,
                -1.76848573e00,
                1.84390296e00,
                -2.47685760e00,
                2.60252026e-01,
                7.12533105e-01,
                1.59515394e00,
                4.79267505e-02,
                3.71350919e-02,
            bb    -2.66279850e-04,
        10  bb      -5.18043486e-05,
                3.12877220e-05,
                -4.51199853e-05,
                -3.90842156e-06,
                -4.22629655e-05,
                6.28065475e-05,
                4.04984708e-05,
            ls    4.62730939e-04,
            ls    -2.26906415e-04,
            sc    -4.65501369e-04,
        20  hc    -6.44129196e-03,
            hc    -1.77048263e-03,
            mw    1.08009684e-03,
            k    -2.69397440e-01,
            k    3.50383255e-01,
            k    1.61944683e00,
                1.00618764e00,
                4.06395120e-03,
                -6.62095997e-03,
                -2.68278933e-04,
            ]
    '''
    object_goal_vals = {'bottom_burner' :  [-0.88, -0.01],
                          'light_switch' :  [ -0.69, -0.05],
                          'slide_cabinet':  [0.37],
                          'hinge_cabinet':   [0., 0.5],
                          'microwave'    :   [-0.5],
                          'kettle'       :   [-0.23, 0.75, 1.62]}

    object_goal_idxs = {'bottom_burner' :  [9, 10],
                    'light_switch' :  [17, 18],
                    'slide_cabinet':  [19],
                    'hinge_cabinet':  [20, 21],
                    'microwave'    :  [22],
                    'kettle'       :  [23, 24, 25]}

    base_task_names = [ 'bottom_burner', 'light_switch', 'slide_cabinet',
                        'hinge_cabinet', 'microwave', 'kettle' ]


    goal_configs = []
    #single task
    for i in range(6):
      goal_configs.append( [base_task_names[i]])

    # let's just focus on single tasks
    # two tasks
    # for i,j  in combinations([1,2,3,5], 2) :
    #   goal_configs.append( [base_task_names[i], base_task_names[j]] )

    obs_element_goals = [] ; obs_element_indices = []
    for objects in goal_configs:
        _goal = np.concatenate([object_goal_vals[obj] for obj in objects])
        _goal_idxs = np.concatenate([object_goal_idxs[obj] for obj in objects])

        obs_element_goals.append(_goal)
        obs_element_indices.append(_goal_idxs)

    return obs_element_goals, obs_element_indices, goal_configs

