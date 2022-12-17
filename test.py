import numpy as np

from lexa_envs.kitchen import KitchenEnv

env = KitchenEnv()
env.reset()

print(f"===== Observation Space =====")
print(env.observation_space)
print(f"===== Action Space =====")
print(env.action_space)

print(f"===== Simulating =====")
for i in range(1000):
    obs, total_reward, done, info = env.step(env.action_space.sample())
    if i == 0:
        for k, v in obs.items():
            if type(v) == np.ndarray:
                print(f"{k}: {v.shape}")
            else:
                print(f"{k}: {type(v)}")
    env.render(mode="human")
    if done:
        print(f"hurrah")