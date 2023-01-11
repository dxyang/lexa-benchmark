import time

import cv2
import numpy as np

from lexa_envs.kitchen import KitchenEnv

def run():
    env = KitchenEnv()
    env.reset()

    print(f"===== Observation Space =====")
    print(env.observation_space)
    print(f"===== Action Space =====")
    print(env.action_space)

    print(f"===== Simulating =====")
    start = time.time()
    for i in range(1000):
        obs, total_reward, done, info = env.step(env.action_space.sample())
        if i == 0:
            import pdb; pdb.set_trace()
            cv2.imwrite("start_image.png", cv2.cvtColor(obs["image"], cv2.COLOR_BGR2RGB))
            cv2.imwrite("goal_image.png", cv2.cvtColor(obs["image_goal"], cv2.COLOR_BGR2RGB))
            same_image = np.allclose(obs["image"], obs["image_goal"])
            for k, v in obs.items():
                if type(v) == np.ndarray:
                    print(f"{k}: {v.shape}")
                else:
                    print(f"{k}: {type(v)}")
        # env.render(mode="human")

        if done:
            print(f"hurrah")
        if i % 100 == 0 and i > 0:
            now = time.time()
            print(f"{i} steps done in {now - start} seconds. {(now - start) / i} fps")

    end = time.time()
    print(f"1000 steps done in {end - start} seconds. {(end - start) / 1000.0} fps")
    print(f"finished!")

if __name__ == "__main__":
    supercloud = False
    if supercloud:

        import jaynes

        jaynes.config()
        jaynes.add(run)
        jaynes.execute()
        jaynes.listen()
    else:
        run()
