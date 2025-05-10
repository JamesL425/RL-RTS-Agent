# wrappers.py
import gymnasium as gym

class TimeLimitWrapper(gym.Wrapper):
    """
    Example wrapper to enforce a max episode length. Already handled in RTSEnv, but shown for completeness.
    """
    def __init__(self, env, max_steps=50):
        super().__init__(env)
        self.max_steps = max_steps
        self.steps = 0

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.steps += 1
        if self.steps >= self.max_steps:
            truncated = True
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)
