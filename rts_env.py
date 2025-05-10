# rts_env.py - Gold Rush Environment with improved rendering
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys

try:
    import pygame
except ImportError:
    pygame = None  # only needed for render_mode="pygame"


class GoldRushEnv(gym.Env):
    """
    Two teams (Blue & Red) with N agents each race to collect gold nuggets.
    Both teams are treated identically for perfect symmetry.
    
    Actions:
        Each team provides actions as ndarray shape (N,) with ints 0..4 
        (0=up, 1=right, 2=down, 3=left, 4=stay)
    
    Observation:
        Channels C = 2*N + 1: First N channels for current team's agents, 
        next N for opponent team's agents, last channel for gold.
        Shape = (C, grid, grid) with 0/1 values.
    
    Reward:
        Sparse, delivered at episode end: +1 to winning team, -1 to loser, 0 if tie.
    """
    metadata = {"render_modes": ["pygame"], "render_fps": 6}

    def __init__(
        self,
        grid_size: int = 5,
        num_agents: int = 2,
        num_gold: int = 3,
        max_steps: int = 40,
        render_mode: str | None = None,
        opponent_policy=None,
    ):
        super().__init__()
        self.G = grid_size
        self.N = num_agents
        self.K = num_gold
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.opponent_policy = opponent_policy

        # Each agent chooses among 5 moves
        self.action_space = spaces.MultiDiscrete([5] * self.N)
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(2 * self.N + 1, self.G, self.G), dtype=np.float32
        )

        # Pygame handles
        self.window = self.clock = self.font = None
        self.cell = 96  # Increased tile size for better visibility
        self.reset()

    def reset(self, *, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        # Place agents and gold randomly on the grid
        coords = [(x, y) for x in range(self.G) for y in range(self.G)]
        self.np_random.shuffle(coords)
        
        self.blue_pos = [coords[i] for i in range(self.N)]
        self.red_pos = [coords[self.N + i] for i in range(self.N)]
        self.gold = coords[2 * self.N : 2 * self.N + self.K]
        self.steps = 0
        
        # Scores for each team (used at end for rewards)
        self.blue_score = 0
        self.red_score = 0
        
        # Initialize pygame if rendering
        if self.render_mode == "pygame" and pygame is not None and self.window is None:
            self._init_pygame()
            
        return self._get_obs(), {}

    def step(self, blue_actions):
        """
        Take a step in the environment. Both teams act simultaneously.
        
        Args:
            blue_actions: Array of actions for blue team
            
        Returns:
            obs: New observation
            reward: Reward for blue team
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional info including red team's reward
        """
        blue_actions = np.asarray(blue_actions, dtype=int).reshape(-1)[: self.N]

        # Get red team actions from policy
        if self.opponent_policy is None:
            red_actions = np.random.randint(5, size=self.N)
        else:
            # Get observation from red's perspective (swap blue and red channels)
            red_obs = self._get_obs_for_red()
            try:
                # The policy gets the red observation, so it acts as if it's blue
                red_actions, *_ = self.opponent_policy(red_obs)
                red_actions = np.asarray(red_actions, dtype=int).reshape(-1)[:self.N]
            except Exception as e:
                print(f"Error getting red actions: {e}")
                red_actions = np.random.randint(5, size=self.N)

        # Compute next positions for both teams (no mutation yet)
        blue_next = [self._move(pos, a) for pos, a in zip(self.blue_pos, blue_actions)]
        red_next = [self._move(pos, a) for pos, a in zip(self.red_pos, red_actions)]

        # Apply moves simultaneously
        self.blue_pos = blue_next
        self.red_pos = red_next

        # Gold collection (simultaneous)
        collected_by_blue = {p for p in self.blue_pos if p in self.gold}
        collected_by_red = {p for p in self.red_pos if p in self.gold}

        # Update scores
        self.blue_score += len(collected_by_blue)
        self.red_score += len(collected_by_red)

        # Remove gold taken by either team
        self.gold = [g for g in self.gold if g not in collected_by_blue | collected_by_red]

        # Check for termination
        self.steps += 1
        terminated = (not self.gold) or (self.steps >= self.max_steps)
        truncated = False
        
        # Calculate rewards (only at episode end)
        r_blue = r_red = 0.0
        if terminated:
            if self.blue_score > self.red_score:
                r_blue, r_red = 1.0, -1.0
            elif self.red_score > self.blue_score:
                r_blue, r_red = -1.0, 1.0

        return self._get_obs(), r_blue, terminated, truncated, {"r_red": r_red}

    def _move(self, pos, a):
        """Helper function to compute new position after action."""
        x, y = pos
        if a == 0 and y > 0:  # Up
            y -= 1
        elif a == 1 and x < self.G - 1:  # Right
            x += 1
        elif a == 2 and y < self.G - 1:  # Down
            y += 1
        elif a == 3 and x > 0:  # Left
            x -= 1
        # a == 4 is stay
        return (x, y)

    def _get_obs(self):
        """
        Get observation from blue team's perspective.
        First N channels: blue agents
        Next N channels: red agents
        Last channel: gold
        """
        C = 2 * self.N + 1
        obs = np.zeros((C, self.G, self.G), dtype=np.float32)
        
        # Blue team (first N channels)
        for i, (x, y) in enumerate(self.blue_pos):
            obs[i, y, x] = 1.0
            
        # Red team (next N channels)
        for i, (x, y) in enumerate(self.red_pos):
            obs[self.N + i, y, x] = 1.0
            
        # Gold (last channel)
        for (x, y) in self.gold:
            obs[-1, y, x] = 1.0
            
        return obs

    def _get_obs_for_red(self):
        """
        Get observation from red team's perspective.
        First N channels: red agents
        Next N channels: blue agents
        Last channel: gold
        """
        obs = self._get_obs()
        # Swap blue and red channels
        blue_channels = obs[:self.N].copy()
        red_channels = obs[self.N:2*self.N].copy()
        obs[:self.N] = red_channels
        obs[self.N:2*self.N] = blue_channels
        return obs

    def _get_flat_obs(self):
        """Return flattened observation."""
        return self._get_obs().flatten()
    
    def _init_pygame(self):
        """Initialize pygame for rendering."""
        if pygame is None:
            return
            
        pygame.init()
        screen_width = self.G * self.cell
        screen_height = self.G * self.cell + 60  # Extra space for text
        self.window = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Gold Rush")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 32)  # Larger font for bigger window

    def render(self):
        """Render the environment using pygame."""
        if self.render_mode != "pygame":
            return
        if pygame is None:
            raise RuntimeError("pygame not installed")
            
        if self.window is None:
            self._init_pygame()

        # Handle events (including graceful exit)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.window = None
                print("\nPygame window closed. Exiting...")
                sys.exit(0)  # Exit the program completely

        # Clear the screen
        self.window.fill((20, 20, 30))  # Darker background

        # Draw the board
        for y in range(self.G):
            for x in range(self.G):
                clr = (40, 40, 50) if (x + y) % 2 else (50, 50, 60)
                pygame.draw.rect(self.window, clr, 
                               (x * self.cell, y * self.cell, self.cell, self.cell))
                
        # Draw grid lines
        for i in range(self.G + 1):
            # Vertical lines
            pygame.draw.line(self.window, (70, 70, 80), 
                           (i * self.cell, 0), 
                           (i * self.cell, self.G * self.cell))
            # Horizontal lines
            pygame.draw.line(self.window, (70, 70, 80), 
                           (0, i * self.cell), 
                           (self.G * self.cell, i * self.cell))
                
        # Draw gold
        for x, y in self.gold:
            pygame.draw.circle(self.window, (255, 215, 0),
                             (x * self.cell + self.cell // 2, 
                              y * self.cell + self.cell // 2),
                             self.cell // 3)
                
        # Draw agents (blue)
        for x, y in self.blue_pos:
            pygame.draw.circle(self.window, (0, 160, 255),  # Brighter blue
                             (x * self.cell + self.cell // 2, 
                              y * self.cell + self.cell // 2),
                             self.cell // 2 - 6)  # Slightly smaller than cell
                
        # Draw agents (red)
        for x, y in self.red_pos:
            pygame.draw.circle(self.window, (255, 60, 180),  # Brighter pink/red
                             (x * self.cell + self.cell // 2, 
                              y * self.cell + self.cell // 2),
                             self.cell // 2 - 6)  # Slightly smaller than cell
                
        # Draw info text
        score_txt = f"Blue: {self.blue_score} | Red: {self.red_score}"
        step_txt = f"Step {self.steps}/{self.max_steps} – Gold left: {len(self.gold)}"
        score_surface = self.font.render(score_txt, True, (220, 220, 220))
        step_surface = self.font.render(step_txt, True, (220, 220, 220))
        
        self.window.blit(score_surface, (10, self.G * self.cell + 5))
        self.window.blit(step_surface, (10, self.G * self.cell + 30))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """Close the environment and clean up pygame resources."""
        if self.window is not None and pygame is not None:
            pygame.quit()
            self.window = None
            sys.exit(0)  # Exit the program when closing