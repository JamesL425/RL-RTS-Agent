# evaluate.py - Symmetric Gold Rush evaluation
import argparse, yaml, torch, numpy as np
from policy import ActorCritic
from rts_env import GoldRushEnv
from torch.distributions import Categorical
import sys


def load_policy(path, obs_dim, act_dim):
    """Load policy from checkpoint file."""
    net = ActorCritic(obs_dim, 128, act_dim)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


def symmetric_evaluate(policy_a, policy_b, cfg, episodes=100, greedy=True, render=False):
    """
    Evaluate policy_a against policy_b with fixed roles.
    
    Args:
        policy_a: Blue policy
        policy_b: Red policy
        cfg: (grid_size, num_agents, num_gold, max_steps) tuple
        episodes: Number of episodes to evaluate
        greedy: Whether to use greedy action selection
        render: Whether to render the game
        
    Returns:
        Dictionary with evaluation results
    """
    G, N, K, max_steps = cfg
    
    try:
        import pygame
    except ImportError:
        pygame = None
        if render:
            print("Warning: pygame not installed, cannot render")
            render = False
    
    # Get actions from policy
    def get_actions(policy, obs, greedy=True):
        """Get actions from policy for a given observation."""
        # Convert observation to the right format
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        # Handle different observation sizes
        if obs.size == (2*N + 1) * G * G:
            # Full observation with all channels
            obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)
        elif obs.size == G * G:
            # Single channel observation (5x5 grid) - likely from opponent policy
            # Create a full observation with zeros
            full_obs = np.zeros((2*N + 1, G, G), dtype=np.float32)
            
            # The 5x5 grid could be either agent positions or gold
            # Assume it's the agent's positions (more likely scenario)
            if obs.ndim == 2:
                grid = obs
            else:
                grid = obs.reshape(G, G)
                
            # Put it in the first N channels (agent's own position)
            for y in range(G):
                for x in range(G):
                    if grid[y, x] > 0:
                        # Find which agent channel to put this in
                        for i in range(N):
                            # Check if we've already placed this agent
                            if np.sum(full_obs[i]) == 0:
                                full_obs[i, y, x] = 1.0
                                break
            
            # Flatten and convert to tensor
            obs_tensor = torch.tensor(full_obs.flatten(), dtype=torch.float32).unsqueeze(0)
        else:
            # Unknown format - create empty observation
            obs_tensor = torch.zeros((1, (2*N + 1) * G * G), dtype=torch.float32)
        
        # Get actions from policy
        with torch.no_grad():
            logits, _ = policy(obs_tensor)
            logits = logits.view(N, 5)  # Reshape to (N, 5) for N agents with 5 actions each
            
            if greedy:
                actions = logits.argmax(dim=-1).cpu().numpy()
            else:
                dist = Categorical(logits=logits)
                actions = dist.sample().cpu().numpy()
            
            return actions
    
    # Create a policy callable for the opponent
    class OpponentPolicy:
        def __init__(self, policy):
            self.policy = policy
        
        def __call__(self, obs):
            """Callable interface for opponent policy."""
            try:
                # Get actions using our helper function
                actions = get_actions(self.policy, obs, greedy)
                
                # Dummy log probs
                log_probs = np.zeros_like(actions, dtype=np.float32)
                return actions, log_probs, None
            except Exception as e:
                print(f"Error in opponent policy: {e}")
                # Fallback to random actions
                actions = np.random.randint(0, 5, size=N)
                log_probs = np.zeros_like(actions, dtype=np.float32)
                return actions, log_probs, None
    
    # Track results
    blue_wins = 0
    red_wins = 0
    draws = 0
    
    # Create red opponent once
    policy_b_opponent = OpponentPolicy(policy_b)
    
    # Create and reuse a single environment for all episodes
    env = GoldRushEnv(G, N, K, max_steps, 
                    render_mode="pygame" if render else None,
                    opponent_policy=policy_b_opponent)
    
    # Run episodes with policy_a as blue and policy_b as red
    for ep in range(episodes):
        print(f"Starting episode {ep+1}/{episodes}")
        
        # Reset the environment for this episode
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done:
            # Get blue actions from policy_a
            blue_actions = get_actions(policy_a, obs, greedy)
            
            obs, reward, done, _, info = env.step(blue_actions)
            step += 1
            
            # Render explicitly
            if render and env.window is not None:
                env.render()
        
        # Record result
        if reward > 0:  # Blue (policy_a) wins
            blue_wins += 1
            print(f"Episode {ep+1} completed in {step} steps: Blue wins")
        elif reward < 0:  # Red (policy_b) wins
            red_wins += 1
            print(f"Episode {ep+1} completed in {step} steps: Red wins")
        else:  # Draw
            draws += 1
            print(f"Episode {ep+1} completed in {step} steps: Draw")
        
        # Small pause between episodes to see the final state
        if render and pygame is not None and env.window is not None:
            pygame.time.wait(200)  # 0.2 second pause
    
    # Calculate results - exclude draws from winrate calculation
    total_decisive_games = blue_wins + red_wins
    blue_win_rate = blue_wins / total_decisive_games if total_decisive_games > 0 else 0.5
    
    # Clean up at the very end
    if render and env.window is not None:
        env.close()  # This also exits the program due to sys.exit in close()
    
    return {
        "episodes": episodes,
        "blue_wins": blue_wins,
        "red_wins": red_wins,
        "draws": draws,
        "blue_win_rate": blue_win_rate
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gold Rush Evaluation")
    parser.add_argument("--blue", required=True, help="Path to blue policy checkpoint")
    parser.add_argument("--red", help="Path to red policy (default: same as blue)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--mode", choices=["greedy", "stochastic"], default="greedy",
                      help="Action selection mode")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--fps", type=int, default=6, help="Rendering frames per second")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    # Print configuration
    print(f"Blue policy: {args.blue}")
    print(f"Red policy: {args.red if args.red else 'Same as blue (self-play)'}")
    print(f"Mode: {args.mode}")
    print(f"Rendering: {args.render}")
    print(f"FPS: {args.fps}")
    print(f"Episodes: {args.episodes}")

    # Load configuration
    cfg_yaml = yaml.safe_load(open(args.config))
    G = int(cfg_yaml["grid_size"])
    N = int(cfg_yaml["num_agents"])
    K = int(cfg_yaml["num_gold"])
    S = int(cfg_yaml["max_steps"])
    
    print(f"Environment config: Grid size={G}, Agents={N}, Gold={K}, Max steps={S}")
    
    # Calculate dimensions
    obs_dim = (2 * N + 1) * G * G
    act_dim = 5 * N

    # Load policies
    blue_policy = load_policy(args.blue, obs_dim, act_dim)
    
    if args.red is None:
        # Self-play evaluation
        red_policy = blue_policy
        is_self_play = True
    else:
        # Load separate red policy
        red_policy = load_policy(args.red, obs_dim, act_dim)
        is_self_play = False

    # Update FPS in environment metadata
    GoldRushEnv.metadata["render_fps"] = args.fps
    
    # Run evaluation
    try:
        results = symmetric_evaluate(
            blue_policy, 
            red_policy,
            (G, N, K, S),
            episodes=args.episodes,
            greedy=(args.mode == "greedy"),
            render=args.render
        )
        
        # Print results
        if is_self_play:
            print(f"Self-play evaluation ({args.mode}):")
            print(f"Blue win rate: {results['blue_win_rate']:.2%} over {results['episodes']} episodes "
                  f"({results['draws']} draws excluded)")
            
            print(f"Blue wins: {results['blue_wins']}, Red wins: {results['red_wins']}, "
                  f"Draws: {results['draws']}")
        else:
            print(f"{args.blue} (Blue) vs {args.red} (Red) evaluation ({args.mode}):")
            print(f"Blue win rate: {results['blue_win_rate']:.2%} over {results['episodes']} episodes "
                  f"({results['draws']} draws excluded)")
            print(f"Blue wins: {results['blue_wins']}, Red wins: {results['red_wins']}, "
                  f"Draws: {results['draws']}")
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(0)