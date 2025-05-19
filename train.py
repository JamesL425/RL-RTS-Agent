# train.py  – Gold-Rush PPO with full detaches
import os, time, copy, argparse, yaml, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from gymnasium.vector import AsyncVectorEnv

from rts_env import GoldRushEnv
from policy  import ActorCritic


def make_env(G, N, K, max_steps):
    def _init():
        return GoldRushEnv(
            grid_size=G,
            num_agents=N,
            num_gold=K,
            max_steps=max_steps,
            render_mode=None,
            opponent_policy=None,
        )
    return _init


def detach_cpu(x):
    return x.detach().cpu()


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ── parameters ─────────────────────────
    G, N, K, MAX_STEPS = map(int, (cfg["grid_size"],
                                   cfg["num_agents"],
                                   cfg["num_gold"],
                                   cfg["max_steps"]))
    NUM_ENVS   = int(cfg["num_envs"])
    TOTAL_UPD  = int(cfg["total_updates"])
    STEPS_PER  = int(cfg["steps_per_update"])
    SAVE_INT   = int(cfg["save_interval"])
    POOL_SIZE  = int(cfg["pool_size"])
    LOG_INT    = int(cfg["log_interval"])

    GAMMA, LAMBDA = float(cfg["gamma"]), float(cfg["lambda"])
    CLIP, ENT_C, VF_C = float(cfg["clip_eps"]), float(cfg["entropy_coef"]), float(cfg["value_coef"])
    LR, EPOCHS, MB = float(cfg["learning_rate"]), int(cfg["ppo_epochs"]), int(cfg["mini_batch_size"])
    MAX_GN = 0.5

    # ── envs ───────────────────────────────
    envs = AsyncVectorEnv([make_env(G, N, K, MAX_STEPS) for _ in range(NUM_ENVS)])

    obs_dim = (2 * N + 1) * G * G
    act_dim = 5 * N

    blue = ActorCritic(obs_dim, 128, act_dim).to(device)
    opt  = Adam(blue.parameters(), lr=LR)
    pool = [copy.deepcopy(blue.state_dict())]
    os.makedirs("red_pool", exist_ok=True)
    last_snap = time.time()

    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device).view(NUM_ENVS, -1)

    for upd in range(1, TOTAL_UPD + 1):
        # --- assign opponents ---
        reds = []
        for _ in range(NUM_ENVS):
            net = ActorCritic(obs_dim, 128, act_dim)
            net.load_state_dict(random.choice(pool))
            net.eval()
            reds.append(net)
        envs.set_attr("opponent_policy", reds)

        o_buf, a_buf, lp_buf, v_buf, r_buf, d_buf = [], [], [], [], [], []

        for _ in range(STEPS_PER):
            logits, value = blue(obs)                      # (N_env, act_dim)
            dist = Categorical(logits=logits.view(-1, 5))
            act_flat = dist.sample()                       # no grad
            logp_flat = dist.log_prob(act_flat)
            act_t = act_flat.view(NUM_ENVS, N)
            logp_env = logp_flat.view(NUM_ENVS, N).sum(-1)

            act_np = act_t.detach().cpu().numpy()
            nxt, rew_np, term_np, trunc_np, _ = envs.step(act_np)
            done_np = np.logical_or(term_np, trunc_np)

            # store DETACHED cpu tensors
            o_buf.append(obs.cpu())
            a_buf.append(act_t.cpu())
            lp_buf.append(detach_cpu(logp_env))
            v_buf.append(detach_cpu(value))
            r_buf.append(torch.tensor(rew_np, dtype=torch.float32))
            d_buf.append(torch.tensor(done_np, dtype=torch.float32))

            obs = torch.tensor(nxt, dtype=torch.float32, device=device).view(NUM_ENVS, -1)

        with torch.no_grad():
            _, last_val = blue(obs)
        v_buf.append(detach_cpu(last_val))

        # stack
        o_t, a_t  = map(torch.stack, (o_buf, a_buf))
        lp_t, v_t = map(torch.stack, (lp_buf, v_buf))
        r_t, d_t  = map(torch.stack, (r_buf, d_buf))

        T = r_t.shape[0]
        # --- GAE ---
        adv = torch.zeros_like(r_t)
        last = torch.zeros(NUM_ENVS)
        for t in reversed(range(T)):
            mask = 1.0 - d_t[t]
            delta = r_t[t] + GAMMA * v_t[t + 1] * mask - v_t[t]
            last = delta + GAMMA * LAMBDA * mask * last
            adv[t] = last
        ret = adv + v_t[:T]

        # flatten
        flat_obs = o_t.view(T * NUM_ENVS, -1).to(device)
        flat_act = a_t.view(T * NUM_ENVS, N).to(device)
        flat_old = lp_t.view(T * NUM_ENVS).to(device)
        flat_adv = adv.view(T * NUM_ENVS).to(device)
        flat_ret = ret.view(T * NUM_ENVS).to(device)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        flat_act_one = flat_act.view(-1)                  # for log_prob look-up

        idx_all = np.arange(T * NUM_ENVS)
        for _ in range(EPOCHS):
            np.random.shuffle(idx_all)
            for start in range(0, len(idx_all), MB):
                mb = idx_all[start:start + MB]
                logits_mb, val_mb = blue(flat_obs[mb])
                dist_mb = Categorical(logits=logits_mb.view(-1, 5))

                act_mb_flat = flat_act[mb].reshape(-1)
                new_logp_agents = dist_mb.log_prob(act_mb_flat)
                new_logp_env = new_logp_agents.view(-1, N).sum(-1)

                ratio = torch.exp(new_logp_env - flat_old[mb])
                surr1 = ratio * flat_adv[mb]
                surr2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * flat_adv[mb]
                pol_loss = -torch.min(surr1, surr2).mean()
                val_loss = F.mse_loss(val_mb, flat_ret[mb])
                ent      = dist_mb.entropy().mean()
                loss = pol_loss + VF_C * val_loss - ENT_C * ent

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(blue.parameters(), MAX_GN)
                opt.step()

        # --- snapshot & timing ---
        if upd % SAVE_INT == 0:
            pool.append(copy.deepcopy(blue.state_dict()))
            if len(pool) > POOL_SIZE:
                pool.pop(0)
            snap = f"red_pool/red_{upd}.pth"
            torch.save(pool[-1], snap)
            now = time.time()
            print(f"[upd {upd}] saved {snap}  Δ{now - last_snap:.1f}s  pool={len(pool)}")
            last_snap = now

        if upd % LOG_INT == 0:
            print(f"Upd {upd}/{TOTAL_UPD} complete")

    torch.save(blue.state_dict(), "blue_final.pth")
    print("✓ training finished")


if __name__ == "__main__":
    ar = argparse.ArgumentParser()
    ar.add_argument("--config", default="config.yaml")
    main(ar.parse_args().config)
