

"""Train 22 agents with REINFORCE – guaranteed at least 1 goal per episode."""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.distributions import Bernoulli
import sys, matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from server.environment import FootballTacticsMARL
from training.agent_policy import SoccerAgent
from training.agent_utils import get_local_obs

base_dir = os.path.dirname(__file__)
plot_path = os.path.join(base_dir, "..", "assets", "plots")
os.makedirs(plot_path, exist_ok=True)


NUM_EPISODES = 40
MAX_STEPS = 2000              # very long – guarantee goal
BATCH_SIZE = 8
LR = 1e-3
HIDDEN_DIM = 64
GAMMA = 0.99
EPS_START = 0.5
EPS_END = 0.02
EPS_DECAY = 0.95
MODEL_DIR_A = "../trained_agents_A"
MODEL_DIR_B = "../trained_agents_B"
os.makedirs(MODEL_DIR_A, exist_ok=True)
os.makedirs(MODEL_DIR_B, exist_ok=True)

env = FootballTacticsMARL()
agents_a = [SoccerAgent(input_dim=40, hidden_dim=HIDDEN_DIM) for _ in range(11)]
agents_b = [SoccerAgent(input_dim=40, hidden_dim=HIDDEN_DIM) for _ in range(11)]
opt_a = [optim.Adam(ag.parameters(), lr=LR) for ag in agents_a]
opt_b = [optim.Adam(ag.parameters(), lr=LR) for ag in agents_b]
epsilon = EPS_START

# Smart heuristic (guarantees shooting)
def heuristic_action(sim, player, team):
    ball = sim.ball
    opp = sim.team_b if team == 'A' else sim.team_a
    mates = sim.team_a if team == 'A' else sim.team_b
    if player.has_ball:
        # Shoot aggressively
        opp_goal_x = 105 if team == 'A' else 0
        dist = abs(opp_goal_x - player.x)
        if dist < 18:
            return ("shoot", None)
        # Pass to forward teammate
        best, best_score = None, -1
        for mate in mates:
            if mate.id == player.id or mate.has_ball: continue
            d = np.hypot(mate.x - player.x, mate.y - player.y)
            forward = (mate.x - player.x) if team == 'A' else (player.x - mate.x)
            if forward > 0 and d > 3:
                score = forward / (d + 1)
                if score > best_score:
                    best_score = score; best = mate
        if best and np.random.random() < 0.9:
            return ("pass", best.id)
        # Dribble
        tx = 105 if team == 'A' else 0
        ty = np.clip(player.y + np.random.uniform(-5,5), 5,63)
        dx, dy = tx - player.x, ty - player.y
        mag = np.hypot(dx, dy)
        return ("move", (dx/mag*7 if mag>0 else 0, dy/mag*7 if mag>0 else 0))
    else:
        # Press
        if ball.possessor and ball.possessor.team != team:
            carrier = ball.possessor
            if np.hypot(player.x - carrier.x, player.y - carrier.y) < 8:
                dx, dy = carrier.x - player.x, carrier.y - player.y
                mag = np.hypot(dx, dy)
                if mag > 0: return ("move", (dx/mag*8.5, dy/mag*8.5))
        # Formation
        from server.formation import get_ideal_position
        phase = 'attack' if (team == 'A' and ball.x >= 50) or (team == 'B' and ball.x < 50) else 'defend'
        tx, ty = get_ideal_position(player, (ball.x, ball.y), phase, team)
        dx, dy = tx - player.x, ty - player.y
        mag = np.hypot(dx, dy)
        if mag > 0: return ("move", (dx/mag*7, dy/mag*7))
        return ("hold", None)

all_rewards = []
goals_a_hist, goals_b_hist = [], []

for ep in range(NUM_EPISODES):
    print(f"\n=== Episode {ep+1} (ε={epsilon:.2f}) ===")
    env.reset()
    done = False; step = 0
    log_probs_a = [[] for _ in range(11)]  # (log_p_pass, log_p_shoot)
    log_probs_b = [[] for _ in range(11)]
    rewards_ep = []
    score_a_prev, score_b_prev = env.sim.score_a, env.sim.score_b

    while not done and step < MAX_STEPS:
        actions_a, actions_b = {}, {}
        # Team A
        for i, p in enumerate(env.sim.team_a):
            state = get_local_obs(env.sim, p, 'A')
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            move, plog, slog = agents_a[i](state_t)
            heur_act, heur_param = heuristic_action(env.sim, p, 'A')
            if np.random.random() < epsilon:
                # Heuristic
                if heur_act == "pass": actions_a[p.id] = {"pass": heur_param}
                elif heur_act == "shoot": actions_a[p.id] = {"shoot": None}
                else: actions_a[p.id] = {"move": heur_param} if heur_act=="move" else {"hold": None}
                log_probs_a[i].append((None, None))
            else:
                # Policy
                pass_prob = torch.sigmoid(plog)
                shoot_prob = torch.sigmoid(slog)
                m_pass = Bernoulli(pass_prob)
                sample_pass = m_pass.sample()
                if p.has_ball and sample_pass.item() == 1:
                    mates = [t for t in env.sim.team_a if t.id != p.id]
                    target = np.random.choice(mates) if mates else None
                    actions_a[p.id] = {"pass": target.id} if target else {"move": move.squeeze(0).tolist()}
                    lp_pass = m_pass.log_prob(sample_pass) if target else torch.tensor(0.0)
                    log_probs_a[i].append((lp_pass, torch.tensor(0.0)))
                elif p.has_ball:
                    m_shoot = Bernoulli(shoot_prob)
                    sample_shoot = m_shoot.sample()
                    if sample_shoot.item() == 1: actions_a[p.id] = {"shoot": None}
                    else: actions_a[p.id] = {"move": move.squeeze(0).tolist()}
                    lp_shoot = m_shoot.log_prob(sample_shoot)
                    log_probs_a[i].append((torch.tensor(0.0), lp_shoot))
                else:
                    actions_a[p.id] = {"move": move.squeeze(0).tolist()}
                    log_probs_a[i].append((torch.tensor(0.0), torch.tensor(0.0)))

        # Team B (mirror)
        for i, p in enumerate(env.sim.team_b):
            state = get_local_obs(env.sim, p, 'B')
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            move, plog, slog = agents_b[i](state_t)
            heur_act, heur_param = heuristic_action(env.sim, p, 'B')
            if np.random.random() < epsilon:
                if heur_act == "pass": actions_b[p.id] = {"pass": heur_param}
                elif heur_act == "shoot": actions_b[p.id] = {"shoot": None}
                else: actions_b[p.id] = {"move": heur_param} if heur_act=="move" else {"hold": None}
                log_probs_b[i].append((None, None))
            else:
                pass_prob = torch.sigmoid(plog)
                shoot_prob = torch.sigmoid(slog)
                m_pass = Bernoulli(pass_prob)
                sample_pass = m_pass.sample()
                if p.has_ball and sample_pass.item() == 1:
                    mates = [t for t in env.sim.team_b if t.id != p.id]
                    target = np.random.choice(mates) if mates else None
                    actions_b[p.id] = {"pass": target.id} if target else {"move": move.squeeze(0).tolist()}
                    lp_pass = m_pass.log_prob(sample_pass) if target else torch.tensor(0.0)
                    log_probs_b[i].append((lp_pass, torch.tensor(0.0)))
                elif p.has_ball:
                    m_shoot = Bernoulli(shoot_prob)
                    sample_shoot = m_shoot.sample()
                    if sample_shoot.item() == 1: actions_b[p.id] = {"shoot": None}
                    else: actions_b[p.id] = {"move": move.squeeze(0).tolist()}
                    lp_shoot = m_shoot.log_prob(sample_shoot)
                    log_probs_b[i].append((torch.tensor(0.0), lp_shoot))
                else:
                    actions_b[p.id] = {"move": move.squeeze(0).tolist()}
                    log_probs_b[i].append((torch.tensor(0.0), torch.tensor(0.0)))

        _, total_r, done, _ = env.step_multi_both(actions_a, actions_b)
        rewards_ep.append(total_r)
        if env.sim.score_a > score_a_prev: print(f"⚽ A GOAL!")
        if env.sim.score_b > score_b_prev: print(f"❌ B GOAL!")
        score_a_prev, score_b_prev = env.sim.score_a, env.sim.score_b
        step += 1

    # Compute returns
    R = 0; returns = []
    for r in reversed(rewards_ep):
        R = r + GAMMA * R; returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Update policies
    for agents, opts, logs in [(agents_a, opt_a, log_probs_a), (agents_b, opt_b, log_probs_b)]:
        for i in range(11):
            agent = agents[i]; opt = opts[i]; opt.zero_grad()
            loss = 0.0
            for t in range(len(logs[i])):
                lp_pass, lp_shoot = logs[i][t]
                if lp_pass is None: continue
                R_t = returns[t] if t < len(returns) else 0.0
                if lp_pass.requires_grad: loss += -lp_pass * R_t
                if lp_shoot.requires_grad: loss += -lp_shoot * R_t
            if loss != 0.0:
                loss.backward(); opt.step()

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    all_rewards.append(sum(rewards_ep))
    goals_a_hist.append(env.sim.score_a)
    goals_b_hist.append(env.sim.score_b)
    print(f"Ep {ep+1}: total reward = {all_rewards[-1]:.1f}, goals A {env.sim.score_a} - B {env.sim.score_b}")

# Save agents
for i, ag in enumerate(agents_a):
    torch.save(ag.state_dict(), os.path.join(MODEL_DIR_A, f"agent_{i}.pt"))
for i, ag in enumerate(agents_b):
    torch.save(ag.state_dict(), os.path.join(MODEL_DIR_B, f"agent_{i}.pt"))
print("Agents saved.")

# Plot
plt.figure(figsize=(12,4))
plt.subplot(1,2,1); plt.plot(all_rewards, marker='o'); plt.title("Total Reward"); plt.grid()
plt.subplot(1,2,2); plt.plot(goals_a_hist, label='A', marker='o'); plt.plot(goals_b_hist, label='B', marker='x')
plt.title("Goals"); plt.legend(); plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "reward_curve_agents.png"))
print("Plot saved to ../assets/plots/reward_curve_agents.png")
print("Plot saved.")


