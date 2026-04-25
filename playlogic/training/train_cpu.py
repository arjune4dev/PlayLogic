
"""Self‑improving football coach – learns attacking & defensive sequences."""
import os, json, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset as TorchDataset
import sys, matplotlib.pyplot as plt



base_dir = os.path.dirname(__file__)
plot_path = os.path.join(base_dir, "..", "assets", "plots")
os.makedirs(plot_path, exist_ok=True)



sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from server.environment import FootballTacticsMARL
from models import Action

# ---------- Configuration ----------
EMBED_DIM = 128
HIDDEN_DIM = 256
VOCAB_SIZE = 500
MAX_OBS_LEN = 400
MAX_ACT_LEN = 200
PRETRAIN_EPISODES = 3         # episodes where model imitates heuristic
RL_EPISODES = 10              # episodes of self‑improvement
MAX_EPISODE_STEPS = 250
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
HEURISTIC_MIX_PROB = 0.3      # 30% chance to use heuristic even after pretrain
GOAL_WINDOW = 15              # steps right before a goal
# -----------------------------------

# ---------- Tokenizer ----------
class SimpleTokenizer:
    def __init__(self, vocab_size=VOCAB_SIZE):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.special = [self.pad_token, self.eos_token, self.unk_token]
        self.c2i = {}
        self.i2c = {}
        self.vocab_size = vocab_size
        self._init_vocab()

    def _init_vocab(self):
        chars = [chr(i) for i in range(32, 127)]
        for i, tok in enumerate(self.special):
            self.c2i[tok] = i
            self.i2c[i] = tok
        idx = len(self.special)
        for c in chars:
            if idx >= self.vocab_size:
                break
            if c not in self.c2i:
                self.c2i[c] = idx
                self.i2c[idx] = c
                idx += 1
        for c in ['\n', '\t', '\r']:
            if c not in self.c2i and idx < self.vocab_size:
                self.c2i[c] = idx
                self.i2c[idx] = c
                idx += 1

    def encode(self, text, max_len=None):
        ids = [self.c2i.get(ch, self.c2i[self.unk_token]) for ch in text]
        if max_len is not None:
            if len(ids) < max_len:
                ids += [self.c2i[self.pad_token]] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
        return ids

    def decode(self, ids):
        return ''.join(
            self.i2c.get(i, self.unk_token)
            for i in ids
            if i not in (self.c2i[self.pad_token], self.c2i[self.eos_token])
        )

# ---------- GRU Policy ----------
class GRUPolicy(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        emb = self.embedding(input_ids)
        out, hidden = self.gru(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

tokenizer = SimpleTokenizer()
model = GRUPolicy(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def build_prompt(obs_text):
    return (
        "You are controlling a football team. Given the current game state, "
        "decide actions for all 11 players. Output ONLY a valid JSON object "
        "mapping player ID to action as specified.\n"
        f"Current state:\n{obs_text}\nActions:\n"
    )

def sample_action_text(prompt):
    """Generate action JSON string from the policy (greedy)."""
    input_ids = torch.tensor([tokenizer.encode(prompt, max_len=MAX_OBS_LEN)], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        _, hidden = model(input_ids)
        start_char = '\n' if '\n' in tokenizer.c2i else ' '
        current = torch.tensor([[tokenizer.c2i[start_char]]], dtype=torch.long)
        generated = []
        for _ in range(MAX_ACT_LEN):
            logits, hidden = model(current, hidden)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            if next_token == tokenizer.c2i[tokenizer.eos_token]:
                break
            generated.append(next_token)
            current = torch.tensor([[next_token]], dtype=torch.long)
        action_str = tokenizer.decode(generated)
    model.train()
    return action_str

def parse_action_dict(action_str):
    try:
        start = action_str.find('{')
        end = action_str.rfind('}')
        if start != -1 and end != -1:
            raw = json.loads(action_str[start:end+1])
            return {int(k): v for k, v in raw.items()}
    except:
        pass
    return None

# ---------- Strong heuristic coach ----------
def heuristic_action(sim, player):
    if player.has_ball:
        # Shoot if close enough
        if player.team == 'A' and player.x > 85 and np.random.random() < 0.6:
            return {"shoot": None}
        elif player.team == 'B' and player.x < 20 and np.random.random() < 0.6:
            return {"shoot": None}

        # Pass to a forward teammate
        teammates = sim.team_a if player.team == 'A' else sim.team_b
        best_target = None
        best_score = -1
        for mate in teammates:
            if mate.id == player.id or mate.has_ball:
                continue
            dist = np.hypot(mate.x - player.x, mate.y - player.y)
            if dist < 2:
                continue
            forward = (mate.x - player.x) if player.team == 'A' else (player.x - mate.x)
            score = forward / (dist + 1)
            if score > best_score:
                best_score = score
                best_target = mate
        if best_target and np.random.random() < 0.8:
            return {"pass": best_target.id}

        # Dribble towards goal
        target_x = 105 if player.team == 'A' else 0
        target_y = np.clip(player.y + np.random.uniform(-8, 8), 5, 63)
        dx, dy = target_x - player.x, target_y - player.y
        mag = np.hypot(dx, dy)
        if mag > 0:
            speed = min(mag, 8)
            return {"move": [dx/mag*speed, dy/mag*speed]}
        return {"hold": None}
    else:
        # Move to ideal formation position
        from server.formation import get_ideal_position
        ball_pos = (sim.ball.x, sim.ball.y)
        phase = 'attack' if sim.ball.x >= 50 else 'defend'
        tx, ty = get_ideal_position(player, ball_pos, phase, player.team)
        tx += np.random.uniform(-2, 2)
        ty += np.random.uniform(-2, 2)
        dx, dy = tx - player.x, ty - player.y
        mag = np.hypot(dx, dy)
        if mag > 0:
            speed = min(mag, 8)
            return {"move": [dx/mag*speed, dy/mag*speed]}
        return {"hold": None}

# ---------- 1. Pretraining (imitate heuristic) ----------
print("=== Pretraining on heuristic coach ===")
env = FootballTacticsMARL()  # <-- env created here
pretrain_data = []
for ep in range(PRETRAIN_EPISODES):
    obs = env.reset()
    done = False
    step = 0
    while not done and step < MAX_EPISODE_STEPS:
        prompt = build_prompt(obs.text)
        actions_a = {p.id: heuristic_action(env.sim, p) for p in env.sim.team_a}
        action_str = json.dumps(actions_a)
        obs, _, done, _ = env.step(Action(text=action_str))
        pretrain_data.append((prompt, action_str))
        step += 1

if len(pretrain_data) > 0:
    class PretrainDataset(TorchDataset):
        def __init__(self, data):
            self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            p, c = self.data[idx]
            ids = tokenizer.encode(p+c, max_len=MAX_OBS_LEN+MAX_ACT_LEN)
            return torch.tensor(ids, dtype=torch.long)

    loader = DataLoader(PretrainDataset(pretrain_data), batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.c2i[tokenizer.pad_token])
    for epoch in range(3):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            input_seq = batch[:, :-1]
            target_seq = batch[:, 1:]
            logits, _ = model(input_seq)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_seq.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Pretrain epoch {epoch+1} loss: {total_loss/len(loader):.4f}")
print("Pretraining done.\n")

# ---------- 2. Self‑improving RL with goal memory ----------
env = FootballTacticsMARL()
goal_memory = []
concede_memory = []
all_episode_rewards = []

for episode in range(RL_EPISODES):
    print(f"\n=== Episode {episode+1}/{RL_EPISODES} ===")
    obs = env.reset()
    done = False
    trajectory = []
    score_a_prev = env.sim.score_a
    score_b_prev = env.sim.score_b

    step = 0
    while not done and step < MAX_EPISODE_STEPS:
        prompt = build_prompt(obs.text)

        # Mix policy with heuristic
        if model is not None and np.random.random() > HEURISTIC_MIX_PROB:
            action_str = sample_action_text(prompt)
            actions_a = parse_action_dict(action_str)
            if actions_a is None:
                actions_a = {p.id: heuristic_action(env.sim, p) for p in env.sim.team_a}
        else:
            actions_a = {p.id: heuristic_action(env.sim, p) for p in env.sim.team_a}

        # Fill missing players
        for p in env.sim.team_a:
            if p.id not in actions_a:
                actions_a[p.id] = heuristic_action(env.sim, p)

        obs, reward, done, info = env.step(Action(text=json.dumps(actions_a)))
        trajectory.append((prompt, json.dumps(actions_a), reward))

        # Detect goal scored by A
        if env.sim.score_a > score_a_prev:
            print(f"⚽ Goal for A at step {step}! Saving attacking sequence.")
            for i in range(max(0, len(trajectory)-GOAL_WINDOW), len(trajectory)):
                goal_memory.append((trajectory[i][0], trajectory[i][1]))

        # Detect goal conceded
        if env.sim.score_b > score_b_prev:
            print(f"❌ Goal for B at step {step}. Saving defensive failure.")
            for i in range(max(0, len(trajectory)-GOAL_WINDOW), len(trajectory)):
                concede_memory.append((trajectory[i][0], trajectory[i][1]))

        score_a_prev = env.sim.score_a
        score_b_prev = env.sim.score_b
        step += 1

    total_reward = sum(r for _, _, r in trajectory)
    all_episode_rewards.append(total_reward)
    print(f"Episode reward: {total_reward:.2f} | Score: {env.sim.score_a}-{env.sim.score_b}")

    # ---- Train on combined memories ----
    if len(goal_memory) > 0 or len(concede_memory) > 0:
        training_data = goal_memory.copy()
        training_data.extend(concede_memory)
        class MemDataset(TorchDataset):
            def __init__(self, data): self.data = data
            def __len__(self): return len(self.data)
            def __getitem__(self, idx):
                p, c = self.data[idx]
                ids = tokenizer.encode(p+c, max_len=MAX_OBS_LEN+MAX_ACT_LEN)
                return torch.tensor(ids, dtype=torch.long)

        dataset = MemDataset(training_data)
        loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True)
        model.train()
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.c2i[tokenizer.pad_token])
        for epoch in range(2):
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                input_seq = batch[:, :-1]
                target_seq = batch[:, 1:]
                logits, _ = model(input_seq)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_seq.reshape(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  RL epoch {epoch+1} loss: {total_loss/len(loader):.4f}")

    goal_memory = goal_memory[-500:]
    concede_memory = concede_memory[-500:]

# Save final model
os.makedirs("../trained_model", exist_ok=True)
torch.save(model.state_dict(), "../trained_model/gru_policy.pt")
with open("../trained_model/tokenizer_config.json", "w") as f:
    json.dump({"vocab_size": VOCAB_SIZE, "max_obs_len": MAX_OBS_LEN}, f)
print("Model saved.")

# Plot
plt.figure(figsize=(8,4))
plt.plot(all_episode_rewards, marker='o')
plt.title("RL Episode Rewards (after pretraining)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_path, "reward_curve.png"))
print("Reward curve saved to ../assets/plots/reward_curve.png")
print("Reward curve saved.")