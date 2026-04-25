"""Graphical football match – trained AI coach + robust fallback."""
import json, os, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from server.environment import FootballTacticsMARL
from models import Action

FIELD_LENGTH, FIELD_WIDTH = 105.0, 68.0
CENTRE_CIRCLE_RADIUS = 9.15

# ---- Must match training configuration ----
VOCAB_SIZE = 500
EMBED_DIM = 128          # <-- changed from 64 to 128
HIDDEN_DIM = 256         # <-- changed from 128 to 256
MAX_OBS_LEN = 400        # <-- increased to match training
MAX_ACT_LEN = 200        # <-- increased to match training
# -------------------------------------------

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
model = None
if os.path.exists("../trained_model/gru_policy.pt"):
    model = GRUPolicy(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)  # <-- uses new dimensions
    model.load_state_dict(torch.load("../trained_model/gru_policy.pt", map_location='cpu'))
    model.eval()
    print("✅ Loaded trained coach model.")
else:
    print("⚠️ No trained model – using smart heuristic coach.")

def build_prompt(obs_text):
    return (
        "You are controlling a football team. Given the current game state, "
        "decide actions for all 11 players. Output ONLY a valid JSON object "
        "mapping player ID to action as specified.\n"
        f"Current state:\n{obs_text}\nActions:\n"
    )

def model_generate_action(obs_text):
    prompt = build_prompt(obs_text)
    input_ids = torch.tensor([tokenizer.encode(prompt, max_len=MAX_OBS_LEN)], dtype=torch.long)
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
        return tokenizer.decode(generated)

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

def heuristic_action(sim, player):
    if player.has_ball:
        if player.team == 'A' and player.x > 85 and np.random.random() < 0.6:
            return {"shoot": None}
        elif player.team == 'B' and player.x < 20 and np.random.random() < 0.6:
            return {"shoot": None}
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
        target_x = 105 if player.team == 'A' else 0
        target_y = np.clip(player.y + np.random.uniform(-8, 8), 5, 63)
        dx, dy = target_x - player.x, target_y - player.y
        mag = np.hypot(dx, dy)
        if mag > 0:
            speed = min(mag, 8)
            return {"move": [dx/mag*speed, dy/mag*speed]}
        return {"hold": None}
    else:
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

def draw_pitch(ax):
    ax.clear()
    ax.set_facecolor('#2e8b57')
    ax.set_xlim(0, FIELD_LENGTH)
    ax.set_ylim(0, FIELD_WIDTH)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.plot([0, FIELD_LENGTH, FIELD_LENGTH, 0, 0],
            [0, 0, FIELD_WIDTH, FIELD_WIDTH, 0], 'white', lw=2)
    ax.plot([FIELD_LENGTH/2]*2, [0, FIELD_WIDTH], 'white', lw=2)
    ax.add_patch(Circle((52.5, 34), 9.15, ec='white', fc='none', lw=2))
    for x in [0, FIELD_LENGTH]:
        ax.plot([x, x], [30.34, 37.66], 'white', lw=5)
        ax.plot([x, x+(-1 if x==0 else 1)*16.5], [13.84, 13.84], 'white', lw=2)
        ax.plot([x, x+(-1 if x==0 else 1)*16.5], [54.16, 54.16], 'white', lw=2)
        ax.plot([x+(-1 if x==0 else 1)*16.5]*2, [13.84, 54.16], 'white', lw=2)
        ax.plot([x, x+(-1 if x==0 else 1)*5.5], [24.84, 24.84], 'white', lw=2)
        ax.plot([x, x+(-1 if x==0 else 1)*5.5], [43.16, 43.16], 'white', lw=2)
        ax.plot([x+(-1 if x==0 else 1)*5.5]*2, [24.84, 43.16], 'white', lw=2)
    ax.set_title("Football Tactical MARL – AI Coach", color='white', fontsize=14)

def update(frame, env, fig, ax):
    sim = env.sim
    if model is not None:
        obs_text = env._build_obs()
        action_str = model_generate_action(obs_text)
        actions_a = parse_action_dict(action_str)
        if actions_a is None:
            actions_a = {p.id: heuristic_action(sim, p) for p in sim.team_a}
        else:
            for p in sim.team_a:
                if p.id not in actions_a:
                    actions_a[p.id] = heuristic_action(sim, p)
    else:
        actions_a = {p.id: heuristic_action(sim, p) for p in sim.team_a}

    obs, _, done, _ = env.step(Action(text=json.dumps(actions_a)))

    draw_pitch(ax)
    ax.scatter([p.x for p in sim.team_a], [p.y for p in sim.team_a], c='blue', s=100)
    ax.scatter([p.x for p in sim.team_b], [p.y for p in sim.team_b], c='red', s=100)
    if sim.ball.possessor:
        c = sim.ball.possessor
        ax.scatter(c.x, c.y, facecolors='none', edgecolors='yellow', s=150, linewidth=2)
    ax.scatter(sim.ball.x, sim.ball.y, c='white', s=70, edgecolors='black')
    ax.text(10, FIELD_WIDTH+2, f"Score: {sim.score_a} - {sim.score_b}", color='white', fontsize=12)
    ax.text(FIELD_LENGTH-30, FIELD_WIDTH+2, f"Step: {sim.step_count}", color='white', fontsize=10)
    if done:
        print(f"Match finished: {sim.score_a} - {sim.score_b}")
    return ax

def main():
    env = FootballTacticsMARL()
    env.reset()
    fig, ax = plt.subplots(figsize=(12,8))
    fig.patch.set_facecolor('#2e8b57')
    ani = animation.FuncAnimation(fig, update, fargs=(env, fig, ax),
                                  frames=300, interval=80, repeat=False)
    plt.show()

if __name__ == '__main__':
    main()