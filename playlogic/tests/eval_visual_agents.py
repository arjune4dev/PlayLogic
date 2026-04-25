"""Visualizer with scoreboard, facing arrows, goalie prediction."""
import os, numpy as np, torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Arc, FancyArrow
from server.environment import FootballTacticsMARL
from training.agent_policy import SoccerAgent
from training.agent_utils import get_local_obs
from server.physics import DT

FIELD_LENGTH, FIELD_WIDTH = 105.0, 68.0
MODEL_DIR_A = "../trained_agents_A"
MODEL_DIR_B = "../trained_agents_B"

agents_a, agents_b = [], []
for i in range(11):
    ag = SoccerAgent(input_dim=40, hidden_dim=64)
    path = os.path.join(MODEL_DIR_A, f"agent_{i}.pt")
    if os.path.exists(path): ag.load_state_dict(torch.load(path, map_location='cpu')); ag.eval()
    agents_a.append(ag if os.path.exists(path) else None)
for i in range(11):
    ag = SoccerAgent(input_dim=40, hidden_dim=64)
    path = os.path.join(MODEL_DIR_B, f"agent_{i}.pt")
    if os.path.exists(path): ag.load_state_dict(torch.load(path, map_location='cpu')); ag.eval()
    agents_b.append(ag if os.path.exists(path) else None)

env = FootballTacticsMARL()
env.reset()

def draw_pitch(ax):
    ax.clear()
    ax.set_facecolor('#0b6623')
    ax.set_xlim(-2, FIELD_LENGTH+2)
    ax.set_ylim(-2, FIELD_WIDTH+2)
    ax.set_aspect('equal'); ax.axis('off')
    # field lines
    ax.plot([0,FIELD_LENGTH,FIELD_LENGTH,0,0],[0,0,FIELD_WIDTH,FIELD_WIDTH,0],'white',lw=3)
    ax.plot([FIELD_LENGTH/2]*2,[0,FIELD_WIDTH],'white',lw=2)
    ax.add_patch(Circle((52.5,34),9.15,ec='white',fc='none',lw=2))
    for x in [0,FIELD_LENGTH]:
        ax.plot([x,x],[30.34,37.66],'white',lw=6)
        # penalty areas
        ax.plot([x, x+(-16.5 if x==0 else 16.5)],[13.84,13.84],'white',lw=2)
        ax.plot([x, x+(-16.5 if x==0 else 16.5)],[54.16,54.16],'white',lw=2)
        ax.plot([x+(-16.5 if x==0 else 16.5)]*2,[13.84,54.16],'white',lw=2)
        # goal areas
        ax.plot([x, x+(-5.5 if x==0 else 5.5)],[24.84,24.84],'white',lw=2)
        ax.plot([x, x+(-5.5 if x==0 else 5.5)],[43.16,43.16],'white',lw=2)
        ax.plot([x+(-5.5 if x==0 else 5.5)]*2,[24.84,43.16],'white',lw=2)
    # penalty arcs
    ax.add_patch(Arc((11,34),18.3,18.3,theta1=150,theta2=210,color='white',lw=2))
    ax.add_patch(Arc((94,34),18.3,18.3,theta1=330,theta2=390,color='white',lw=2))
    # corners
    for (x,y) in [(0,0),(0,68),(105,0),(105,68)]:
        ax.add_patch(Circle((x,y),1,ec='white',fc='none',lw=2))
    # scoreboard (top right)
    score_text = f"{env.sim.score_a} : {env.sim.score_b}"
    ax.text(FIELD_LENGTH-10, FIELD_WIDTH-2, score_text, fontsize=18, color='white', weight='bold',
            ha='right', va='top', bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.3'))
    ax.set_title("⚽ 22 AI Agents Playing ⚽", color='white', fontsize=16)

def update(frame):
    sim = env.sim
    actions_a, actions_b = {}, {}
    # Team A
    for i, p in enumerate(sim.team_a):
        if agents_a[i] is not None:
            state = get_local_obs(sim, p, 'A')
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                move, plog, slog = agents_a[i](state_t)
                pp = torch.sigmoid(plog).item()
                sp = torch.sigmoid(slog).item()
                if p.has_ball:
                    if np.random.random() < pp:
                        mates = [t for t in sim.team_a if t.id != p.id]
                        if mates: actions_a[p.id] = {"pass": np.random.choice(mates).id}; continue
                    if np.random.random() < sp:
                        actions_a[p.id] = {"shoot": None}; continue
                actions_a[p.id] = {"move": move.squeeze(0).tolist()}
        else:
            from server.formation import get_ideal_position
            ball_pos = (sim.ball.x, sim.ball.y)
            phase = 'attack' if sim.ball.x >= 50 else 'defend'
            tx, ty = get_ideal_position(p, ball_pos, phase, 'A')
            dx, dy = tx - p.x, ty - p.y
            mag = np.hypot(dx, dy)
            if mag > 0: actions_a[p.id] = {"move": [dx/mag*7, dy/mag*7]}
            else: actions_a[p.id] = {"hold": None}
    # Team B
    for i, p in enumerate(sim.team_b):
        if agents_b[i] is not None:
            state = get_local_obs(sim, p, 'B')
            state_t = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                move, plog, slog = agents_b[i](state_t)
                pp = torch.sigmoid(plog).item()
                sp = torch.sigmoid(slog).item()
                if p.has_ball:
                    if np.random.random() < pp:
                        mates = [t for t in sim.team_b if t.id != p.id]
                        if mates: actions_b[p.id] = {"pass": np.random.choice(mates).id}; continue
                    if np.random.random() < sp:
                        actions_b[p.id] = {"shoot": None}; continue
                actions_b[p.id] = {"move": move.squeeze(0).tolist()}
        else:
            from server.formation import get_ideal_position
            ball_pos = (sim.ball.x, sim.ball.y)
            phase = 'attack' if sim.ball.x < 50 else 'defend'
            tx, ty = get_ideal_position(p, ball_pos, phase, 'B')
            dx, dy = tx - p.x, ty - p.y
            mag = np.hypot(dx, dy)
            if mag > 0: actions_b[p.id] = {"move": [dx/mag*7, dy/mag*7]}
            else: actions_b[p.id] = {"hold": None}

    env.step_multi_both(actions_a, actions_b)

    draw_pitch(ax)
    # Draw players with facing arrows
    for p in sim.team_a:
        ax.scatter(p.x, p.y, c='#1E90FF', s=180, edgecolors='white', linewidth=1.5, zorder=5)
        ax.text(p.x, p.y, str(p.id), color='white', fontsize=8, ha='center', va='center', zorder=6)
        # arrow towards ball
        dx, dy = sim.ball.x - p.x, sim.ball.y - p.y
        mag = np.hypot(dx, dy)
        if mag > 0.1:
            ax.arrow(p.x, p.y, dx/mag*2, dy/mag*2, head_width=1.2, head_length=1.2, fc='cyan', ec='cyan', alpha=0.6, zorder=7)
        # goalie prediction line (if id==0)
        if p.id == 0:
            # predict ball trajectory (simple: extrapolate current ball velocity)
            bx, by = sim.ball.x, sim.ball.y
            vx, vy = sim.ball.vx, sim.ball.vy
            steps = 10
            pred_x = bx + vx * DT * steps
            pred_y = by + vy * DT * steps
            ax.plot([bx, pred_x], [by, pred_y], 'yellow', lw=2, alpha=0.5, linestyle='--', zorder=3)

    for p in sim.team_b:
        ax.scatter(p.x, p.y, c='#FF4500', s=180, edgecolors='white', linewidth=1.5, zorder=5)
        ax.text(p.x, p.y, str(p.id), color='white', fontsize=8, ha='center', va='center', zorder=6)
        dx, dy = sim.ball.x - p.x, sim.ball.y - p.y
        mag = np.hypot(dx, dy)
        if mag > 0.1:
            ax.arrow(p.x, p.y, dx/mag*2, dy/mag*2, head_width=1.2, head_length=1.2, fc='yellow', ec='yellow', alpha=0.6, zorder=7)
        if p.id == 0:
            bx, by = sim.ball.x, sim.ball.y
            vx, vy = sim.ball.vx, sim.ball.vy
            steps = 10
            pred_x = bx + vx * DT * steps
            pred_y = by + vy * DT * steps
            ax.plot([bx, pred_x], [by, pred_y], 'yellow', lw=2, alpha=0.5, linestyle='--', zorder=3)

    # Ball
    ax.scatter(sim.ball.x, sim.ball.y, c='white', s=100, edgecolors='black', linewidth=1.5, zorder=10)
    if sim.ball.possessor:
        c = sim.ball.possessor
        ax.scatter(c.x, c.y, facecolors='none', edgecolors='yellow', s=220, linewidth=2.5, zorder=4)

fig, ax = plt.subplots(figsize=(14,9))
fig.patch.set_facecolor('#0b6623')
ani = animation.FuncAnimation(fig, update, frames=2000, interval=50, repeat=False)
plt.show()