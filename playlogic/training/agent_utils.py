import numpy as np

def get_local_obs(sim, player, team='A'):
    """40‑dim observation vector for a specific player."""
    obs = []
    # Own state (4)
    obs.extend([player.x / 105.0, player.y / 68.0,
                player.vx / 8.0, player.vy / 8.0])
    # Ball state (4)
    obs.extend([sim.ball.x / 105.0, sim.ball.y / 68.0,
                sim.ball.vx / 8.0, sim.ball.vy / 8.0])
    # Has ball (1)
    obs.append(1.0 if player.has_ball else 0.0)
    # Distance to opponent goal (1)
    if team == 'A':
        goal_dist = (105.0 - player.x) / 105.0
    else:
        goal_dist = player.x / 105.0
    obs.append(goal_dist)
    # Nearest teammate (4)
    teammates = sim.team_a if team == 'A' else sim.team_b
    min_tm = 1e9
    near_tm = [0.0, 0.0, 0.0, 0.0]
    for tm in teammates:
        if tm.id == player.id:
            continue
        d = np.hypot(tm.x - player.x, tm.y - player.y)
        if d < min_tm:
            min_tm = d
            near_tm = [tm.x / 105.0, tm.y / 68.0, tm.vx / 8.0, tm.vy / 8.0]
    obs.extend(near_tm)
    # Nearest opponent (4)
    opponents = sim.team_b if team == 'A' else sim.team_a
    min_op = 1e9
    near_op = [0.0, 0.0, 0.0, 0.0]
    for op in opponents:
        d = np.hypot(op.x - player.x, op.y - player.y)
        if d < min_op:
            min_op = d
            near_op = [op.x / 105.0, op.y / 68.0, op.vx / 8.0, op.vy / 8.0]
    obs.extend(near_op)
    # Role one‑hot (3): goalie, defender, attacker
    role = [0, 0, 0]
    if player.id == 0:
        role = [1, 0, 0]        # goalie
    elif 1 <= player.id <= 4:
        role = [0, 1, 0]        # defender
    else:
        role = [0, 0, 1]        # attacker
    obs.extend(role)
    # Pad to exactly 40
    while len(obs) < 40:
        obs.append(0.0)
    return np.array(obs[:40], dtype=np.float32)