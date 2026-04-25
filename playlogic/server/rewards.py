import numpy as np
from collections import defaultdict
from server.formation import get_ideal_position

def compute_rewards(sim, events):
    r = defaultdict(float)
    if events['goal_a']: r['goal'] = 5.0
    if events['goal_b']: r['goal_conceded'] = -5.0

    for pe in sim.pass_events:
        if pe['team'] == 'A':
            if pe['type'] == 'success': r['pass_success'] += 0.5
            else: r['pass_intercepted'] -= 0.3

    # formation cohesion
    ball_pos = (sim.ball.x, sim.ball.y)
    phase = 'attack' if sim.ball.x >= 50 else 'defend'
    for p in sim.team_a:
        ideal = get_ideal_position(p, ball_pos, phase, 'A')
        dist = np.hypot(p.x-ideal[0], p.y-ideal[1])
        r['cohesion'] += 0.02 * (1 - np.tanh(dist))

    # spacing penalty
    for i in range(len(sim.team_a)):
        for j in range(i+1, len(sim.team_a)):
            if np.hypot(sim.team_a[i].x - sim.team_a[j].x, sim.team_a[i].y - sim.team_a[j].y) < 2.0:
                r['spacing'] -= 0.01

    # pressing reward
    if sim.ball.possessor and sim.ball.possessor.team != 'A':
        carrier = sim.ball.possessor
        for p in sim.team_a:
            if np.hypot(p.x-carrier.x, p.y-carrier.y) < 2.0:
                r['pressing'] += 0.1
                break

    return dict(r)