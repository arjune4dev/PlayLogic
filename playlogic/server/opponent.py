"""Rule‑based opponent AI (Team B)."""
import numpy as np
from server.physics import FIELD_LENGTH, FIELD_WIDTH, MAX_SPEED

def opponent_actions(sim):
    actions = {}
    for p in sim.team_b:
        act = {}
        if p.has_ball:
            if should_shoot(p, sim):
                act["shoot"] = None
            elif np.random.random() < 0.7:
                target = choose_pass_target(p, sim)
                if target is not None:
                    act["pass"] = target.id
                else:
                    act["move"] = dribble_move(p)
            else:
                act["move"] = dribble_move(p)
        else:
            act["move"] = move_towards_ideal(p, sim)
        actions[p.id] = act
    return actions

def should_shoot(player, sim):
    if player.team != 'B': return False
    dist_to_goal = player.x   # shoot left -> x=0
    if dist_to_goal > 20: return False
    for p in sim.team_b:
        if p.id != player.id and p.x < player.x - 2:
            return False
    return np.random.random() < 0.5

def choose_pass_target(player, sim):
    best_val = -1
    best_target = None
    for p in sim.team_b:
        if p.id == player.id or p.has_ball: continue
        dx = p.x - player.x
        dy = p.y - player.y
        dist = np.hypot(dx, dy)
        if dist < 2: continue
        opp_near = 0
        for opp in sim.team_a:
            if point_to_segment_distance(opp.x, opp.y, player.x, player.y, p.x, p.y) < 3.0:
                opp_near += 1
        val = 1.0 / (dist + opp_near * 5)
        if val > best_val:
            best_val = val
            best_target = p
    return best_target

def point_to_segment_distance(px, py, ax, ay, bx, by):
    ABx, ABy = bx - ax, by - ay
    if ABx == 0 and ABy == 0: return np.hypot(px - ax, py - ay)
    APx, APy = px - ax, py - ay
    t = (APx * ABx + APy * ABy) / (ABx*ABx + ABy*ABy)
    t = max(0, min(1, t))
    closest_x = ax + t * ABx
    closest_y = ay + t * ABy
    return np.hypot(px - closest_x, py - closest_y)

def dribble_move(player):
    target_x = 0 + np.random.uniform(10, 30)
    target_y = player.y + np.random.uniform(-5, 5)
    dx = target_x - player.x
    dy = target_y - player.y
    mag = np.hypot(dx, dy)
    if mag > 0:
        speed = min(mag, MAX_SPEED)
        return [dx/mag * speed, dy/mag * speed]
    return [0.0, 0.0]

def move_towards_ideal(player, sim):
    from server.formation import get_ideal_position
    ball_pos = (sim.ball.x, sim.ball.y)
    phase = 'attack' if sim.ball.x < 50 else 'defend'
    tx, ty = get_ideal_position(player, ball_pos, phase, 'B')
    dx = tx - player.x
    dy = ty - player.y
    mag = np.hypot(dx, dy)
    if mag > 0:
        speed = min(mag, MAX_SPEED)
        return [dx/mag * speed, dy/mag * speed]
    return [0.0, 0.0]