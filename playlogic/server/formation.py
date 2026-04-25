import numpy as np
FIELD_LENGTH, FIELD_WIDTH = 105.0, 68.0

def get_initial_positions_team_a():
    return [(5,34), (20,15), (20,27.7), (20,40.3), (20,53),
            (40,20), (40,34), (40,48), (55,15), (55,34), (55,53)]

def get_initial_positions_team_b():
    return [(100,34), (85,15), (85,27.7), (85,40.3), (85,53),
            (65,20), (65,34), (65,48), (50,15), (50,34), (50,53)]

def get_ideal_position(player, ball_pos, phase, team):
    xb, yb = ball_pos
    if team == 'A':
        if phase == 'attack': off, mid, deff = xb+10, xb-10, xb-25
        else: off, mid, deff = xb-5, xb-15, xb-30
    else:
        if phase == 'attack': off, mid, deff = xb-10, xb+10, xb+25
        else: off, mid, deff = xb+5, xb+15, xb+30
    deff = max(5, min(100, deff)); mid = max(5, min(100, mid)); off = max(5, min(100, off))
    pid = player.id
    if team == 'A':
        if pid == 0: return (5,34)
        if 1 <= pid <= 4: y = [15,27.7,40.3,53][pid-1]; return (deff, y)
        if 5 <= pid <= 7: y = [20,34,48][pid-5]; return (mid, y)
        y = [15,34,53][pid-8]; return (off, y)
    else:
        if pid == 0: return (100,34)
        if 1 <= pid <= 4: y = [15,27.7,40.3,53][pid-1]; return (deff, y)
        if 5 <= pid <= 7: y = [20,34,48][pid-5]; return (mid, y)
        y = [15,34,53][pid-8]; return (off, y)