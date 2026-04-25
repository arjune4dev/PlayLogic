import numpy as np
from playlogic.server.physics import SPACING_THRESHOLD, FIELD_WIDTH
from playlogic.server.formation import get_ideal_position

def compute_reward_for_team(state, prev_state, team):
    """Compute reward for specified team ('A' or 'B')."""
    reward = 0.0
    breakdown = {}
    players = state['players']
    ball = state['ball']
    score = state['score']
    my_team = [p for p in players if p['team'] == team]
    opp_team = [p for p in players if p['team'] != team]

    # Goal (use score change)
    if prev_state:
        if team == 'A' and score['A'] > prev_state['score']['A']:
            return 5.0, {'goal_scored': 5.0}
        if team == 'B' and score['B'] > prev_state['score']['B']:
            return 5.0, {'goal_scored': 5.0}
        if team == 'A' and score['B'] > prev_state['score']['B']:
            return -5.0, {'goal_conceded': -5.0}
        if team == 'B' and score['A'] > prev_state['score']['A']:
            return -5.0, {'goal_conceded': -5.0}

    # Turnover
    if prev_state and state['possession_team'] != prev_state['possession_team']:
        if prev_state['possession_team'] == team and state['possession_team'] != team:
            reward -= 0.1

    # Pass result
    if 'pass_result' in state and state['pass_result'] == 'success':
        # Check if the passing team is the one we're computing for
        pass_team = None
        for p in players:
            if p['id'] == state['pass_from']:
                pass_team = p['team']
                break
        if pass_team == team:
            reward += 0.5
        elif pass_team is not None:
            reward -= 0.3  # opponent's successful pass is bad
    if 'pass_result' in state and state['pass_result'] == 'intercepted':
        # interceptor gains possession
        if state['possession_team'] == team:
            reward += 0.3
        else:
            reward -= 0.1

    # Formation cohesion
    cohesion = 0.0
    ball_pos = ball['pos']
    for p in my_team:
        if p['id'] == 0:  # skip GK for simplicity
            continue
        ideal = get_ideal_position(team, p['id'], ball_pos)
        dist = np.linalg.norm(p['pos'] - ideal)
        cohesion += 1.0 - np.tanh(dist)
    reward += 0.02 * cohesion

    # Spacing penalty (if players bunch)
    # (Implementation omitted for brevity)
    return reward, {}