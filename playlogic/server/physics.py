"""Realistic football physics – guaranteed goals, fast tackles."""
import math, numpy as np

FIELD_LENGTH, FIELD_WIDTH = 105.0, 68.0
GOAL_WIDTH = 7.32
GOAL_HALF = GOAL_WIDTH / 2
GOAL_Y_MIN = (FIELD_WIDTH / 2) - GOAL_HALF
GOAL_Y_MAX = (FIELD_WIDTH / 2) + GOAL_HALF
MAX_SPEED = 9.0
MAX_ACCEL = 7.0
PLAYER_RADIUS = 0.5
BALL_RADIUS = 0.11
DT = 0.1
BALL_FRICTION = 0.97
TACKLE_DIST = 0.65
POSS_DIST = 0.5 + BALL_RADIUS
PASS_SPEED = 16.0
SHOOT_SPEED = 22.0
INTERCEPT_DIST = 1.2

class Player:
    def __init__(self, x, y, team, player_id):
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0
        self.team = team
        self.id = player_id
        self.has_ball = False

class Ball:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0
        self.last_touch_team = None
        self.possessor = None

class FootballSim:
    def __init__(self):
        self.team_a = []
        self.team_b = []
        self.ball = Ball(FIELD_LENGTH/2, FIELD_WIDTH/2)
        self.score_a, self.score_b = 0, 0
        self.step_count = 0
        self.pass_events = []
        self.passes_in_flight = []
        self._init_players()

    def _init_players(self):
        from server.formation import get_initial_positions_team_a, get_initial_positions_team_b
        pos_a = get_initial_positions_team_a()
        pos_b = get_initial_positions_team_b()
        self.team_a = [Player(*pos_a[i], 'A', i) for i in range(11)]
        self.team_b = [Player(*pos_b[i], 'B', i) for i in range(11)]

    def reset(self):
        self.score_a = self.score_b = 0
        self.step_count = 0
        from server.formation import get_initial_positions_team_a, get_initial_positions_team_b
        pos_a = get_initial_positions_team_a()
        pos_b = get_initial_positions_team_b()
        for i, p in enumerate(self.team_a):
            p.x = pos_a[i][0] + np.random.uniform(-0.5,0.5)
            p.y = pos_a[i][1] + np.random.uniform(-0.5,0.5)
            p.vx = p.vy = 0.0; p.has_ball = False
        for i, p in enumerate(self.team_b):
            p.x = pos_b[i][0] + np.random.uniform(-0.5,0.5)
            p.y = pos_b[i][1] + np.random.uniform(-0.5,0.5)
            p.vx = p.vy = 0.0; p.has_ball = False
        self.ball.x, self.ball.y = FIELD_LENGTH/2, FIELD_WIDTH/2
        self.ball.vx = self.ball.vy = 0.0
        self.ball.last_touch_team = 'A'
        self.ball.possessor = None
        self._assign_ball_to_nearest('A')
        self.pass_events.clear()
        self.passes_in_flight.clear()

    def all_players(self):
        return self.team_a + self.team_b

    def _assign_ball_to_nearest(self, team):
        players = self.team_a if team == 'A' else self.team_b
        if not players: return
        best = min(players, key=lambda p: np.hypot(p.x-self.ball.x, p.y-self.ball.y))
        best.has_ball = True
        self.ball.possessor = best
        self.ball.last_touch_team = team
        self.ball.vx = self.ball.vy = 0.0
        self.ball.x, self.ball.y = best.x, best.y

    def step(self, actions_a, actions_b):
        self.pass_events.clear()
        self._apply_actions('A', actions_a)
        self._apply_actions('B', actions_b)
        self._update_physics()
        events = self._check_events()
        self._update_possession()
        self._update_passes()
        self.step_count += 1
        return events

    def _apply_actions(self, team, actions):
        players = self.team_a if team == 'A' else self.team_b
        for p in players:
            act = actions.get(p.id, {"hold": None})
            if "move" in act:
                dx, dy = act["move"]
                mag = np.hypot(dx, dy)
                if mag > MAX_SPEED:
                    dx = dx / mag * MAX_SPEED; dy = dy / mag * MAX_SPEED
                p.vx = p.vx * 0.3 + dx * 0.7
                p.vy = p.vy * 0.3 + dy * 0.7
            elif "shoot" in act and p.has_ball:
                self._initiate_shoot(p)
            elif "pass" in act and p.has_ball:
                target_id = act["pass"]
                target = next((m for m in players if m.id == target_id), None)
                if target:
                    self._initiate_pass(p, target)
            else:
                p.vx *= 0.9; p.vy *= 0.9

    def _initiate_pass(self, passer, target):
        dx, dy = target.x - passer.x, target.y - passer.y
        dist = np.hypot(dx, dy)
        if dist < 0.1: return
        self.ball.vx = dx / dist * PASS_SPEED
        self.ball.vy = dy / dist * PASS_SPEED
        passer.has_ball = False
        self.ball.possessor = None
        self.ball.last_touch_team = passer.team
        self.passes_in_flight.append({
            'team': passer.team, 'passer': passer,
            'target_id': target.id, 'target_pos': (target.x, target.y),
            'start_step': self.step_count
        })

    def _initiate_shoot(self, shooter):
        goal_y = np.random.uniform(GOAL_Y_MIN + 0.5, GOAL_Y_MAX - 0.5)
        goal_x = FIELD_LENGTH if shooter.team == 'A' else 0
        dx, dy = goal_x - shooter.x, goal_y - shooter.y
        dist = np.hypot(dx, dy)
        if dist > 0:
            self.ball.vx = dx / dist * SHOOT_SPEED
            self.ball.vy = dy / dist * SHOOT_SPEED
        shooter.has_ball = False
        self.ball.possessor = None
        self.ball.last_touch_team = shooter.team

    def _update_physics(self):
        for p in self.all_players():
            p.x += p.vx * DT; p.y += p.vy * DT
            p.x = max(PLAYER_RADIUS, min(FIELD_LENGTH-PLAYER_RADIUS, p.x))
            p.y = max(PLAYER_RADIUS, min(FIELD_WIDTH-PLAYER_RADIUS, p.y))
            p.vx *= 0.98; p.vy *= 0.98
            if abs(p.vx) < 0.05: p.vx = 0
            if abs(p.vy) < 0.05: p.vy = 0
            if p.has_ball:
                self.ball.x, self.ball.y = p.x, p.y
        if self.ball.possessor is None:
            self.ball.x += self.ball.vx * DT; self.ball.y += self.ball.vy * DT
            self.ball.vx *= BALL_FRICTION; self.ball.vy *= BALL_FRICTION
        # collisions
        players = self.all_players()
        for i in range(len(players)):
            for j in range(i+1, len(players)):
                pi, pj = players[i], players[j]
                dx, dy = pi.x - pj.x, pi.y - pj.y
                dist = np.hypot(dx, dy)
                if dist < 2*PLAYER_RADIUS and dist > 0:
                    overlap = 2*PLAYER_RADIUS - dist
                    nx, ny = dx/dist, dy/dist
                    pi.x += nx*overlap/2; pi.y += ny*overlap/2
                    pj.x -= nx*overlap/2; pj.y -= ny*overlap/2

    def _check_events(self):
        events = {'goal_a': False, 'goal_b': False, 'out': False}
        if self.ball.x > FIELD_LENGTH and GOAL_Y_MIN <= self.ball.y <= GOAL_Y_MAX:
            events['goal_a'] = True; self.score_a += 1
            self._reset_after_goal('B')
        elif self.ball.x < 0 and GOAL_Y_MIN <= self.ball.y <= GOAL_Y_MAX:
            events['goal_b'] = True; self.score_b += 1
            self._reset_after_goal('A')
        elif self.ball.x < 0 or self.ball.x > FIELD_LENGTH or self.ball.y < 0 or self.ball.y > FIELD_WIDTH:
            events['out'] = True
            if self.ball.last_touch_team == 'A': self._assign_ball_to_nearest('B')
            else: self._assign_ball_to_nearest('A')
            self.ball.x = FIELD_LENGTH/2; self.ball.y = FIELD_WIDTH/2
            self.ball.vx = self.ball.vy = 0
        return events

    def _reset_after_goal(self, kickoff_team):
        self.reset()
        self.ball.last_touch_team = kickoff_team
        self._assign_ball_to_nearest(kickoff_team)

    def _update_possession(self):
        if self.ball.possessor is None:
            for p in self.all_players():
                if not p.has_ball and np.hypot(p.x-self.ball.x, p.y-self.ball.y) < POSS_DIST:
                    p.has_ball = True; self.ball.possessor = p
                    self.ball.last_touch_team = p.team
                    self.ball.vx = self.ball.vy = 0.0
                    self.ball.x, self.ball.y = p.x, p.y
                    break
        if self.ball.possessor is not None:
            carrier = self.ball.possessor
            opps = self.team_b if carrier.team == 'A' else self.team_a
            for opp in opps:
                dist = np.hypot(carrier.x-opp.x, carrier.y-opp.y)
                if dist < TACKLE_DIST:
                    prob = min(1.0, 2.0 * (1 - dist/TACKLE_DIST))
                    if np.random.random() < prob:
                        carrier.has_ball = False
                        opp.has_ball = True
                        self.ball.possessor = opp
                        self.ball.last_touch_team = opp.team
                        self.ball.vx = self.ball.vy = 0.0
                        self.ball.x, self.ball.y = opp.x, opp.y
                        break

    def _update_passes(self):
        for flight in self.passes_in_flight[:]:
            if self.ball.possessor is not None or self.step_count - flight['start_step'] > 30:
                self.passes_in_flight.remove(flight); continue
            opp_team = self.team_b if flight['team'] == 'A' else self.team_a
            for opp in opp_team:
                if np.hypot(opp.x-self.ball.x, opp.y-self.ball.y) < INTERCEPT_DIST:
                    self.pass_events.append({'type':'intercept', 'team':flight['team']})
                    opp.has_ball = True; self.ball.possessor = opp
                    self.ball.last_touch_team = opp.team
                    self.ball.vx = self.ball.vy = 0.0
                    self.ball.x, self.ball.y = opp.x, opp.y
                    self.passes_in_flight.remove(flight); return
            tx, ty = flight['target_pos']
            if np.hypot(self.ball.x-tx, self.ball.y-ty) < 2.0:
                self.pass_events.append({'type':'success', 'team':flight['team']})
                target = next((p for p in (self.team_a if flight['team']=='A' else self.team_b) if p.id == flight['target_id']), None)
                if target:
                    target.has_ball = True; self.ball.possessor = target
                    self.ball.last_touch_team = flight['team']
                    self.ball.vx = self.ball.vy = 0.0
                    self.ball.x, self.ball.y = target.x, target.y
                self.passes_in_flight.remove(flight)

    def get_state_dict(self):
        return {
            'step': self.step_count,
            'score': f"{self.score_a}-{self.score_b}",
            'ball': {'x':self.ball.x, 'y':self.ball.y, 'vx':self.ball.vx, 'vy':self.ball.vy},
            'possession_team': self.ball.last_touch_team,
            'team_a': [{'id':p.id, 'x':p.x, 'y':p.y, 'vx':p.vx, 'vy':p.vy, 'has_ball':p.has_ball} for p in self.team_a],
            'team_b': [{'id':p.id, 'x':p.x, 'y':p.y, 'vx':p.vx, 'vy':p.vy, 'has_ball':p.has_ball} for p in self.team_b],
        }