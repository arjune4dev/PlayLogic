"""OpenEnv‑style environment + multi‑agent stepping."""
import json
import numpy as np
from models import Observation, Action
from server.physics import FootballSim
from server.opponent import opponent_actions
from server.rewards import compute_rewards

# ----- Fallback OpenEnv base (remove when real OpenEnv is used) -----
class Environment:
    pass
def reset(func): return func
def step(func): return func
def state(func): return func
# --------------------------------------------------------------------

class FootballTacticsMARL(Environment):
    def __init__(self):
        super().__init__()
        self.sim = FootballSim()

    @reset
    def reset(self) -> Observation:
        self.sim.reset()
        return Observation(text=self._build_obs())

    @step
    def step(self, action: Action):
        """Original JSON‑string step (for LLM interface)."""
        try:
            acts = json.loads(action.text)
            acts = {int(k): v for k, v in acts.items()}
        except:
            return Observation(text=self._build_obs()), -1.0, True, {"error": "invalid_json"}
        for pid in acts:
            if pid not in range(11):
                return Observation(text=self._build_obs()), -1.0, True, {"error": f"bad_id {pid}"}
        actions_b = opponent_actions(self.sim)
        events = self.sim.step(acts, actions_b)
        rewards = compute_rewards(self.sim, events)
        total = sum(rewards.values())
        done = (self.sim.step_count >= 2000 or self.sim.score_a >= 5 or self.sim.score_b >= 5)
        return Observation(text=self._build_obs()), total, done, {"rewards": rewards, "score": f"{self.sim.score_a}-{self.sim.score_b}"}

    def step_multi_both(self, team_a_actions: dict, team_b_actions: dict):
        for p in self.sim.team_a:
            if p.id not in team_a_actions:
                team_a_actions[p.id] = {"hold": None}
        for p in self.sim.team_b:
            if p.id not in team_b_actions:
                team_b_actions[p.id] = {"hold": None}
        events = self.sim.step(team_a_actions, team_b_actions)
        rewards = compute_rewards(self.sim, events)
        total = sum(rewards.values())
        done = (self.sim.step_count >= 2000 or self.sim.score_a >= 5 or self.sim.score_b >= 5)
        return self._build_obs(), total, done, {"rewards": rewards, "score": f"{self.sim.score_a}-{self.sim.score_b}"}
    
    
    def step_multi(self, team_a_actions: dict):
        """Step with per‑player action dicts (from multiple agents)."""
        for p in self.sim.team_a:
            if p.id not in team_a_actions:
                team_a_actions[p.id] = {"hold": None}
        actions_b = opponent_actions(self.sim)
        events = self.sim.step(team_a_actions, actions_b)
        rewards = compute_rewards(self.sim, events)
        total = sum(rewards.values())
        done = (self.sim.step_count >= 2000 or self.sim.score_a >= 5 or self.sim.score_b >= 5)
        return self._build_obs(), total, done, {"rewards": rewards, "score": f"{self.sim.score_a}-{self.sim.score_b}"}

    def _build_obs(self):
        s = self.sim.get_state_dict()
        t = f"Step: {s['step']}\nScore: {s['score']}\nBall: ({s['ball']['x']:.1f},{s['ball']['y']:.1f})\n"
        t += "Team A:\n"
        for p in s['team_a']:
            t += f"{p['id']}: ({p['x']:.1f},{p['y']:.1f}) v=({p['vx']:.1f},{p['vy']:.1f}) ball={p['has_ball']}\n"
        t += "Team B:\n"
        for p in s['team_b']:
            t += f"{p['id']}: ({p['x']:.1f},{p['y']:.1f}) v=({p['vx']:.1f},{p['vy']:.1f}) ball={p['has_ball']}\n"
        return t

    @state
    def state(self):
        return self.sim.get_state_dict()