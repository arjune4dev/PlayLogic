"""Run random policy for 10 episodes and print average reward."""
import json
import numpy as np
from server.environment import FootballTacticsMARL
from models import Action

def main():
    env = FootballTacticsMARL()
    total_rewards = []
    for ep in range(10):
        obs = env.reset()
        done = False
        ep_reward = 0
        while not done:
            # random actions
            actions = {}
            for i in range(11):
                if np.random.random() < 0.5:
                    actions[i] = {"move": [np.random.uniform(-8,8), np.random.uniform(-8,8)]}
                else:
                    actions[i] = {"hold": None}
            act = Action(text=json.dumps(actions))
            obs, reward, done, info = env.step(act)
            ep_reward += reward
        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}, Reward: {ep_reward:.2f}, Score: {info['score']}")
    print(f"Average reward over 10 episodes: {np.mean(total_rewards):.2f}")

if __name__ == "__main__":
    main()