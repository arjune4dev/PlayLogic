import requests, json

BASE = "http://localhost:8000"

def reset(): return requests.post(f"{BASE}/reset").json()
def step(action_text): return requests.post(f"{BASE}/step", json={"text": action_text}).json()
def state(): return requests.get(f"{BASE}/state").json()

if __name__ == "__main__":
    obs = reset()
    print(obs)
    for i in range(10):
        act = {pid: {"move": [0.5,0]} for pid in range(11)}
        obs, reward, done, info = step(json.dumps(act))
        print(reward, info)
        if done: break