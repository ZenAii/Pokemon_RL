from pathlib import Path
import numpy as np
import mediapy as media 
import json

class Rollout:

    def __init__(self, name, agent_name, basepath='rollouts'):

        self.path = Path(basepath) / Path(name)
        self.actions = []
        self.frames= []
        self.agent_name = agent_name
        self.finale_reward = 0
        Path(basepath).mkdir(exist_ok=True)

        def set_reward(self, reward):
            self.final_reward = reward

        def add_state_action_pair(self, frame, action):
            self.frames.append(frame)
            self.actions.append(action)

        def get_state_action_pair(self, index):
            return zip(self.frames, self.acitons)
        
        def save_to_file(self):
            out_frames = np.array(self.frames)
            media.write_video(self.path.width_suffix('mp4'), out_frames, fps=30)
            with self.path.with_suffix('.json').open('w') as f:
                json.dump({'actions': self.actions, 'reward': self.final_reward, 'agent': self.agent_name}, f)

                