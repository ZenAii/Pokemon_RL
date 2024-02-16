import sys
import time
import numpy as np
from PIL.Image import init
from numpy.core.numeric import roll
from pyboy import PyBoy, WindowEvent
import hnswlib

from rollout import Rollout

class RedEnv:

    def __init__(self, headless=True, action_freq=5, init_state='init.state', debug=False):

        self.debug = debug
        self.vec_dim = 1080
        self.num_elements = 20000
        self.init_state = init_state
        self.act_freq = action_freq
        self.downsample_factor = 8
        self.similar_frame_dist = 1500000.0

        head = 'headless' if headless else 'SDL2'

        self.pyboy = PyBoy(
            "./PokemonRed.gb",
            debugging=False,
            disable_input=head,
            hide_window="--quiet" in sys.argv,
        )

        self.reset_game() 


        self.pyboy.set_emulation_speed(0)

        def reset_game(self):
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)

                def play_episode(self, agent, max_episode_steps):

                    frame = 0
                    self.reset_game()

                    self.knn_index = hnswlib.Index(space='12', dim=self.vec_dim)
                    self.knn_index.init_index(max_elements=self.num_elements, ef_construction=100, M=16)

                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    rollout = Rollout(timestr, agen.get_name())

                    while not self.pyboy.tick() and frame < max_episode_steps:

                        if frame % self.act_freq == 0:

                            game_pixels_render = self.pyboy.screen_image()
                            x, y = game_pixels_render.size
                            state_np = np.array(game_pixels_render)

                            next_action = agent.get_action(state_np, rollout)
                            if next_action is not None:
                                self.pyboy.send_input(next_action)

                                rollout.add_state_action_pair(state_np, next_action)

                                state = np.array(game_pixels_render.resize(
                                    (x//self.downsample_factor,
                                     y//self.downsample_factor)
                                )).reshape(-1)

                                if (self.knn_index.get_current_count() == 0):
                                    self.knn_index.add_items(state, np.array([self.knn_index.get_current_count()]))

                                    labels, distances = self.knn_index.knn_query(state, k = 1)
                                    if (distances[0] > self.similar_frame_dist):
                                        self.knn_index.add_items(state, np.array([self.knn_index.get_current_count()]))

                                        rollout.set_reward(self.knn_index.get_current_count())

                                        if self.debug:
                                            print(frame)
                                            print(f'{self.knn_index.get_current_count()} total frames indexed, current closest is: {distance[0]}')

                                    frame += 1

                                    rollout.save_to_file()

                                    return self.knn_index.get_current_count()