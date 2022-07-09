import random
import time


class NavDemo:
    def __init__(self):
        self.evaded_obstacle = False
        self.threshold_space = [1.40,1.5, 1.60,1.7, 1.8]
        self.threshold = random.choice(self.threshold_space)
        self.action_space = [1, 2]  # left , right
        self.evaded_action = None
        self.n_seconds = 1.8
        self.done = False
        print(self.threshold)

    def policy(self, distance_rob_obs):
        if distance_rob_obs <= self.threshold and self.evaded_obstacle == False:
            print(self.threshold)
            action = random.choice(self.action_space)
            self.evaded_action = action
            self.evaded_obstacle = True
            # self.threshold = random.choice(self.threshold_space)
        elif self.evaded_obstacle == True and self.done == False:
            time.sleep(self.n_seconds)
            #print(self.evaded_action)
            action = 0
            self.done = True
        else:
            action = 0
        return action
