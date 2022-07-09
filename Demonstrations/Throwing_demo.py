import random


class ThrowDemo:
    def __init__(self):
        self.right_buck = [[0.401025, 0.5200125, -0.208229067794172, 0.3503409, 0.998564],
                           [0.397587, 0.526538, -0.48908229067, 0.41258968, -0.8548954],
                           [0.265945, -0.3615, -0.32082267794172, -0.5303409, 0.7548954]]

    def policy(self):
        action = random.choice(self.right_buck)
        return action
