import aprel
import numpy as np
import gym
from Throwing_task_demo import RobotEnv
env_name = 'Throw_env-v0'
gym_env = gym.make(env_name, gui=True, rec=True)
import math

np.random.seed(0)
gym_env.seed(0)


def feature_func(traj):
    """Returns the features of the given Throwing task trajectory, i.e. \Phi(traj).

    Args:
        traj: List of state-action tuples, e.g. [(state0, action0), (state1, action1), ...]

    Returns:
        features: a numpy vector corresponding the features of the trajectory
    """
    states = np.array([pair[0] for pair in traj])
    actions = np.array([pair[1] for pair in traj[:-1]])
    left_right = states[:, 1].max()
    x1, y1, z1 = states[:, 0], states[:, 1], states[:, 2]
    x2, y2, z2 = states[:, 3], states[:, 4], states[:, 5]
    l2normbuck1 = np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2) + ((z1 -  z2) ** 2))
    slope1 = list((l2normbuck1[1:] / l2normbuck1[:-1] - 1))

    slope1_sum = np.array(slope1).sum()
    print(slope1_sum)

    return np.array([left_right,slope1_sum])


env = aprel.Environment(gym_env, feature_func)

trajectory_set = aprel.generate_trajectories_randomly(env, num_trajectories=10,
                                                      max_episode_length=None,
                                                      file_name=env_name, seed=0)
features_dim = len(trajectory_set[0].features)

query_optimizer = aprel.QueryOptimizerDiscreteTrajectorySet(trajectory_set)

true_user = aprel.HumanUser(delay=3)

params = {'weights': aprel.util_funs.get_random_normalized_vector(features_dim)}
user_model = aprel.SoftmaxUser(params)
belief = aprel.SamplingBasedBelief(user_model, [], params)
print('Estimated user parameters: ' + str(belief.mean))

query = aprel.PreferenceQuery(trajectory_set[:2])

for query_no in range(3):
    queries, objective_values = query_optimizer.optimize('mutual_information', belief, query)
    print('Objective Value: ' + str(objective_values[0]))

    responses = true_user.respond(queries[0])
    belief.update(aprel.Preference(queries[0], responses[0]))
    print('Estimated user parameters: ' + str(belief.mean))
