# Robot Learning from Human-Preference via Active Querying


Welcome!

This repository contains the code I have used in my bachelor's thesis project about teaching a robot through preferenced-based learning through actively querying the user. I use the [APReL](https://github.com/Stanford-ILIAD/APReL)  library to query users and calculate the weights of the trajectory features according to the users' preference. For my thesis research, there are 2 tasks to be learned by the robot: an obstacle avoidance task and throwing a ball into a bucket. Both tasks are done in simulation using [qiBullet](https://github.com/softbankrobotics-research/qibullet). Moreover, in order to use both tasks with APReL, they were implemented as [Gym](https://gym.openai.com/docs/) environments. Finally, to investigate how consistency in preference affects the learning performance of a robot, I implmented a simple oracle with diffrent corruption (consistency) rates that corrupts (changes) a user's preference choice.
### Tasks

| Obstacle Avoidance Task  | Throwing Task |
| ------------- | ------------- |
| ![Obstacle Avoidance Task](https://i.ibb.co/185rGbz/Picture1.gif)  | ![Throwing Task](https://i.ibb.co/KzbtTfX/Picture2.gif)  |


## Getting Started
In order to use this code there are some prerequisites that need to be fulfilled.
### Prerequisites
Please install the following:
- [APReL](https://github.com/Stanford-ILIAD/APReL)
- [PyBullet](https://github.com/bulletphysics/bullet3)
- [FFmpeg](https://github.com/F-jaber/PBL)
- [numpy](https://numpy.org/install/)
- [qiBullet](https://github.com/softbankrobotics-research/qibullet)
- [Gym v0.21.0](https://github.com/openai/gym/releases/tag/v0.21.0)

### Installing

Just clone this repository onto your pc.

```
git clone https://github.com/F-jaber/PBL.git
```
After cloning, please make sure to register both tasks' environments as Gym environments on your pc. Follow this [tutorial](https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952) for more information on registering Gym environments.

Moreover, replace the original generate_trajectories.py file in the utils folder in the APReL library by the generate_trajectories.py file cloned from this repository.
## Running the project
To query for a user's preference for the obstacle avoidance task, simply run the run_Obs.py file. If you would like to query the preference for the throwing task simply run the run_Throw.py file. You can set the number of queries to ask the user in the run files. Users will always be shown two trajectories of the task at a time and can give their preference of which trajectory they prefer through keyboard input (0 if they  the first shown trajectory, 1 if they prefer the second shown trajectory). When the querying is done, APReL will output the weights of the trajectory features. You can then incorporate these weights in the reward function implemented in the task's environment file. Afterward, you can use your favorite RL training algorithm to train the robot using the incorporated weights which should have captured the user's preference. I personally used [Truly PPO](https://github.com/wisnunugroho21/reinforcement_learning_ppo_rnd) in my thesis. Please make sure the environment's gui parameter is set to True (and rec to False) while training.

Additionally, to use the experment demonstrations I provided please simply past them into the utils file of APReL and make sure to uncomment the approperiate lines of code  in the generate_trajectories.py file depending on which task you're using.


Finally, if you would like to use the consistency oracle, simply replace the original data_types.py in the learning folder of APReL with the data_types.py provided here. To change the corruption rate simple change the value of the c_rate variable in the data_types.py file.

Enjoy!

