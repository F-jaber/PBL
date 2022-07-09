# Robot Learning from Human-Preference via Active Querying


Welcome!

This repository contains the code I have used in my bachelor's thesis project about teaching a robot through preferenced-based learning through actively querying the user. I use the [APReL](https://github.com/Stanford-ILIAD/APReL)  library to query users and calculate the weights of the trajectory features according to the users' preference. For my thesis research, there are 2 tasks to be learned by the robot: an obstacle avoidance task and throwing a ball into a bucket. Both tasks are done in simulation using [qiBullet](https://github.com/softbankrobotics-research/qibullet). Moreover, in order to use both tasks with APReL, they were implemented as [Gym](https://gym.openai.com/docs/) environments.


## Getting Started
In order to use this code there are some prerequisites that need to be fulfilled.
### Prerequisites
Please install the following:
- [APReL](https://github.com/Stanford-ILIAD/APReL)
- [PyBullet](https://github.com/bulletphysics/bullet3)
- [FFmpeg](https://github.com/bulletphysics/bullet3)
- [numpy](https://numpy.org/install/)
- [qiBullet](https://github.com/softbankrobotics-research/qibullet)
- [Gym v0.21.0](https://github.com/openai/gym/releases/tag/v0.21.0)

### Installing

Just clone this repository onto your pc.

```
git clone https://github.com/F-jaber/PBL.git
```
After cloning, please make sure to register both tasks' environments as Gym environments on your pc. Follow this [tutorial](https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952) for more information on registering Gym environments.
## Running the project
To run 

![Obstacle Avoidance Task](https://i.ibb.co/185rGbz/Picture1.gif)
![Throwing Task](https://i.ibb.co/KzbtTfX/Picture2.gif)
