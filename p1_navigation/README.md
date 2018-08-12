# Submission - Project 1: Navigation

>-- :banana: :bowtie: :banana:  -- Spoiler: problem solved in 431 Episodes --  :banana: :bowtie: :banana: 

| ![GIF of my trained agent, trained in 431 episodes](report_submission/demo_banana.gif "GIF of my trained agent, trained in 431 episodes")  | 
|:--:| 
| *GIF of my trained agent, trained in 431 episodes* |

### Problem Setting
For this project, I had to train an agent to navigate and collect bananas in a large, square world.  

#### Details about the project environment
-  The state space
    - The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.
-  The action space
    - Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
        - **`0`** - move forward.
        - **`1`** - move backward.
        - **`2`** - turn left.
        - **`3`** - turn right.
-  Reward function
    - A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  
-  When the environment is considered solved
    - The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 


### Instructions
The repository is structured as follow.
- [`main_banana.ipynb`](src_submission/main_banana.ipynb) is *the central file you want to use*. It contains
    - all the import statements and instructions to start the environment
    - the calls to *train*
    - the calls to *test*
- [`dqn_agent_banana.py`](src_submission/dqn_agent_banana.py) defines two classes
    - Agent with methods such as step, act, learn 
    - ReplayBuffer to store experience tuples 
- [`model_banana.py`](src_submission/model_banana.py) defines the Q-Network used by the Agent as a function approximation for Q-values
- [`checkpoint_banana_431.pth`](src_submission/checkpoint_banana_431.pth) are the saved model weights of the successful agent weights

### Report
[`report.ipynb`](report.ipynb) describes choices and details results. It includes
- Description of the model architectures 
- Description of the hyperparameters
- Plot of Rewards
- Ideas for Future Work

| ![GIF of my agent being trained at episode #200, with epsilon=0.5](report_submission/training-eps-50-percent.gif "training-eps-50-percent") | 
|:--:| 
| *GIF of my agent being trained at episode #200, with epsilon=0.5* |


### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
