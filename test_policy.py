import gym
import matplotlib
import torch
import numpy as np

from sac.model import GaussianPolicy, QNetwork, DeterministicPolicy

# from core.notebook_utils import animate
# from core.notebook_utils import gen_video


seed = 123456
hidden_size = 256
device = 'cpu'

# env_name = 'Hopper-v2'
env_name = 'Walker2d-v2'
# env_name = 'HalfCheetah'

model_path = './model_last.pt'

env = gym.make(env_name)
env.seed(seed)

policy = GaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0], hidden_size, env.action_space).to(device)
policy.load_state_dict(torch.load(model_path, map_location=torch.device(device))['Policy'])


def select_action(state, policy, device):
    state = torch.FloatTensor(state).to(device).unsqueeze(0)
    action, log_prob, mean = policy.sample(state)
#     return mean.detach().cpu().numpy()[0]
    return action.detach().cpu().numpy()[0]


# Now we generate video to see the performance.
state = env.reset()
state = np.expand_dims(state, axis=0)

frames = []
rewards = []
for i in range(1000):
    # frame = env.render(mode='rgb_array')
    state, reward, done, info = env.step(select_action(state, policy, device))
    # frames.append(frame.copy())
    rewards.append(reward)
    if i % 100 == 0:
        print("Step: {}, reward: {}".format(i, reward) )
    if done:
        break

print('total reward: {}'.format(np.sum(rewards)))