import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.model import ActorCritic
import torch.nn.functional as F
import numpy as np

import gfootball.env as football_env


def get_args():
    parser = argparse.ArgumentParser("A3C for gfootball")
    parser.add_argument('--env_name', type=str, default='academy_3_vs_1_with_keeper')
    # parser.add_argument("--save_interval", type=int, default=50, help="Number of episode between savings")
    parser.add_argument("--load_path", type=str, default="/content/drive/My Drive/Super-mario-bros-A3C-pytorch-master/trained_models/params2.pkl")
    parser.add_argument("--play_episodes",type=int,default=2000)
    args = parser.parse_args()
    return args
    
    

# convert the numpy array to tensors
def _get_tensors(opt, obs):
    obs_tensor = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    # decide if put the tensor on the GPU
    if torch.cuda.is_available():
        obs_tensor = obs_tensor.cuda()
    return obs_tensor
    

def test(opt):
    torch.manual_seed(123)
    env = football_env.create_environment(env_name=opt.env_name,
                                      stacked=True,
                                      representation='extracted',
                                      render=False)
    num_states = 16
    num_actions = env.action_space.n
    model = ActorCritic(num_states, num_actions)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(opt.load_path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(opt.load_path, map_location=torch.device('cpu')))
    model.eval()
    state = env.reset()
    done = True
    
    num_episode = 0
    total_reward = 0.0
    total_step = 0
    while num_episode<=opt.play_episodes:
        if done:
            num_episode += 1
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            env.reset()
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
        
        obs_tensor = _get_tensors(opt,state)
        logits, value, h_0, c_0 = model(obs_tensor, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        state, reward, done, info = env.step(action)
        total_reward += reward
        total_step += 1
        
    print('After {} episodes, the average reward is {}, average step is {}'.format(num_episode-1,total_reward/(num_episode-1),total_step/(num_episode-1)))



if __name__ == "__main__":
    opt = get_args()
    test(opt)
