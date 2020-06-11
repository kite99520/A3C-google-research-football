import torch
from src.model import ActorCritic
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
# from tensorboardX import SummaryWriter
import timeit
import gfootball.env as football_env
import numpy as np
from datetime import datetime


# convert the numpy array to tensors
def _get_tensors(opt, obs):
    obs_tensor = torch.tensor(np.transpose(obs, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    # decide if put the tensor on the GPU
    if opt.use_gpu:
        obs_tensor = obs_tensor.cuda()
    return obs_tensor


# adjust the learning rate
def _adjust_learning_rate(update, num_updates, opt, optimizer):
    lr_frac = 1 - (update / num_updates)
    adjust_lr = opt.lr * lr_frac
    for param_group in optimizer.param_groups:
        param_group['lr'] = adjust_lr
          

def local_train(index, opt, global_model, optimizer, save=False):
    torch.manual_seed(123 + index)  # set random seed
    if save:
        start_time = timeit.default_timer()
    # writer = SummaryWriter(opt.log_path)
    env = football_env.create_environment(env_name=opt.env_name,
                                                                   stacked=True,
                                                                   representation='extracted',
                                                                   render=False)
    num_states = 16
    num_actions = env.action_space.n
    local_model = ActorCritic(num_states, num_actions)
    if opt.use_gpu:
        local_model.cuda()
    local_model.train()

    state = env.reset()
    done = True
    curr_step = 0
    curr_episode = 0
    total_reward = 0 
    while True:
        if save:
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                torch.save(global_model.state_dict(),opt.saved_path_file)  # save path
        if curr_episode % opt.print_interval == 0 and curr_episode > 0:    
            print("[{}] Process {}. Episode {}, average_reward {:2f}".format(datetime.now(), index, curr_episode, total_reward/curr_episode))
        
        curr_episode += 1
        local_model.load_state_dict(global_model.state_dict())  # load global model
        if done:  # an episode done
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if opt.use_gpu:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        log_policies = []
        values = []
        rewards = []
        entropies = []

        for _ in range(opt.num_local_steps):
            curr_step += 1
            obs_tensor = _get_tensors(opt, state)
            logits, value, h_0, c_0 = local_model(obs_tensor, h_0, c_0)
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()

            state, reward, done, _ = env.step(action)
            if curr_step > opt.num_global_steps:  # greater than the total max step, stop
                done = True

            if done:
                curr_step = 0
                state = env.reset()

            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            total_reward += reward

            if done:
                break

        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not done:
            obs_tensor = _get_tensors(opt, state)
            _, R, _, _ = local_model(obs_tensor, h_0, c_0)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R

        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
        # writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)

        if opt.lr_decay:
            _adjust_learning_rate(curr_episode, int(opt.num_global_steps / opt.num_local_steps),opt,optimizer)
        optimizer.zero_grad()
        total_loss.backward()

        # shared the gradient between local and global 
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        optimizer.step()

        # print info
        if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return


