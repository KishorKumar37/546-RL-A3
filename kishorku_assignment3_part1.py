import torch
from torch import nn
import gymnasium as gym
from torch import multiprocessing as mp

from tqdm import tqdm
import optuna
import numpy as np
from collections import namedtuple

ENV_NAME        = "CartPole-v1"

env = gym.make(ENV_NAME)
N_STATES        = env.observation_space.shape[0]
N_ACTIONS       = env.action_space.n
del env

N_HIDDEN        = 64

N_WORKERS       = 1
# N_WORKERS       = mp.cpu_count()

LR              = 1e-2
DISCOUNT_FACTOR = 0.9

MAX_EPISODES    = 10000
N_STEP          = 256

DEVICE          = torch.device("cpu")
DTYPE           = torch.float64

Transition = namedtuple(typename="Transition",
                        field_names=("policy",
                                     "action",
                                     "reward",
                                     "value_current_state"))

class ReplayBuffer(object):

    def __init__(self):
        self.buffer = []

    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def reset(self):
        self.buffer = []

class SharedAdam(torch.optim.Adam):

    def __init__(self,
                 params,
                 lr=LR,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):

        super(SharedAdam).__init__(params,
                                   lr,
                                   betas,
                                   eps,
                                   weight_decay)
        
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = p.data.new().resize_as_(p.data).zero_()
                state["exp_avg_sq"] = p.data.new().resize_as_(p.data).zero_()
        
        self.share_memory()

class ActorCritic(nn.Module):

    def __init__(self):

        super().__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(N_STATES, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_HIDDEN),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(N_HIDDEN, N_ACTIONS),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(N_HIDDEN, 1)

    def forward(self, x):

        x = self.shared_fc(x)
        return self.actor(x), self.critic(x)

class A3CMaster():

    def __init__(self):

        self.global_model = ActorCritic().to(self.device)
        self.global_mode.share_memory()

        self.global_optimizer = SharedAdam(self.global_model.parameters())

        self.episode_rewards = []
        self.episode_losses = []

        self.global_reward_queue, self.global_loss_queue = mp.Queue(), mp.Queue
        self.global_step_counter = mp.Value("i", 0)
    
    def train(self):

        workers = [A3CWorker(self.global_model,
                             self.global_optimizer,
                             self.global_step_counter,
                             self.global_reward_queue,
                             self.global_loss_queue)
                   for _ in range(N_WORKERS)]

        [worker.start() for worker in workers]

        done = 0
        while done < N_WORKERS:
            r = self.global_reward_queue.get()
            l = self.global_loss_queue.get()

            if r is not None:
                self.episode_rewards.append(r)
            else:
                done += 1
            
            if l is not None:
                self.episode_losses.append(l)

        [worker.join() for worker in workers]

        return self.episode_rewards, self.episode_losses
        
class A3CWorker(mp.Process):

    def __init__(self,
                 global_model,
                 global_optimizer,
                 globa_step_counter,
                 global_reward_queue,
                 global_loss_queue):

        super(A3CWorker, self).__init__()

        self.env = gym.make(ENV_NAME)
        self.local_model = ActorCritic(n_states=self.env.observation_space.shape[0],
                                       n_actions=self.env.action_space.n).to(self.device)
        self.local_optimizer = torch.optim.Adam(self.local_model.parameters(),
                                                lr=LR)

        self.global_model = global_model
        self.global_optimizer = global_optimizer

        self.global_step_counter = globa_step_counter
        self.global_reward_queue = global_reward_queue
        self.global_loss_queue = global_loss_queue

        self.replay_buffer = ReplayBuffer()

        self.step_counter = 0

    def run(self):

        for episode in tqdm(range(MAX_EPISODES)):
            self.synchronize_models()

            state, _ = self.env.reset()
            
            episode_reward = 0
            episode_loss = 0

            done = False

            start_step = self.step_counter

            while not done:
                state = torch.tensor(state,
                                     dtype=DTYPE,
                                     device=DEVICE)
                policy, value_current_state = self.local_model.forward(state)
                action = policy.max(dim=-1).indices

                next_state, reward, terminated, truncated, _ = env.step(action.item())

                self.global_step_counter += 1
                self.step_counter += 1

                next_state = torch.tensor(next_state,
                                          dtype=DTYPE,
                                          device=DEVICE)

                episode_reward = episode_reward + reward
                reward = torch.tensor(reward,
                                      dtype=DTYPE,
                                      device=DEVICE)

                self.replay_buffer.push(policy,
                                        action,
                                        reward,
                                        value_current_state)

                state = next_state

                done = terminated or truncated

                if done or (start_step - self.step_counter == N_STEP):
                    episode_loss += self.accumulate_gradients(terminated,
                                                              next_state,
                                                              start_step)
                    self.asynchronous_update()
                
            self.global_reward_queue.put(episode_reward)
            self.global_loss_queue.put(episode_loss)
        
        self.global_reward_queue.put(None)

    def accumulate_gradients(self,
                             terminated,
                             next_state,
                             start_step):
        
        memory = Transition(*zip(*self.replay_buffer))

        update_loss = 0

        if terminated:
            R = 0
        else:
            _, value_next_state = self.local_model.forward(next_state)
            R = value_next_state
        
        for t in reversed(range(start_step, self.step_counter)):
            R *= DISCOUNT_FACTOR
            R += memory.reward[t]

            advantage = R - memory.value_current_state[t]
            log_policy = -torch.log(memory.policy[t], dim=-1)[memory.action[t]]

            actor_loss = log_policy*advantage
            critic_loss = 0.5 * torch.square(advantage)

            loss = actor_loss + critic_loss
            loss.backward()
            update_loss += loss.item()
        
        nn.utils.clip_grad_value_(self.local_model.parameters(),
                                    clip_value=500)
        
        self.replay_buffer.reset()

        return update_loss

    def asynchronous_update(self):
        self.global_optimizer.zero_grad()

        for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
            global_param.grad += local_param.grad.clone().detach()
        
        self.global_optimizer.step()

    
    def synchronize_models(self):
        self.local_optimizer.zero_grad()
        self.local_model.load_state_dict(self.global_model.state_dict())
           
if __name__ == "__main__":

    trainer = A3CMaster()
    episode_rewards, episode_losses = trainer.train()

    torch.save(episode_rewards,
               "kishorku_vveera_assignment3_part1_a3c_cartpole_rewards.pkl")
    torch.save(episode_rewards,
               "kishorku_vveera_assignment3_part1_a3c_cartpole_losses.pkl")
    torch.save(trainer.global_model.state_dict(),
               "kishorku_vveera_assignment3_part1_a3c_cartpole_checkpoint.pkl")