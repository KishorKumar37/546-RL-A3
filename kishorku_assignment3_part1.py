import torch
from torch import nn
import gymnasium as gym
from torch import multiprocessing as mp

from tqdm import tqdm
import optuna
import numpy as np
from collections import deque, namedtuple
import random

ENV_NAME        = "CartPole-v1"

env = gym.make(ENV_NAME)
N_STATES        = env.observation_space.shape[0]
N_ACTIONS       = env.action_space.n
del env

N_HIDDEN        = 64

# N_WORKERS       = 1
N_WORKERS       = mp.cpu_count()

LR              = 1e-1
DISCOUNT_FACTOR = 0.9

MAX_EPISODES    = 1200
N_STEP          = 1
GRAD_CLIP       = 500

DEVICE          = torch.device("cpu")
DTYPE           = torch.float64

class sarsd:
    policy: any
    action: int
    reward: float
    value: any

Transition = namedtuple(typename="Transition",
                        field_names=("policy",
                                     "action",
                                     "reward",
                                     "value_current_state"))

class ReplayBuffer(object):

    def __init__(self):
        self.buffer = deque([],
                            maxlen=N_STEP)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size=N_STEP):
        return random.sample(self.buffer,
                             k=batch_size)

    def __len__(self):
        return len(self.buffer)
    
    def reset(self):
        self.buffer.clear()

class SharedAdam(torch.optim.Adam):
    def _init_(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self)._init_(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

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

        self.global_model = ActorCritic().to(DEVICE).to(DTYPE)
        self.global_model.share_memory()

        self.global_optimizer = SharedAdam(self.global_model.parameters())

        self.episode_rewards = []
        self.episode_losses = []

        self.global_queue = mp.Queue()
        self.global_step_counter = mp.Value("i", 0)
    
    def train(self):

        workers = [A3CWorker(self.global_model,
                             self.global_optimizer,
                             self.global_step_counter,
                             self.global_queue)
                   for _ in range(N_WORKERS)]

        [worker.start() for worker in workers]

        done = 0
        while done < N_WORKERS:
            r, l = self.global_queue.get()

            if r is not None:
                self.episode_rewards.append(r)
                self.episode_losses.append(l)
            else:
                done += 1
                print(done)
                
        print("Exit Loop")

        # [worker.join() for worker in workers]

        return self.episode_rewards, self.episode_losses
        
class A3CWorker(mp.Process):

    def __init__(self,
                 global_model,
                 global_optimizer,
                 globa_step_counter,
                 global_queue):

        super(A3CWorker, self).__init__()

        self.env = gym.make(ENV_NAME)
        self.local_model = ActorCritic().to(DEVICE).to(DTYPE)
        self.local_optimizer = torch.optim.Adam(self.local_model.parameters(),
                                                lr=LR)

        self.global_model = global_model
        self.global_optimizer = global_optimizer

        self.global_step_counter = globa_step_counter
        self.global_queue = global_queue

        self.replay_buffer = ReplayBuffer()

        self.step_counter = 0

    def run(self):

        for _ in tqdm(range(MAX_EPISODES)):
            self.synchronize_models()

            state, _ = self.env.reset()
            state = torch.tensor(state,
                                 dtype=DTYPE,
                                 device=DEVICE)
            
            episode_reward = 0
            episode_loss = 0

            done = False

            start_step = self.step_counter

            while not done:
                
                policy, value_current_state = self.local_model.forward(state)
                action = policy.max(dim=-1).indices

                next_state, reward, terminated, truncated, _ = self.env.step(action.item())

                self.global_step_counter.value += 1

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

                if done or (self.step_counter - start_step == N_STEP):
                    episode_loss += self.accumulate_gradients(terminated,
                                                              next_state,
                                                              start_step)
                    self.asynchronous_update()
                    start_step = self.step_counter
                
            self.global_queue.put((episode_reward, episode_loss))
        
        self.global_queue.put((None, None))
        print("Worker completed and put None in queue")

    def accumulate_gradients(self,
                             terminated,
                             next_state,
                             start_step):
        
        memory = Transition(*zip(*self.replay_buffer.sample(self.replay_buffer.__len__())))

        # for data in zip(*self.replay_buffer):
        #     memory = Transition(*data)      

        update_loss = 0

        if terminated:
            R = 0
        else:
            _, value_next_state = self.local_model.forward(next_state)
            R = value_next_state
        
        for t in reversed(range(self.step_counter - start_step)):
            # print(self.step_counter, start_step, t)
            # print(len(memory.reward))
            R *= DISCOUNT_FACTOR
            R += memory.reward[t]

            advantage = R - memory.value_current_state[t]
            log_policy = -torch.log(memory.policy[t])[memory.action[t]]

            actor_loss = log_policy*advantage
            critic_loss = 0.5 * torch.square(advantage)

            loss = actor_loss + critic_loss
            loss.backward()
            update_loss += loss.item()
        
        nn.utils.clip_grad_value_(self.local_model.parameters(),
                                  clip_value=GRAD_CLIP)
        
        self.replay_buffer.reset()

        return update_loss

    def asynchronous_update(self):

        self.global_optimizer.zero_grad()

        for global_param, local_param in zip(self.global_model.parameters(), self.local_model.parameters()):
            if global_param.grad is None:
                global_param.grad = local_param.grad.clone().detach()
            else:
                global_param.grad += local_param.grad.clone().detach()
        
        self.global_optimizer.step()
    
    def synchronize_models(self):
        self.local_optimizer.zero_grad()
        self.local_model.load_state_dict(self.global_model.state_dict())

# def objective(trial):

#     global LR, DISCOUNT_FACTOR

#     LR = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
#     DISCOUNT_FACTOR = trial.suggest_uniform('discount_factor', 0.5, 0.9)

#     trainer = A3CMaster()
#     episode_rewards, episode_losses = trainer.train()

#     torch.save(episode_rewards,
#                "kishorku_vveera_assignment3_part1_a3c_cartpole_rewards.pkl")
#     torch.save(episode_losses,
#                "kishorku_vveera_assignment3_part1_a3c_cartpole_losses.pkl")
#     torch.save(trainer.global_model.state_dict(),
#                "kishorku_vveera_assignment3_part1_a3c_cartpole_checkpoint.pkl")

#     # Return the metric to optimize (e.g., average episode rewards)
#     return episode_rewards  # Replace with actual metric to optimize
           
# if __name__ == "__main__":

#     study = optuna.create_study(direction='maximize')  # or 'minimize' depending on your objective
#     study.optimize(objective)  # Adjust n_trials as needed
#     best_params = study.best_params
#     best_value = study.best_value

#     print("Best Hyperparameters:", best_params)
#     print("Best Value:", best_value)

if __name__ == "__main__":
    trainer = A3CMaster()
    episode_rewards, episode_losses = trainer.train()
    torch.save(episode_rewards,
               "kishorku_vveera_assignment3_part1_a3c_cartpole_rewards.pkl")
    torch.save(episode_losses,
               "kishorku_vveera_assignment3_part1_a3c_cartpole_losses.pkl")
    torch.save(trainer.global_model.state_dict(),
               "kishorku_vveera_assignment3_part1_a3c_cartpole_checkpoint.pkl")