
import torch
from torch import nn
import gymnasium as gym
from itertools import count
from torch import multiprocessing as mp
import math
import random
from tqdm import tqdm

# mp.set_start_method('fork')
# torch.autograd.set_detect_anomaly(True)

class ActorCritic(nn.Module):

    def __init__(self,
                 n_states,
                 n_actions):
        super().__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared_fc(x)
        return self.actor(x), self.critic(x)

class A3CAgent():
    def __init__(self,
                 env_name,
                 global_model,
                 num_workers,
                 device,
                 lr,
                 max_episodes,
                 discount_factor):
        self.env_name = env_name
        self.global_model = global_model
        self.num_workers = num_workers
        
        self.lr = lr
        self.discount_factor = discount_factor
        self.max_episodes = max_episodes
        self.device = device

        # self.episode_rewards = []
        # self.episode_steps = []
        # self.episode_losses = []

        print("Finished initializing of A3C agent")
    
    def train(self):
        envs = [gym.make(self.env_name) for _ in range(self.num_workers)]
        worker_models = [ActorCritic(n_states=envs[0].observation_space.shape[0],
                                     n_actions=envs[0].action_space.n).to(self.device)
                                     for _ in range(self.num_workers)]
        
        for i, (env, worker_model) in enumerate(zip(envs, worker_models)):
            #  print("Entered multiprocess loop")
            worker = mp.Process(target=self.train_worker,
                                args=(env,
                                      self.global_model,
                                      worker_model,
                                      self.lr,
                                      self.max_episodes,
                                      i))
            worker.start()
            #  print("Executed worker")
             

    def train_worker(self,
                     env,
                     global_model,
                     worker_model,
                     lr,
                     max_episodes,
                     worker_name):
        # print(f"Worker {worker_name} initiated")
        ''' Kishorkumar Devasenapathy - 04-15-2024
        # This optimizer is only in the scope of this worker but has a shared copy of the global model's
        # parameters to perform gradient descent and update
        # torch.multiprocessing takes care of synchronizing this step across the workers
        '''
        optimizer = torch.optim.Adam(global_model.parameters(),
                                     lr=lr)
        # print(f"Worker {worker_name} initialized optimizer for global model")
        # actor_loss_fn = nn.LogSoftmax()
        # critic_loss_fn = nn.MSELoss()

        state, _ = env.reset()
        state = torch.tensor(state,
                             dtype=torch.float32,
                             device=self.device).unsqueeze(0)
        # print(f"Worker {worker_name} interacted first time with environment")
        
        episode_rewards = []
        episode_steps = []
        episode_losses = []

        for episode in tqdm(range(max_episodes)):
            # print(f"Worker {worker_name} started episode {episode}")
            # optimizer.zero_grad()
            worker_model.load_state_dict(global_model.state_dict()) # synchronize worker's model with that of global model
            # print(f"Worker {worker_name} loaded global model's parameters")

            state, _ = env.reset()
            
            episode_reward = 0
            episode_loss = 0
            # print(state.shape)

            for step in count():
                # print("Started first step")
                state_tensor = torch.tensor(state,
                                 dtype=torch.float32,
                                 device=self.device)
                # print("Trying model forward")
                action_probs, current_state_value = worker_model.forward(state_tensor)
                # print("Trying model Worked")
                # print(action_probs.shape)
                action = torch.max(action_probs,
                                   dim=-1).indices
                # print(action.shape)

                next_state, reward, terminated, truncated, _ = env.step(action.item())
                next_state_tensor = torch.tensor(next_state,
                                          dtype=torch.float32,
                                          device=self.device)
                # print("Trying model forward again")
                _, next_state_value = worker_model.forward(next_state_tensor)
                # print("Trying model forward again worked")

                episode_reward = episode_reward + reward

                # reward = torch.tensor([reward],
                #                       dtype=torch.float32,
                #                       device=self.device)
                # print(reward.shape)

                # reward *= 5

                if terminated:
                    target_current_state_value = reward
                else:
                    target_current_state_value = reward + self.discount_factor*next_state_value.item()

                # actor_loss = -torch.log_softmax(action_probs,
                #                                 dim=-1)
                # actor_loss = actor_loss_fn(action_probs)
                # critic_loss = critic_loss_fn(target_current_state_value, current_state_value.item())
                advantage = target_current_state_value - current_state_value

                actor_loss = -torch.log_softmax(action_probs, dim=-1)[action] * advantage.clone()
                critic_loss = 0.5 * advantage.pow(2)

                episode_loss = torch.add(episode_loss, (actor_loss + critic_loss))
                
                state = next_state

                # optimizer.zero_grad()
                # print("Trying loss calc")
                # episode_loss.backward(retain_graph=True)
                # print("Loss calc worked")

                # print("STarted copying params")
                # for global_param, worker_param in zip(global_model.parameters(), worker_model.parameters()):
                #     if global_param.grad is None:
                #         global_param.grad = worker_param.grad.clone().detach()
                #     else:
                #         global_param.grad += global_param.grad + worker_param.grad.clone().detach()
                # optimizer.step()
                # print("Finished copying params")

                if terminated or truncated:
                    # print(f"Worker {worker_name} completed an episode")
                    episode_rewards.append(episode_reward)
                    episode_steps.append(step+1)
                    episode_losses.append(episode_loss.item())
                    break
            optimizer.zero_grad()
            # print("Trying loss calc")
            episode_loss.backward(retain_graph=True)
            # print("Loss calc worked")

            # print("STarted copying params")
            for global_param, worker_param in zip(global_model.parameters(), worker_model.parameters()):
                if global_param.grad is None:
                    global_param.grad = worker_param.grad.clone().detach()
                else:
                    global_param.grad += global_param.grad + worker_param.grad.clone().detach()
            optimizer.step()
            nn.utils.clip_grad_value_(global_model.parameters(),
                                      clip_value=500)
            # print("Finished copying params")
        # print(f"Worker {worker_name} completed episode {episode}")

        if worker_name == 0:
            #  self.episode_rewards = (episode_rewards)
            #  self.episode_steps = (episode_steps)
            #  self.episode_losses = (episode_losses)
    
            torch.save(episode_rewards, f"A3C_cartpole_episode_rewards.pkl")
            torch.save(episode_steps, f"A3C_cartpole_episode_steps.pkl")
            torch.save(episode_losses, f"A3C_cartpole_episode_losses.pkl")

           
if __name__ == "__main__":
    ENV_NAME        = "CartPole-v1"
    NUM_WORKERS     = 8

    LR              = 1e-4
    MAX_EPISODES    = 2000

    DISCOUNT_FACTOR = 0.99


    DEVICE = "cpu"

    env = gym.make(ENV_NAME)
    global_model = ActorCritic(n_states=env.observation_space.shape[0],
                               n_actions=env.action_space.n).to(DEVICE)
    global_model.share_memory()

    trainer = A3CAgent(env_name=ENV_NAME,
                    global_model=global_model,
                    num_workers=NUM_WORKERS,
                    device=DEVICE,
                    lr=LR,
                    max_episodes=MAX_EPISODES,
                    discount_factor=DISCOUNT_FACTOR)

    trainer.train()
