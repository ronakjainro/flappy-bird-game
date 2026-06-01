import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory

import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random

# Device selection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:

    def __init__(self, param_set):
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_params = yaml.safe_load(f)
            params = all_params[param_set]

        self.env_id = params["env_id"]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(
            RUNS_DIR,
            f"{self.param_set}.log"
        )

        self.MODEL_FILE = os.path.join(
            RUNS_DIR,
            f"{self.param_set}.pt"
        )

    def run(self, is_training=True, render=False):

        env = gym.make(
            self.env_id,
            render_mode="human" if render else None
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(
            num_states,
            num_actions
        ).to(device)

        if is_training:

            memory = ReplayMemory(
                self.replay_memory_size
            )

            epsilon = self.epsilon_init

            target_dqn = DQN(
                num_states,
                num_actions
            ).to(device)

            target_dqn.load_state_dict(
                policy_dqn.state_dict()
            )

            self.optimizer = optim.Adam(
                policy_dqn.parameters(),
                lr=self.alpha
            )

            steps = 0
            best_reward = float("-inf")

        else:

            if not os.path.exists(self.MODEL_FILE):
                raise FileNotFoundError(
                    f"Model file not found: {self.MODEL_FILE}"
                )

            policy_dqn.load_state_dict(
                torch.load(
                    self.MODEL_FILE,
                    map_location=device
                )
            )

            policy_dqn.eval()

        try:
            for episode in itertools.count():

                state, _ = env.reset()

                state = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=device
                )

                episode_reward = 0
                done = False

                while not done and episode_reward < self.reward_threshold:

                    # Epsilon-greedy action
                    if is_training and random.random() < epsilon:

                        action = torch.tensor(
                            env.action_space.sample(),
                            dtype=torch.long,
                            device=device
                        )

                    else:

                        with torch.no_grad():

                            action = (
                                policy_dqn(
                                    state.unsqueeze(0)
                                )
                                .squeeze()
                                .argmax()
                            )

                    next_state, reward, terminated, truncated, _ = env.step(
                        action.item()
                    )

                    done = terminated or truncated

                    episode_reward += reward

                    reward_tensor = torch.tensor(
                        reward,
                        dtype=torch.float32,
                        device=device
                    )

                    next_state_tensor = torch.tensor(
                        next_state,
                        dtype=torch.float32,
                        device=device
                    )

                    if is_training:

                        memory.append(
                            (
                                state,
                                action,
                                next_state_tensor,
                                reward_tensor,
                                done,
                            )
                        )

                        steps += 1

                    state = next_state_tensor

                # Logging
                if is_training:
                    print(
                        f"Episode {episode+1} | "
                        f"Reward={episode_reward:.2f} | "
                        f"Epsilon={epsilon:.4f}"
                    )
                else:
                    print(
                        f"Episode {episode+1} | "
                        f"Reward={episode_reward:.2f}"
                    )

                if is_training:

                    epsilon = max(
                        epsilon * self.epsilon_decay,
                        self.epsilon_min
                    )

                    if episode_reward > best_reward:

                        best_reward = episode_reward

                        with open(self.LOG_FILE, "a") as f:
                            f.write(
                                f"Episode {episode+1}: "
                                f"Reward={episode_reward}\n"
                            )

                        torch.save(
                            policy_dqn.state_dict(),
                            self.MODEL_FILE
                        )

                    if len(memory) >= self.mini_batch_size:

                        mini_batch = memory.sample(
                            self.mini_batch_size
                        )

                        self.optimize(
                            mini_batch,
                            policy_dqn,
                            target_dqn
                        )

                        if steps >= self.network_sync_rate:

                            target_dqn.load_state_dict(
                                policy_dqn.state_dict()
                            )

                            steps = 0

        finally:
            env.close()

    def optimize(
        self,
        mini_batch,
        policy_dqn,
        target_dqn
    ):

        states, actions, next_states, rewards, dones = zip(
            *mini_batch
        )

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=device
        )

        with torch.no_grad():

            next_q = target_dqn(
                next_states
            ).max(dim=1)[0]

            target_q = rewards + (
                1 - dones
            ) * self.gamma * next_q

        current_q = policy_dqn(
            states
        ).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze()

        loss = self.loss_fn(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train or test DQN Flappy Bird"
    )

    parser.add_argument(
        "hyperparameters",
        help="Parameter set name from YAML"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Training mode"
    )

    args = parser.parse_args()

    agent = Agent(
        param_set=args.hyperparameters
    )

    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(
            is_training=False,
            render=True
        )