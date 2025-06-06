import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# ----- Game Settings -----
WIDTH, HEIGHT = 400, 400
CELL_SIZE = 20
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE

# ----- DQN Network -----
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# ----- Snake Game -----
class SnakeGame:
    def __init__(self, render=True, base_obstacles=2, speed=10):
        self.render = render
        self.base_obstacles = base_obstacles
        self.dynamic_speed = speed
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Training Snake RL")
            self.clock = pygame.time.Clock()

        self.food_types = {
            'normal': {'color': (255, 0, 0), 'reward': 5},
            'gold': {'color': (255, 215, 0), 'reward': 10},
            'green': {'color': (0, 255, 0), 'reward': 7}
        }

        self.reset()

    def reset(self, episode=0):
        self.snake = [(5, 5)]
        self.direction = (1, 0)
        self.num_obstacles = self.base_obstacles + (episode // 25)
        self.speed = self.dynamic_speed = 15
        self.obstacles = self._place_obstacles(self.num_obstacles)
        self.food_items = self._place_food_items(count=3)
        self.score = 0
        self.done = False
        self.lives = 1
        self.loop_counter = 0
        return self._get_state()

    def _place_obstacles(self, count):
        obs = set()
        while len(obs) < count:
            pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if pos not in self.snake:
                obs.add(pos)
        return list(obs)

    def _place_food_items(self, count=3):
        food_list = []
        placed = set()
        while len(food_list) < count:
            pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if pos not in self.snake and pos not in self.obstacles and pos not in placed:
                kind = random.choices(['normal', 'gold', 'green'], weights=[0.5, 0.3, 0.2])[0]
                food_list.append((pos, kind))
                placed.add(pos)
        return food_list

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}

        self.loop_counter += 1
        reward = -0.05  # stronger penalty for wasting time
        self._move(action)
        head = self.snake[0]

        if (head[0] < 0 or head[0] >= GRID_WIDTH or
            head[1] < 0 or head[1] >= GRID_HEIGHT or
            head in self.snake[1:] or head in self.obstacles):
            self.done = True
            return self._get_state(), -10, True, {}

        for i, (pos, kind) in enumerate(self.food_items):
            if head == pos:
                props = self.food_types.get(kind, {'reward': 1})
                self.score += props['reward']
                reward += props['reward']
                #self.speed += props['speed']
                del self.food_items[i]
                self.food_items.extend(self._place_food_items(1))
                break
        else:
            self.snake.pop()

        # kill if looping too long
        if self.loop_counter > 200:
            self.done = True
            reward = -5

        return self._get_state(), reward, self.done, {}

    def _move(self, action):
        x, y = self.direction
        if action == 1:
            self.direction = (-y, x)
        elif action == 2:
            self.direction = (y, -x)
        new_head = (self.snake[0][0] + self.direction[0],
                    self.snake[0][1] + self.direction[1])
        self.snake.insert(0, new_head)

    def _get_state(self):
        head_x, head_y = self.snake[0]

        best_food = min(self.food_items, key=lambda f: abs(f[0][0]-head_x) + abs(f[0][1]-head_y))[0]
        food_dx = best_food[0] - head_x
        food_dy = best_food[1] - head_y

        distance_to_wall_x = min(head_x, GRID_WIDTH - head_x - 1) / GRID_WIDTH
        distance_to_wall_y = min(head_y, GRID_HEIGHT - head_y - 1) / GRID_HEIGHT

        is_food_ahead = int((food_dx * self.direction[0] > 0) or (food_dy * self.direction[1] > 0))
        is_food_left = int((food_dx * -self.direction[1] > 0) or (food_dy * self.direction[0] > 0))
        is_food_right = int((food_dx * self.direction[1] > 0) or (food_dy * -self.direction[0] > 0))

        return np.array([
            head_x / GRID_WIDTH,
            head_y / GRID_HEIGHT,
            food_dx / GRID_WIDTH,
            food_dy / GRID_HEIGHT,
            self.direction[0],
            self.direction[1],
            distance_to_wall_x,
            distance_to_wall_y,
            is_food_ahead,
            is_food_left,
            is_food_right
        ], dtype=np.float32)

    def render_game(self):
        if not self.render:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((0, 0, 0))

        for segment in self.snake:
            pygame.draw.rect(self.screen, (255, 255, 255),
                             (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for pos, kind in self.food_items:
            color = self.food_types[kind]['color']
            pygame.draw.rect(self.screen, color,
                             (pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (100, 100, 100),
                             (obs[0] * CELL_SIZE, obs[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        pygame.display.flip()
        self.clock.tick(self.speed)

# ----- DQN Agent -----
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.target = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.batch_size = 64
        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.01
        self.update_target()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            return torch.argmax(self.model(state_tensor)).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        max_next_q = self.target(next_states).max(1)[0].unsqueeze(1)
        expected_q = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path="snake_model.pth"):
        torch.save(self.model.state_dict(), path)

# ----- Training Loop -----
if __name__ == "__main__":
    episodes = 500
    env = SnakeGame(render=True)
    agent = DQNAgent(state_size=11, action_size=3)  # state size updated
    scores = []

    for ep in range(episodes):
        state = env.reset(episode=ep)
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward
            env.render_game()
            if done:
                break

        agent.update_target()
        scores.append(env.score)
        print(f"Episode {ep+1:>3} | Score: {env.score:<3} | Obstacles: {env.num_obstacles:<2} | Speed: {env.speed:<2} | Epsilon: {agent.epsilon:.3f}")

    agent.save("snake_model_2.pth")
    print("✅ Training complete. Model saved as 'snake_model.pth'.")

    # Plot scores
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Snake Training Progress")
    plt.grid(True)
    plt.show()


