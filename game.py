import pygame
import random
import sys
import torch
import torch.optim as optim
from neural_network import DQN
import torch.nn as nn


BATCH_SIZE = 100


class Game:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.clock = pygame.time.Clock()

        pygame.init()
        pygame.display.set_caption("Управление квадратом")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.font = pygame.font.Font(None, 36)

        self.model = DQN(input_size=4, output_size=4)
        self.target_model = DQN(input_size=4, output_size=4)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.99

        self.game_over = False
        self.score = 0
        try:
            with open('score.txt', 'r') as file:
                best_score = int(file.read().strip())
        except FileNotFoundError:
            best_score = 0

        self.best_score = best_score

        self.square_size = 50
        self.boundary_size = 50
        self.speed = 10

        self.square_x = self.screen_width // 2 - self.square_size // 2
        self.square_y = self.screen_height // 2 - self.square_size // 2
        self.food_x = random.randint(self.boundary_size, self.screen_width - self.boundary_size - self.square_size)
        self.food_y = random.randint(self.boundary_size, self.screen_height - self.boundary_size - self.square_size)

    def train_step(self, transitions):
        states, actions, next_states, rewards = zip(*transitions)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Вычисляем Q-значения для текущих и следующих состояний
        q_values = self.model(states)
        q_values_next = self.target_model(next_states)

        # Получаем Q-значения для выбранных действий
        q_values = q_values.gather(1, actions.view(-1, 1)).squeeze()

        # Вычисляем целевые Q-значения по формуле DQN
        target_q_values = rewards + self.gamma * q_values_next.max(1)[0]

        # Вычисляем функцию потерь
        loss = nn.MSELoss()(q_values, target_q_values.detach())

        # Обновляем параметры модели
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return q_values.argmax().item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def draw_square(self):
        pygame.draw.rect(self.screen, (255, 0, 0), (self.square_x, self.square_y, self.square_size, self.square_size))

    def draw_food(self):
        pygame.draw.circle(self.screen, (0, 255, 0), (self.food_x, self.food_y), 10)

    def draw_score(self):
        myFont = pygame.font.SysFont("Times New Roman", 18)
        scoreDisplay = myFont.render(str(self.score), 1, (0, 0, 0))
        self.screen.blit(scoreDisplay, (520, 30))

    def run(self):
        episode_transitions = []
        while not self.game_over:
            self.screen.fill((255, 255, 255))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True

            state = [self.square_x, self.square_y, self.food_x, self.food_y]
            action = self.get_action(state)

            if action == 0 and self.square_x > self.boundary_size:
                self.square_x -= self.speed
            elif action == 1 and self.square_x < self.screen_width - self.boundary_size - self.square_size:
                self.square_x += self.speed
            elif action == 2 and self.square_y > self.boundary_size:
                self.square_y -= self.speed
            elif action == 3 and self.square_y < self.screen_height - self.boundary_size - self.square_size:
                self.square_y += self.speed

            self.draw_square()
            self.draw_food()
            self.draw_score()

            if self.square_x < self.food_x < self.square_x + self.square_size and self.square_y < self.food_y < self.square_y + self.square_size:
                self.food_x = random.randint(self.boundary_size, self.screen_width - self.boundary_size - self.square_size)
                self.food_y = random.randint(self.boundary_size, self.screen_height - self.boundary_size - self.square_size)
                self.score += 1

            pygame.display.flip()
            self.clock.tick(30)

            if len(episode_transitions) >= BATCH_SIZE:
                # Проводим один шаг обучения на основе собранного опыта
                self.train_step(episode_transitions)
                episode_transitions = []

        if self.score > self.best_score:
            print('Новый рекорд')
            torch.save(self.model.state_dict(), 'trained_model.pth')
            with open('score.txt', 'w') as file:
                file.write(str(self.score))

        pygame.quit()
        sys.exit()

game = Game(800, 600)
game.run()