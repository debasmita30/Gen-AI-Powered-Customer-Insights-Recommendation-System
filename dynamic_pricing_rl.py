import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import time

# --- Environment Setup ---
class DynamicPricingEnv(gym.Env):
    def __init__(self):
        super(DynamicPricingEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=50, high=500, shape=(1,), dtype=np.float32)  # Price range
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)  # demand, competitor, inventory
        self.reset()

    def reset(self, seed=None, options=None):
        self.state = np.array([
            np.random.rand(),  # demand
            np.random.rand(),  # competitor price factor
            np.random.rand()   # inventory
        ])
        return self.state, {}

    def step(self, action):
        price = float(action[0])
        demand_factor, competitor_price, inventory = self.state
        demand = max(0, (1 - price / 500) + (competitor_price * 0.3))
        sales = demand * 100
        profit = (price - 100) * sales - (inventory * 50)
        reward = profit / 10000.0  # scaled reward
        self.state = np.clip(np.random.rand(3), 0, 1)
        terminated = False
        truncated = False
        return self.state, reward, terminated, truncated, {}

# --- PPO Agent ---
class PPOAgent:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = 0.99
        self.actor_lr = 0.0003
        self.critic_lr = 0.001
        self._build_models()

    def _build_models(self):
        # Actor network
        self.actor = tf.keras.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_dim, activation='sigmoid')
        ])
        # Critic network
        self.critic = tf.keras.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        action = self.actor(state)[0].numpy()
        return action * self.action_bound

    def train(self, states, actions, rewards):
        discounted_rewards = []
        cumulative = 0
        for r in rewards[::-1]:
            cumulative = r + self.gamma * cumulative
            discounted_rewards.insert(0, cumulative)
        discounted_rewards = np.array(discounted_rewards)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            values = tf.squeeze(self.critic(states))
            advantages = discounted_rewards - values
            probs = tf.squeeze(self.actor(states))
            actor_loss = -tf.reduce_mean(probs * advantages)
            critic_loss = tf.reduce_mean(tf.square(advantages))

        actor_grads = tape1.gradient(actor_loss, self.actor.trainable_variables)
        critic_grads = tape2.gradient(critic_loss, self.critic.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

# --- Training Loop ---
def train_agent(episodes=200):
    env = DynamicPricingEnv()
    agent = PPOAgent(state_dim=3, action_dim=1, action_bound=500)
    rewards_history = []

    os.makedirs("models", exist_ok=True)

    print(f"--- Starting Training for {episodes} episodes ---")
    start_time = time.time()

    for episode in range(episodes):
        state, _ = env.reset()
        episode_rewards = []
        for step in range(100):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.train(np.array([state]), np.array([action]), [reward])
            episode_rewards.append(reward)
            state = next_state
            if terminated or truncated:
                break

        total_reward = np.sum(episode_rewards)
        rewards_history.append(total_reward)

        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {episode+1}/{episodes} | Avg Profit: {total_reward*10000:.2f} | Time: {elapsed:.1f}s")

    # --- Save model ---
    model_json = agent.actor.to_json()
    with open("models/pricing_model.json", "w") as json_file:
        json_file.write(model_json)
    agent.actor.save_weights("models/pricing_model.weights.h5")
    print("✅ Model saved to 'models/' folder")

    # --- Plot rewards ---
    plt.plot(rewards_history)
    plt.title("Training Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Total Profit")
    plt.savefig("training_rewards.png")
    print("Saved training graph → training_rewards.png")
    print("Training complete.")

# --- Run ---
if __name__ == "__main__":
    train_agent(episodes=200)
