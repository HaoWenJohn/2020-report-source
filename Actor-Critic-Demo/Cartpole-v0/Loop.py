from datetime import time
from typing import Tuple

import tensorflow as tf
import numpy as np
import tqdm
import gym
from Agent import ActorCritic
import os
env = gym.make("Acrobot-v1")
seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)
eps = np.finfo(np.float32).eps.item()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
max_episodes = 5000
max_steps_per_episode = 300
#reward_threshold = 195

gamma = 0.99


@tf.function
def train_step(init_state: tf.Tensor, agent: tf.keras.Model, max_steps: int, gamma: float) -> tf.Tensor:
    with tf.GradientTape() as tape:
        action_probability, values, rewards = run_episode(init_state, agent, max_steps)
        returns = get_expect_return(rewards, gamma)
        action_probability, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probability, values, returns]]
        loss = get_loss(action_probability, returns, values)

    grads = tape.gradient(loss, agent.trainable_variables)
    optimizer.apply_gradients(zip(grads, agent.trainable_variables))
    return tf.math.reduce_sum(rewards)


def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    state, reward, done, _ = env.step(action)
    return state.astype(np.float32), np.array(reward, np.int32), np.array(done, np.int32)


def tf_env_step(action: np.ndarray):
    return tf.numpy_function(env_step, [action], [tf.float32, tf.int32, tf.int32])


def run_episode(init_state: tf.Tensor, agent: tf.keras.Model, max_steps: int) \
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    action_probability = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    state_shape = init_state.shape
    state = init_state

    for i in tf.range(max_steps):
        # choose an action
        state = tf.expand_dims(state, 0)
        actions_probability, value = agent(state)
        action = tf.random.categorical(actions_probability, 1)[0, 0]
        actions_probability_t = tf.nn.softmax(actions_probability)
        # store value
        values=values.write(i, tf.squeeze(value))

        # store action probability
        action_probability=action_probability.write(i, actions_probability_t[0, action])

        # run a step
        state, reward, done = tf_env_step(action)
        state.set_shape(state_shape)
        # store reward
        rewards=rewards.write(i, reward)

        if tf.cast(done, tf.bool):
            break

    action_probability = action_probability.stack()
    values = values.stack()
    rewards = rewards.stack()
    return action_probability, values, rewards


def get_expect_return(rewards: tf.Tensor, gamma: tf.float32, standardize=True) -> tf.Tensor:
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(tf.float32, size=n)

    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    discount_sum = tf.constant(0.0)
    discount_sum_shape = discount_sum.shape

    for i in tf.range(n):
        reward = rewards[i]
        discount_sum = reward + gamma * discount_sum
        discount_sum.set_shape(discount_sum_shape)
        returns = returns.write(i, discount_sum)
    returns = returns.stack()[::-1]

    if standardize:
        returns = (returns - tf.reduce_mean(returns)) / (tf.math.reduce_std(returns) + eps)

    return returns


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def get_loss(action_probability: tf.Tensor, returns: tf.Tensor, values: tf.Tensor) -> tf.Tensor:

    advantage = returns - values

    actor_loss = - tf.reduce_sum(tf.math.log(action_probability) * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss

def train(model):
    running_reward = 0
    for i in range(max_episodes):
        first_state = tf.constant(env.reset(), dtype=tf.float32)
        episode_reward = train_step(first_state, model, max_steps_per_episode, gamma)
        running_reward = int(episode_reward) * 0.01 + running_reward * .99
        if i%100 == 0:
            print('\rcurrent_episode:{},episode_reward:{}, running_reward:{}'.format(
                i, episode_reward, running_reward), end='')

if __name__ == '__main__':

    model = ActorCritic(env.action_space.n, [128])

    if not os.path.exists("./weights"):
        train(model)
        model.save_weights("./weights/w")
    else:
        model.load_weights("./weights/w")
    state = tf.constant(env.reset(), dtype=tf.float32)

    for i in range(1, 1000):
        state = tf.expand_dims(state, 0)
        action_probs, _ = model(state)
        action = np.argmax(np.squeeze(action_probs))

        state, _, done, _ = env.step(action)
        state = tf.constant(state, dtype=tf.float32)
        env.render()
        if done:
           break