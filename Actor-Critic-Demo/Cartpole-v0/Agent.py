from typing import Tuple

import tensorflow as tf


class ActorCritic(tf.keras.Model):
    def __init__(self, num_actions: int, num_hidden_units: []):
        super().__init__()
        self.common = []
        for h_u in num_hidden_units:
            self.common.append(tf.keras.layers.Dense(h_u, activation=tf.keras.activations.relu))
        self.actor = tf.keras.layers.Dense(num_actions)
        self.critic = tf.keras.layers.Dense(1)

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor]:
        for common_layer in self.common:
            inputs = common_layer(inputs)
        return self.actor(inputs), self.critic(inputs)
