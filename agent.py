from constants import *
    

class Agent:
    def __init__(self) -> None:
        self.buffer = deque(maxlen=int(1e5)) # buffer
        self.model = self._build_model()
        self.target_model = self._build_model()
        print("Agent is initialized")

    def _build_model(self):
        S1t = Input(shape=(WINDOW_WIDTH, 2*NUM_USERS+NUM_CHANNELS))
        x = LSTM(units=NUM_UNITS)(S1t)
        x = Dense(units=NUM_USERS, activation='tanh')(x)

        x = Reshape(target_shape=(1, NUM_USERS))(x)
        x = tf.cast(x, dtype=tf.float32)
        Ht = Input(shape=(1, NUM_USERS))
        x = concatenate([x, Ht], axis=2)
        
        x = LSTM(units=NUM_UNITS)(x)
        x = Dense(units=ACTION_SPACE_SIZE, activation='tanh')(x)
        model = Model(inputs=[S1t, Ht], outputs=x)
        
        return model
    
    def predict_batteries(self):
        return backend.function(inputs=self.model.layers[0].input, outputs=self.model.layers[2].output)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def replay(self):
        optimizer = optimizers.SGD(learning_rate=LEARNING_RATE)
        loss_fn = losses.MeanSquaredError()

        minibatch = random.sample(self.buffer, BATCH_SIZE)
        states, actions, rewards, next_states = zip(*minibatch)


        curr_S1 = np.array([states[i][0] for i in range(BATCH_SIZE)])
        curr_H = np.array([states[i][1] for i in range(BATCH_SIZE)])
        next_S1 = np.array([next_states[i][0] for i in range(BATCH_SIZE)])
        next_H = np.array([next_states[i][1] for i in range(BATCH_SIZE)])

        future_rewards = self.target_model.predict([next_S1, next_H], verbose=0)
        updated_q_values = rewards + GAMMA*tf.reduce_max(future_rewards, axis=1)
        masks = tf.one_hot(actions, ACTION_SPACE_SIZE)

        with tf.GradientTape() as tape:
            q_values = self.model([curr_S1, curr_H])

            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss_value = loss_fn(updated_q_values, q_action)

        grads = tape.gradient(loss_value, self.model.trainable_weights)

        optimizer.apply_gradients(zip(grads, self.model.trainable_weights))


        
