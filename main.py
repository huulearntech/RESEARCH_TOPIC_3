from agent import *
from constants import *
from env import *

import matplotlib.pyplot as plt
  
# EPSILON_FRAMES = 50000
update_after_actions = 5
def main():
    #initialize environment:
    env = Env()

    agent = Agent()
    total_reward = 0
    moving_average = []
    # initialize total number of episodes
    for episode in range(1, TOTAL_EPISODES + 1):
        epsilon = EPSILON_START
        episode_reward = 0

        # initial observation
        S_t = env.reset()
        S1_t = S_t[0]
        H_t = S_t[1]

        for t in range(1, T+ 1):

            #perform 1st layer
            S1_tensor = tf.convert_to_tensor(S1_t)
            S1_tensor = tf.expand_dims(S1_tensor, axis=0)
            H_tensor = tf.convert_to_tensor(H_t)
            H_tensor = tf.expand_dims(H_tensor, axis=0)
            if np.random.random() < epsilon:
                action_id = np.random.choice(ACTION_SPACE_SIZE)
            else:
                Q_values = agent.model(inputs=[S1_tensor, H_tensor], training=False)
                action_id = np.argmax(Q_values[0])
            
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

            predicted_batteries = agent.predict_batteries()(S1_tensor)[0]
            #execute At, observe reward            
            next_state, reward = env.step(action_id, predicted_batteries)

            #store transition (S_t, A_t, R_t, S_t+1) in buffer
            episode_reward += reward
            agent.buffer.append((S_t, action_id, reward, next_state))

            S_t = next_state
            
            if t%update_after_actions == 0 and len(agent.buffer) > BATCH_SIZE:
                agent.replay()

            

            if t % UPDATE_RATE == 0:
                agent.update_target_model()
        print("Total reward: {:.2f} of episode {}".format(episode_reward, episode))
        total_reward += episode_reward
        moving_average.append(total_reward/episode)

    moving_average = np.array(moving_average)
    np.savetxt('moving_avg.txt', moving_average, fmt='%f')

    moving_average = np.loadtxt('moving_avg.txt', dtype=float)

    plt.plot(moving_average)
    plt.xlabel('Episodes')
    plt.ylabel('Moving Avg')
    plt.title('moving avg')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
