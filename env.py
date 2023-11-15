from constants import *

def generate_actions(n: int, k: int) -> list:
    actions = [0] * n
    comb = list(combinations(range(n), k))
    for c in comb:
        for i in c:
            actions[i] = 1
        yield actions
        actions = [0] * n


class Env:
    ACTION_SPACE = list(generate_actions(NUM_USERS, NUM_CHANNELS)) # actions that the AP (agent) can take
    BS_LOCATION = np.array([RANGE/2, RANGE/2])

    def __init__(self):
        self.user_locations = np.random.randint(low=0, high=RANGE, size=(NUM_USERS, 2))
        self.energy_rate = np.random.randint(low=1, high=3, size=NUM_USERS) #energy rate, poisson lambda
        self.channels = np.zeros(shape=(1, NUM_USERS))
        print('Environment is initialized')
        self.state = self.reset()

    def reset(self):
        for i in range(NUM_USERS):
            while (self.user_locations[i] == Env.BS_LOCATION).all():
                self.user_locations[i] = np.random.randint(low=0, high=RANGE, size=(1, 2))
        
        #chạy file env 1 lần sinh ra file, đọc file ra từ main

        self.user_batteries = np.full(shape=NUM_USERS, fill_value=BATTERY_CAPACITY)
        self.update_channel_gains()

        #state = {(X_t, M_t, G_t)_*window, H_t}

        S1t = deque(maxlen=WINDOW_WIDTH)
        for i in range(WINDOW_WIDTH):
            S1t.append(np.random.rand(2*NUM_USERS+NUM_CHANNELS))
        state = [S1t, self.channels]

        return state


    def update_user_locations(self) -> None:
        for i in range(NUM_USERS):
            while True:
                walk_step = random.choice([[1, 0], [0, 1], [-1, 0], [0, -1]])
                if(self.user_locations[i][0] + walk_step[0] in range(RANGE) and self.user_locations[i][1] + walk_step[1] in range(RANGE)):
                    self.user_locations[i] += np.array(walk_step)
                    break


    def update_channel_gains(self) -> None:
        for i in range(NUM_USERS):
            path_loss = 128.1 + 37.6 * math.log10(math.dist(self.user_locations[i], Env.BS_LOCATION)/1000) #BS at the center of area
            self.channels[0][i] = math.pow(10, -path_loss / 10) * math.pow(np.random.rayleigh(), 2) / 2


    def step(self, action_id: int, predicted_batteries: np.ndarray) -> tuple:
        reward = 0
        action = np.array(Env.ACTION_SPACE[action_id])
        B_it = []
        #broadcast action to all users
        for i in range(NUM_USERS):
            if action[i] == 1:
                if self.user_batteries[i] >= TRANSMIT_POWER:
                    B_it.append(self.user_batteries[i])
                    self.user_batteries[i] -= TRANSMIT_POWER
                    reward += BANDWIDTH * math.log2(1 + pow(10, TRANSMIT_POWER/10) * self.channels[0][i] / NOISE)
                else:
                    B_it.append(0)
                    action[i] = 0


            #collect energy from environment
            self.user_batteries[i] = min(BATTERY_CAPACITY, self.user_batteries[i] + np.random.poisson(self.energy_rate[i]))

        #compute reward
        P_loss = np.linalg.norm(action*(predicted_batteries - self.user_batteries))
        # print(f"p_loss: {P_loss}")
        reward -= BETA*P_loss


        # self.update_user_locations()
        self.update_channel_gains()
        

        self.state[0].append(np.hstack((action, predicted_batteries, B_it)))
        self.state[1] = self.channels


        return self.state, reward