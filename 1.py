import numpy as np
import gym
import random
from numpy import linalg as LA
import matplotlib.pyplot as plt



def main(learning_rate , discount_rate ):
    differences = []


    # create Taxi environment
    env = gym.make('Taxi-v3', render_mode='ansi')

    # initialize q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    # learning_rate = 0.9
    # discount_rate = 0.8
    epsilon = 1.0
    decay_rate= 0.005

    # training variables
    num_episodes = 1000
    max_steps = 99 # per episode
    mat_old = np.copy(qtable)
    # training
    episode = -1
    # for episode in range(num_episodes):
    
    while True:
        episode +=1
        
        # reset the environment
        state, _= env.reset()
        done = False

        mat_new = qtable
        
        for s in range(max_steps):

            # exploration-exploitation tradeoff
            if random.uniform(0,1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state,:])

            # take action and observe reward
            new_state, reward, done, info ,_ = env.step(action)

            # Q-learning algorithm
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])

            # Update to our new state
            state = new_state

            # if done, finish episode
            if done == True:
                break

        # Decrease epsilon
        epsilon = np.exp(-decay_rate*episode)
        differences.append(LA.norm(mat_new - mat_old))
        # differences.append(np.mean(np.array(mat_new))-np.mean(np.array(mat_old)))
        mat_old = np.copy(mat_new)
        if differences[-1] < 10e-6:
            
            # return len(differences)
            break



    print(f"Training completed over {num_episodes} episodes")
    # input("Press Enter to watch trained agent...")
    print("diff---- "  ,differences)
    # watch trained agent
    state, _ = env.reset()
    done = False
    rewards = 0

    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s+1))

        action = np.argmax(qtable[state,:])
        print(env.step(action))
        new_state, reward, done, info ,h  = env.step(action)
        
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state

        if done == True:
            break



    env.close()
    return len(differences)

# if __name__ == "__main__":



x_val = np.arange(0.05, 1.05, 0.05).tolist()

all_y_vals = []
for discount_rate in range (3 , 10 , 2):
    y_val = []
    for i in x_val:
        to_get_avg = []
        for k in range(20):
            to_get_avg.append(main( i , discount_rate/10.0))

        y_val.append(np.mean(np.array(to_get_avg)))
    all_y_vals.append(y_val)
# print(y_val)

# window_size = 20
# y_val = np.convolve(y_val, np.ones(window_size) / window_size, mode='valid')

        

file_ = open("ans.txt", "w")
for row in all_y_vals:
    file_.write(" ".join(map(str, row)) + "\n")
file_.close()


# Create the plot
    # Create x values based on the indexes of y_values

    # Create the plot

colors = ['blue' , 'red' , 'green' , '#FFA500']
i = 0
for y_val in all_y_vals:
    plt.plot(x_val, y_val, linestyle='-' , color = colors[i])
    i += 1

    # Add labels and a title
plt.xlabel('Index')
plt.ylabel('Value (y)')
plt.title('Episodes till convergence vs. Discount Rate')
 # Show the plot
plt.grid(True)
plt.show()

