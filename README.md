# SARSA Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
SARSA Stands for STATE,ACTION,REWARD,NEXT STATE,NEXT ACTION.

## SARSA LEARNING ALGORITHM
Include the steps involved in the SARSA Learning algorithm
~~~
# Developed by:Koduru Sanath Kumar Reddy
# Regno: 212221240024
~~~

## SARSA LEARNING FUNCTION
~~~
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))
    alphas=decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)

    epsilons=decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        next_state,reward,done,_=env.step(action)
        next_action=select_action(next_state,Q,epsilons[e])
        td_target=reward+gamma*Q[next_state][next_action]*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state,action=next_state,next_action
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    return Q, V, pi, Q_track, pi_track
~~~

## OUTPUT:
Mention the optimal policy, optimal value function , success rate for the optimal policy.
<img width="854" alt="image" src="https://github.com/KoduruSanathKumarReddy/sarsa-learning/assets/69503902/b28bdfb3-390b-4bb0-a666-3c21728b2392">

<img width="1332" alt="image" src="https://github.com/KoduruSanathKumarReddy/sarsa-learning/assets/69503902/c335bd7c-50d0-4e17-8ae7-c79cea6cea30">
<img width="1332" alt="image" src="https://github.com/KoduruSanathKumarReddy/sarsa-learning/assets/69503902/c101abbd-968a-4479-aac4-d9bc45cf2fbc">





## RESULT:
Therefore a python has been successfully developed to find the optimal policy for the given rl environment using SARSA-Learning and state values are compared with Monte Carlo method.

