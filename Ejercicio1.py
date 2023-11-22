import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored, cprint

from ambientes import *
from agents import *
from algoritmos import *
from utils import Episode, Experiment
from plot_utils import PlotGridValues, Plot
from tests import *

# Punto 1

# Create environment
shape = (1,3)
env = ABC()
# Create agent
parameters = {\
    'nS': 3,\
    'nA': 2,\
    'gamma': 0.8,\
    'epsilon': 0.1,\
    'alpha': 0.1,\
}
agent_SARSA = SARSA(parameters=parameters)


#Create experiment
experiment = Experiment(environment=env,\
                        env_name='ABC', \
                        num_rounds=200, \
                        num_episodes=500, \
                        num_simulations=10)

# Train agent
agents = experiment.run_experiment(agents=[agent_SARSA],\
                                  names=['SARSA'], \
                                  measures=['round_reward'], \
                                  learn=True)

# Punto 2

# Create experiment
experiment = Experiment(environment=env,\
                        env_name='ABC', \
                        num_rounds=10, \
                        num_episodes=10, \
                        num_simulations=10)
# Test agent already trained
agents = experiment.run_experiment(agents=[agent_SARSA],\
                                  names=['Random'], \
                                  measures=['hist_reward'], \
                                  learn=False)

# Punto 3
# Create environment
env = ABC()
shape=(1,3)
pp = PlotGridValues(shape=shape,dict_acciones=env.dict_acciones)
s_agent = agents[0]
p = s_agent.policy
policy = [np.argmax(p[s,]) for s in range(env.nS)]
policy = np.flipud(np.reshape(policy, shape))
pp.plot_policy(policy)

#Punto 4


