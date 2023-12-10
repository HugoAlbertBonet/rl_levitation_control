from gymnasium import Env
from gymnasium.spaces.box import Box
import numpy as np
import random
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import pickle
import warnings
import math
from planta import Bobina
warnings.filterwarnings("ignore")

class BobinaEnv(Env):
  """
  Custom Gym Environment for the Hybrid Electromagnetic Suspension, considering just one coil.
  """
  def __init__(self, masa_pod = 200, airgap = 10, duration = 10):
    # Actions: Voltage applied to the coil
    self.action_space = Box(low=np.array([-100]), high = np.array([50]), dtype=np.float32)
    # Observations: Airgap to the ceiling, velocity, distance to the objective, current of the coil
    self.observation_space = Box(low=np.array([10, -np.inf, -10, -45]), high = np.array([23, np.inf, 10, 45]), dtype=np.float32)
    # Initial parameters
    self.state = np.array([airgap, 0, airgap - 19.5, 0])
    self.airgapinicial = airgap
    self.duration = duration
    self.timeleft = duration
    self.crash = False
    self.masa_pod = masa_pod
    self.bobina = Bobina(masa_pod = self.masa_pod, airgap = self.airgapinicial)
    self.steps = []
    self.airgap = airgap
    self.distancia = 0
    self.velocidad = 0
    self.current = 0


  def step(self, action):
    # Apply action
    self.state, self.crash = self.bobina.step(action[0])
    self.airgap, self.velocidad, self.distancia, self.current = self.state

    # Reduce the time of the experiment
    self.timeleft -= 0.001

    # Calculate reward
    reward = -abs(self.distancia)

    # Check if experiment is done
    if self.timeleft <= 0:
      truncated = True
    else: truncated = False
    # Set placeholder for info
    info = {}
    # Only if we implemented the crash
    if self.crash:
      reward = - 50000
      terminated = True
    else:
      terminated = False
    self.steps.append(self.state)
    return self.state, reward, terminated, truncated, info


  def render(self, yes = "yes"):
    # create data
    x = list(range(0,len(self.steps)))
    y = [step[0] for step in self.steps]
    objective = [19.5]*len(self.steps)

    # plot lines
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.tick_params(axis='x', colors='white')    #setting up X-axis tick color to red
    ax.tick_params(axis='y', colors='white')  #setting up Y-axis tick color to black

    plt.plot(x, y, label = "States")
    plt.plot(x, objective, label = "Objective")
    plt.legend()
    plt.show()

  def reset(self, seed = 0):
    # Reset experiment choosing randomly the start (ceiling or floor)
    airgap = random.choice([10, 23])
    self.state = self.airgapinicial
    self.bobina = Bobina(masa_pod = self.masa_pod, airgap = airgap)
    # Reset time
    self.timeleft = self.duration
    self.crash = False
    self.steps = []
    return self.state, seed