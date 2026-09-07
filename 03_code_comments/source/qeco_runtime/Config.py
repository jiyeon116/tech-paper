from __future__ import annotations

from experiment_support import SharedExperiment


class Config(object):
    # System setup
    N_UE = SharedExperiment.NUM_USERS
    N_EDGE = SharedExperiment.NUM_EDGES
    UE_COMP_CAP = 2.6
    UE_TRAN_CAP = 14
    EDGE_COMP_CAP = 42

    # Energy consumption settings
    UE_ENERGY_STATE = [0.25, 0.50, 0.75]
    UE_COMP_ENERGY = 2
    UE_TRAN_ENERGY = 2.3
    UE_IDLE_ENERGY = 0.1
    EDGE_COMP_ENERGY = 5

    # Task requirement
    TASK_COMP_DENS = [0.197, 0.297, 0.397]
    TASK_MIN_SIZE = 1
    TASK_MAX_SIZE = 7
    N_COMPONENT = SharedExperiment.NUM_COMPONENTS
    MAX_DELAY = SharedExperiment.MAX_DELAY

    # Simulation scenario
    N_EPISODE = SharedExperiment.QECO_EPISODES
    N_TIME_SLOT = SharedExperiment.QECO_TIME_SLOTS
    DURATION = 0.1
    TASK_ARRIVE_PROB = SharedExperiment.QECO_TASK_ARRIVE_PROB
    N_TIME = N_TIME_SLOT + MAX_DELAY

    # Algorithm settings
    LEARNING_RATE = 0.01
    REWARD_DECAY = 0.9
    E_GREEDY = 0.99
    N_NETWORK_UPDATE = 200
    MEMORY_SIZE = 500
