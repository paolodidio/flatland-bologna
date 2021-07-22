from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import RailAgentStatus
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
import numpy as np


def is_action_required(handle:int, env:RailEnv):
    agent = env.agents[handle]
    if agent.status == RailAgentStatus.READY_TO_DEPART:
        agent_virtual_position = agent.initial_position
    elif agent.status == RailAgentStatus.ACTIVE:
        agent_virtual_position = agent.position
    elif agent.status == RailAgentStatus.DONE:
        agent_virtual_position = agent.target

    # to verity dead end case
    try:
        possible_transitions = env.rail.get_transitions(*agent_virtual_position, agent.direction)
        for agent_new_direction in Grid4TransitionsEnum:
            if possible_transitions[agent_new_direction]:
                new_cell = get_new_position(agent_virtual_position, agent_new_direction)
                for direction in Grid4TransitionsEnum:
                    possible_transitions_new_cell = env.rail.get_transitions(*new_cell, direction)
                    num_transitions = np.count_nonzero(possible_transitions_new_cell)
                    if num_transitions > 1:
                        return True
    except:
        pass
    return False