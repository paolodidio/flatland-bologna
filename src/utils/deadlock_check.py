from typing import List
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
import numpy as np
def check_for_deadlock(env):
    location_has_agent = {}
    agents = env.agents
    for agent in agents:
        if agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and agent.position:
            location_has_agent[tuple(agent.position)] = agent
    for agent in agents:
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        else:
            continue
        possible_transitions = env.rail.get_transitions(*agent_virtual_position, agent.direction)
        orientation = agent.direction
        blocking_agents = []
        for branch_direction in [(orientation + i) % 4 for i in range(-1, 3)]:
            if possible_transitions[branch_direction]:
                new_position = get_new_position(agent_virtual_position, branch_direction)
                if new_position in location_has_agent:
                    blocking_agents.append(location_has_agent[(new_position)])
        if len(blocking_agents) == np.count_nonzero(possible_transitions):
            other_blocked =  other_agent_blocked(env, blocking_agents, location_has_agent)
            if other_blocked:
                return True
            else:
                agents = list(filter(lambda x: x not in blocking_agents, agents))
    return False
def other_agent_blocked(env, blocking_agents, location_has_agent):
    for agent in blocking_agents:
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        else:
            continue
        possible_transitions = env.rail.get_transitions(*agent_virtual_position, agent.direction)
        orientation = agent.direction
        blocked_transitions = np.where(possible_transitions, False , True)
        for branch_direction in [(orientation + i) % 4 for i in range(-1, 3)]:
            if possible_transitions[branch_direction]:
                new_position = get_new_position(agent_virtual_position, branch_direction)
                if new_position in location_has_agent:
                    blocked_transitions[branch_direction] = True
                    
        if len(blocked_transitions) != np.count_nonzero(blocked_transitions):
            return False
    return True

def reverse_dir(direction):
    return int((direction + 2) % 4)