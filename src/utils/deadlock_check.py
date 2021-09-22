from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.agent_utils import RailAgentStatus
from flatland.core.grid.grid4 import Grid4TransitionsEnum


def check_if_all_blocked(env):
    """
    Checks whether all the agents are blocked (full deadlock situation).
    In that case it is pointless to keep running inference as no agent will be able to move.
    :param env: current environment
    :return:
    """

    # First build a map of agents in each position
    location_has_agent = {}
    for agent in env.agents:
        if agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and agent.position:
            location_has_agent[tuple(agent.position)] = 1

    # Looks for any agent that can still move
    for handle in env.get_agent_handles():
        agent = env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            continue

        possible_transitions = env.rail.get_transitions(*agent_virtual_position, agent.direction)
        orientation = agent.direction

        for branch_direction in [(orientation + i) % 4 for i in range(-1, 3)]:
            if possible_transitions[branch_direction]:
                new_position = get_new_position(agent_virtual_position, branch_direction)

                if new_position not in location_has_agent:
                    return False

    # No agent can move at all: full deadlock!
    return True

    

# not_in_deadlock = {}

def get_all_trains_in_deadlock(env): 
    cell_has_agent = {}
    blocked_agents = set()     #set of agents that are for sure in a deadlock state
    free_agents = set()        #set of agents that are not in a deadlock state
    for agent in env.agents:
        if agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        else:
            continue
        cell_has_agent[agent_virtual_position] = agent.handle
    for position in cell_has_agent:
        handle = cell_has_agent[position]
        agent = env.agents[handle]
        direction = agent.direction
        possible_transitions = env.rail.get_transitions(*position, direction)
        if not handle in free_agents and not handle in blocked_agents:
            free_agents, blocked_agents = get_inherent_blocked(handle, env, position, possible_transitions, cell_has_agent, free_agents, blocked_agents, {handle})
            if not handle in free_agents and not handle in blocked_agents:
                #it means that all agents close to the agent are wither bocked or checking_agents, but not resolved. this means that the agents are blocking each other ciclically
                blocked_agents.add(handle)
    return free_agents, blocked_agents


def get_inherent_blocked(handle, env, position, possible_transitions, cell_has_agent, free_agents, blocked_agents, checking_agents):
    has_one_other_agent_in_checking = False
    for direction in Grid4TransitionsEnum:
        if possible_transitions[direction]:
            new_position = get_new_position(position, direction)
            if new_position in cell_has_agent:
                other_agent_handle = cell_has_agent[new_position]
                other_agent_direction = env.agents[other_agent_handle].direction
                if other_agent_handle in free_agents:
                    free_agents.add(handle)
                    return free_agents, blocked_agents
                elif other_agent_handle in blocked_agents:
                    pass
                elif other_agent_handle in checking_agents:
                    #avoid cycles (if all agents are in checking_agents it means that they are all blocking each other, it is a deadlock)
                    has_one_other_agent_in_checking = True
                else:
                    #other_agent must be recursively resolved
                    checking_agents.add(other_agent_handle)
                    new_possible_transitions = env.rail.get_transitions(*new_position, other_agent_direction)
                    free_agents, blocked_agents = get_inherent_blocked(other_agent_handle, env, new_position, new_possible_transitions, cell_has_agent, free_agents, blocked_agents, checking_agents)
                    if other_agent_handle in free_agents:
                        free_agents.add(handle)
                        return free_agents, blocked_agents
            else:
                free_agents.add(handle)
                return free_agents, blocked_agents
    if not has_one_other_agent_in_checking:
        blocked_agents.add(handle)
    return free_agents, blocked_agents
            
        
    
