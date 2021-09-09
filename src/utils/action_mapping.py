from flatland.envs.rail_env import RailEnvActions
from flatland.envs.agent_utils import RailAgentStatus
from flatland.core.grid.grid4 import Grid4TransitionsEnum
import numpy as np
def action_mapping(action, env, handle):
    agent = env.agents[handle]
    if agent.status == RailAgentStatus.READY_TO_DEPART:
        agent_virtual_position = agent.initial_position
    elif agent.status == RailAgentStatus.ACTIVE:
        agent_virtual_position = agent.position
    elif agent.status == RailAgentStatus.DONE:
        agent_virtual_position = agent.target

    possible_transitions = env.rail.get_transitions(*agent_virtual_position, agent.direction)
    possible_actions = [False, False, False, False, True]
    if np.count_nonzero(possible_transitions) == 1:
        possible_actions[cardinal_to_action(agent.direction, np.argmax(possible_transitions))] = True    
    else:
        for direction in Grid4TransitionsEnum:
            if possible_transitions[direction] == 1:
                # RailEnvActions include DO_NOTHING as 0
                possible_actions[cardinal_to_action(agent.direction, direction)] = True
    if action == 0:
       if possible_actions[RailEnvActions.MOVE_LEFT]:
           return RailEnvActions.MOVE_LEFT
       if possible_actions[RailEnvActions.MOVE_FORWARD] == True:
           return RailEnvActions.MOVE_FORWARD
       if possible_actions[RailEnvActions.MOVE_RIGHT] == True:
           return RailEnvActions.MOVE_RIGHT
    elif action == 1:
        if possible_actions[RailEnvActions.MOVE_RIGHT] == True:
            return RailEnvActions.MOVE_RIGHT
        if possible_actions[RailEnvActions.MOVE_FORWARD] == True:
            return RailEnvActions.MOVE_FORWARD
    else:
        return RailEnvActions.STOP_MOVING
    
def get_legal_actions(env, handle):
    agent = env.agents[handle]
    if agent.status == RailAgentStatus.READY_TO_DEPART:
        agent_virtual_position = agent.initial_position
    elif agent.status == RailAgentStatus.ACTIVE:
        agent_virtual_position = agent.position
    elif agent.status == RailAgentStatus.DONE:
        agent_virtual_position = agent.target

    legal_actions = [False, False, True]
    possible_transitions = env.rail.get_transitions(*agent_virtual_position, agent.direction)
    possible_actions = [False, False, False, False, True]
    if  np.count_nonzero(possible_transitions) == 1:
        possible_actions[cardinal_to_action(agent.direction, np.argmax(possible_transitions))] = True
        if possible_actions[RailEnvActions.MOVE_FORWARD]:
            return[True, False, True]
        if possible_actions[RailEnvActions.MOVE_LEFT]:
            return[True, False, True]
        if possible_actions[RailEnvActions.MOVE_RIGHT]:
            return[True, False, True]
    for direction in Grid4TransitionsEnum:
        if possible_transitions[direction] == 1:
            # RailEnvActions include DO_NOTHING as 0
            possible_actions[cardinal_to_action(agent.direction, direction)] = True
    if possible_actions[RailEnvActions.MOVE_LEFT]:
        legal_actions[0] = True
    if possible_actions[RailEnvActions.MOVE_RIGHT]:
        legal_actions[1] = True 
    if possible_actions[RailEnvActions.MOVE_FORWARD]:
        if possible_actions[RailEnvActions.MOVE_LEFT]:
            legal_actions[1] = True
        elif possible_actions[RailEnvActions.MOVE_RIGHT]:
            legal_actions[0] = True
    return legal_actions   
    
#TODO: how to improve?
def cardinal_to_action(in_dir, out_dir):
    if in_dir == out_dir:
        return 2
    elif (in_dir + 1) % 4 == out_dir:
        return 3
    elif (in_dir + 2) % 4 == out_dir:
        return 4
    else:
        return 1
