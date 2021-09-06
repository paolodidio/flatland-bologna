"""
Collection of environment-specific ObservationBuilder.
"""
import collections
from typing import Optional, List, Dict, Tuple
from networkx.classes import graph
from networkx.drawing.nx_pylab import draw

import numpy as np

from flatland.core.env import Environment

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.env_prediction_builder import PredictionBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import coordinate_to_position
from flatland.envs.agent_utils import RailAgentStatus, EnvAgent
from flatland.utils.ordered_set import OrderedSet
from numpy.core.numeric import NaN
from src.graph.graph import Graph
import networkx as nx
from flatland.core.grid.grid4 import Grid4TransitionsEnum


# Node = collections.namedtuple('Node', 'dist_own_target_encountered '
#                                         'dist_other_target_encountered '
#                                         'dist_other_agent_encountered '
#                                         'dist_potential_conflict '
#                                         'dist_unusable_switch '
#                                         'dist_to_next_branch '
#                                         'dist_min_to_target '
#                                         'num_agents_same_direction '
#                                         'num_agents_opposite_direction '
#                                         'num_agents_malfunctioning '
#                                         'speed_min_fractional '
#                                         'num_agents_ready_to_depart '
#                                         'childs')
Node = collections.namedtuple('Node', 'dist_own_target_encountered '
                                        'dist_other_target_encountered '
                                        'dist_other_agent_encountered '
                                        'dist_potential_conflict '
                                        'dist_unusable_switch '
                                        'dist_to_next_branch '
                                        'dist_min_to_target '
                                        'num_agents_same_direction '
                                        'num_agents_opposite_direction '
                                        'num_agents_malfunctioning '
                                        'speed_min_fractional '
                                        'num_agents_ready_to_depart '
                                        'is_deadlock '
                                        'childs')
#region TreeObsForRailEnv
class TreeObsForRailEnv(ObservationBuilder):
    """
    TreeObsForRailEnv object.

    This object returns observation vectors for agents in the RailEnv environment.
    The information is local to each agent and exploits the graph structure of the rail
    network to simplify the representation of the state of the environment for each agent.

    For details about the features in the tree observation see the get() function.
    """


    tree_explored_actions_char = ['L', 'F', 'R', 'B']

    def __init__(self, max_depth: int, predictor: PredictionBuilder = None):
        super().__init__()
        self.max_depth = max_depth
        self.observation_dim = 11
        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.predictor = predictor
        self.location_has_target = None

    def reset(self):
        self.location_has_target = {tuple(agent.target): 1 for agent in self.env.agents}

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, Node]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """

        if handles is None:
            handles = []
        if self.predictor:
            self.max_prediction_depth = 0
            self.predicted_pos = {}
            self.predicted_dir = {}
            self.predictions = self.predictor.get()
            if self.predictions:
                for t in range(self.predictor.max_depth + 1):
                    pos_list = []
                    dir_list = []
                    for a in handles:
                        if self.predictions[a] is None:
                            continue
                        pos_list.append(self.predictions[a][t][1:3])
                        dir_list.append(self.predictions[a][t][3])
                    self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                    self.predicted_dir.update({t: dir_list})
                self.max_prediction_depth = len(self.predicted_pos)
        # Update local lookup table for all agents' positions
        # ignore other agents not in the grid (only status active and done)
        # self.location_has_agent = {tuple(agent.position): 1 for agent in self.env.agents if
        #                         agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE]}

        self.location_has_agent = {}
        self.location_has_agent_direction = {}
        self.location_has_agent_speed = {}
        self.location_has_agent_malfunction = {}
        self.location_has_agent_ready_to_depart = {}

        for _agent in self.env.agents:
            if _agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and \
                _agent.position:
                self.location_has_agent[tuple(_agent.position)] = 1
                self.location_has_agent_direction[tuple(_agent.position)] = _agent.direction
                self.location_has_agent_speed[tuple(_agent.position)] = _agent.speed_data['speed']
                self.location_has_agent_malfunction[tuple(_agent.position)] = _agent.malfunction_data[
                    'malfunction']

            if _agent.status in [RailAgentStatus.READY_TO_DEPART] and \
                _agent.initial_position:
                
                self.location_has_agent_ready_to_depart[tuple(_agent.initial_position)] = \
                    self.location_has_agent_ready_to_depart.get(tuple(_agent.initial_position), 0) + 1

        observations = super().get_many(handles)

        return observations

    def get(self, handle: int = 0) -> Node:
        """
        Computes the current observation for agent `handle` in env

        The observation vector is composed of 4 sequential parts, corresponding to data from the up to 4 possible
        movements in a RailEnv (up to because only a subset of possible transitions are allowed in RailEnv).
        The possible movements are sorted relative to the current orientation of the agent, rather than NESW as for
        the transitions. The order is::

            [data from 'left'] + [data from 'forward'] + [data from 'right'] + [data from 'back']

        Each branch data is organized as::

            [root node information] +
            [recursive branch data from 'left'] +
            [... from 'forward'] +
            [... from 'right] +
            [... from 'back']

        Each node information is composed of 9 features:

        #1:
            if own target lies on the explored branch the current distance from the agent in number of cells is stored.

        #2:
            if another agents target is detected the distance in number of cells from the agents current location\
            is stored

        #3:
            if another agent is detected the distance in number of cells from current agent position is stored.

        #4:
            possible conflict detected
            tot_dist = Other agent predicts to pass along this cell at the same time as the agent, we store the \
             distance in number of cells from current agent position

            0 = No other agent reserve the same cell at similar time

        #5:
            if an not usable switch (for agent) is detected we store the distance.

        #6:
            This feature stores the distance in number of cells to the next branching  (current node)

        #7:
            minimum distance from node to the agent's target given the direction of the agent if this path is chosen

        #8:
            agent in the same direction
            n = number of agents present same direction \
                (possible future use: number of other agents in the same direction in this branch)
            0 = no agent present same direction

        #9:
            agent in the opposite direction
            n = number of agents present other direction than myself (so conflict) \
                (possible future use: number of other agents in other direction in this branch, ie. number of conflicts)
            0 = no agent present other direction than myself

        #10:
            malfunctioning/blokcing agents
            n = number of time steps the oberved agent remains blocked

        #11:
            slowest observed speed of an agent in same direction
            1 if no agent is observed

            min_fractional speed otherwise
        #12:
            number of agents ready to depart but no yet active

        Missing/padding nodes are filled in with -inf (truncated).
        Missing values in present node are filled in with +inf (truncated).


        In case of the root node, the values are [0, 0, 0, 0, distance from agent to target, own malfunction, own speed]
        In case the target node is reached, the values are [0, 0, 0, 0, 0].
        """

        if handle > len(self.env.agents):
            print("ERROR: obs _get - handle ", handle, " len(agents)", len(self.env.agents))
        agent = self.env.agents[handle]  # TODO: handle being treated as index

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        num_transitions = np.count_nonzero(possible_transitions)

        # Here information about the agent itself is stored
        distance_map = self.env.distance_map.get()

        # was referring to TreeObsForRailEnv.Node
        root_node_observation = Node(dist_own_target_encountered=0, dist_other_target_encountered=0,
                                                       dist_other_agent_encountered=0, dist_potential_conflict=0,
                                                       dist_unusable_switch=0, dist_to_next_branch=0,
                                                       dist_min_to_target=distance_map[
                                                           (handle, *agent_virtual_position,
                                                            agent.direction)],
                                                       num_agents_same_direction=0, num_agents_opposite_direction=0,
                                                       num_agents_malfunctioning=agent.malfunction_data['malfunction'],
                                                       speed_min_fractional=agent.speed_data['speed'],
                                                       num_agents_ready_to_depart=0,
                                                       childs={})
        #print("root node type:", type(root_node_observation))

        visited = OrderedSet()

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # If only one transition is possible, the tree is oriented with this transition as the forward branch.
        orientation = agent.direction

        if num_transitions == 1:
            orientation = np.argmax(possible_transitions)
        for i, branch_direction in enumerate([(orientation + i) % 4 for i in range(-1, 3)]):

            if possible_transitions[branch_direction]:
                new_cell = get_new_position(agent_virtual_position, branch_direction)

                
                branch_observation, branch_visited = \
                    self._explore_branch(handle, new_cell, branch_direction, 1, 1)
                root_node_observation.childs[self.tree_explored_actions_char[i]] = branch_observation

                visited |= branch_visited
            else:
                # add cells filled with infinity if no transition is possible
                root_node_observation.childs[self.tree_explored_actions_char[i]] = -np.inf
        self.env.dev_obs_dict[handle] = visited

        return root_node_observation

    def _explore_branch(self, handle, position, direction, tot_dist, depth):
        """
        Utility function to compute tree-based observations.
        We walk along the branch and collect the information documented in the get() function.
        If there is a branching point a new node is created and each possible branch is explored.
        """

        # [Recursive branch opened]
        if depth >= self.max_depth + 1:
            return [], []

        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops
        exploring = True
        last_is_switch = False
        last_is_dead_end = False
        last_is_terminal = False  # wrong cell OR cycle;  either way, we don't want the agent to land here
        last_is_target = False

        visited = OrderedSet()
        agent = self.env.agents[handle]
        time_per_cell = np.reciprocal(agent.speed_data["speed"])
        own_target_encountered = np.inf
        other_agent_encountered = np.inf
        other_target_encountered = np.inf
        potential_conflict = np.inf
        unusable_switch = np.inf
        other_agent_same_direction = 0
        other_agent_opposite_direction = 0
        malfunctioning_agent = 0
        min_fractional_speed = 1.
        num_steps = 1
        other_agent_ready_to_depart_encountered = 0
        while exploring:
            # #############################
            # #############################
            # Modify here to compute any useful data required to build the end node's features. This code is called
            # for each cell visited between the previous branching node and the next switch / target / dead-end.
            if position in self.location_has_agent:
                if tot_dist < other_agent_encountered:
                    other_agent_encountered = tot_dist

                # Check if any of the observed agents is malfunctioning, store agent with longest duration left
                if self.location_has_agent_malfunction[position] > malfunctioning_agent:
                    malfunctioning_agent = self.location_has_agent_malfunction[position]

                other_agent_ready_to_depart_encountered += self.location_has_agent_ready_to_depart.get(position, 0)

                if self.location_has_agent_direction[position] == direction:
                    # Cummulate the number of agents on branch with same direction
                    other_agent_same_direction += 1

                    # Check fractional speed of agents
                    current_fractional_speed = self.location_has_agent_speed[position]
                    if current_fractional_speed < min_fractional_speed:
                        min_fractional_speed = current_fractional_speed

                else:
                    # If no agent in the same direction was found all agents in that position are other direction
                    # Attention this counts to many agents as a few might be going off on a switch.
                    other_agent_opposite_direction += self.location_has_agent[position]

                # Check number of possible transitions for agent and total number of transitions in cell (type)
            cell_transitions = self.env.rail.get_transitions(*position, direction)
            transition_bit = bin(self.env.rail.get_full_transitions(*position))
            total_transitions = transition_bit.count("1")
            crossing_found = False
            if int(transition_bit, 2) == int('1000010000100001', 2):
                crossing_found = True

            # Register possible future conflict
            predicted_time = int(tot_dist * time_per_cell)
            if self.predictor and predicted_time < self.max_prediction_depth:
                int_position = coordinate_to_position(self.env.width, [position])
                if tot_dist < self.max_prediction_depth:

                    pre_step = max(0, predicted_time - 1)
                    post_step = min(self.max_prediction_depth - 1, predicted_time + 1)

                    # Look for conflicting paths at distance tot_dist
                    if int_position in np.delete(self.predicted_pos[predicted_time], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[predicted_time] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[predicted_time][ca] and cell_transitions[
                                self._reverse_dir(
                                    self.predicted_dir[predicted_time][ca])] == 1 and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                    # Look for conflicting paths at distance num_step-1
                    elif int_position in np.delete(self.predicted_pos[pre_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[pre_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[pre_step][ca] \
                                and cell_transitions[self._reverse_dir(self.predicted_dir[pre_step][ca])] == 1 \
                                and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist

                    # Look for conflicting paths at distance num_step+1
                    elif int_position in np.delete(self.predicted_pos[post_step], handle, 0):
                        conflicting_agent = np.where(self.predicted_pos[post_step] == int_position)
                        for ca in conflicting_agent[0]:
                            if direction != self.predicted_dir[post_step][ca] and cell_transitions[self._reverse_dir(
                                self.predicted_dir[post_step][ca])] == 1 \
                                and tot_dist < potential_conflict:  # noqa: E125
                                potential_conflict = tot_dist
                            if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
                                potential_conflict = tot_dist
            
            if position in self.location_has_target and position != agent.target:
                if tot_dist < other_target_encountered:
                    other_target_encountered = tot_dist

            if position == agent.target and tot_dist < own_target_encountered:
                own_target_encountered = tot_dist

            # #############################
            # #############################
            if (position[0], position[1], direction) in visited:
                last_is_terminal = True
                break
            visited.add((position[0], position[1], direction))

            # If the target node is encountered, pick that as node. Also, no further branching is possible.
            if np.array_equal(position, self.env.agents[handle].target):
                last_is_target = True
                break

            # Check if crossing is found --> Not an unusable switch
            if crossing_found:
                # Treat the crossing as a straight rail cell
                total_transitions = 2
            num_transitions = np.count_nonzero(cell_transitions)

            exploring = False

            # Detect Switches that can only be used by other agents.
            if total_transitions > 2 > num_transitions and tot_dist < unusable_switch:
                unusable_switch = tot_dist

            if num_transitions == 1:
                # Check if dead-end, or if we can go forward along direction
                nbits = total_transitions
                if nbits == 1:
                    # Dead-end!
                    last_is_dead_end = True

                if not last_is_dead_end:
                    # Keep walking through the tree along `direction`
                    exploring = True
                    # convert one-hot encoding to 0,1,2,3
                    direction = np.argmax(cell_transitions)
                    position = get_new_position(position, direction)
                    num_steps += 1
                    tot_dist += 1
            elif num_transitions > 0:
                # Switch detected
                last_is_switch = True
                break

            elif num_transitions == 0:
                # Wrong cell type, but let's cover it and treat it as a dead-end, just in case
                print("WRONG CELL TYPE detected in tree-search (0 transitions possible) at cell", position[0],
                      position[1], direction)
                last_is_terminal = True
                break

        # `position` is either a terminal node or a switch

        # #############################
        # #############################
        # Modify here to append new / different features for each visited cell!

        if last_is_target:
            dist_to_next_branch = tot_dist
            dist_min_to_target = 0
        elif last_is_terminal:
            dist_to_next_branch = np.inf
            dist_min_to_target = self.env.distance_map.get()[handle, position[0], position[1], direction]
        else:
            dist_to_next_branch = tot_dist
            dist_min_to_target = self.env.distance_map.get()[handle, position[0], position[1], direction]

        # TreeObsForRailEnv.Node
        node = Node(dist_own_target_encountered=own_target_encountered,
                                      dist_other_target_encountered=other_target_encountered,
                                      dist_other_agent_encountered=other_agent_encountered,
                                      dist_potential_conflict=potential_conflict,
                                      dist_unusable_switch=unusable_switch,
                                      dist_to_next_branch=dist_to_next_branch,
                                      dist_min_to_target=dist_min_to_target,
                                      num_agents_same_direction=other_agent_same_direction,
                                      num_agents_opposite_direction=other_agent_opposite_direction,
                                      num_agents_malfunctioning=malfunctioning_agent,
                                      speed_min_fractional=min_fractional_speed,
                                      num_agents_ready_to_depart=other_agent_ready_to_depart_encountered,
                                      childs={})

        # #############################
        # #############################
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # Get the possible transitions
        possible_transitions = self.env.rail.get_transitions(*position, direction)
        for i, branch_direction in enumerate([(direction + 4 + i) % 4 for i in range(-1, 3)]):
            if last_is_dead_end and self.env.rail.get_transition((*position, direction),
                                                                 (branch_direction + 2) % 4):
                # Swap forward and back in case of dead-end, so that an agent can learn that going forward takes
                # it back
                new_cell = get_new_position(position, (branch_direction + 2) % 4)
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          (branch_direction + 2) % 4,
                                                                          tot_dist + 1,
                                                                          depth + 1)
                node.childs[self.tree_explored_actions_char[i]] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited
            elif last_is_switch and possible_transitions[branch_direction]:
                new_cell = get_new_position(position, branch_direction)
                branch_observation, branch_visited = self._explore_branch(handle,
                                                                          new_cell,
                                                                          branch_direction,
                                                                          tot_dist + 1,
                                                                          depth + 1)
                node.childs[self.tree_explored_actions_char[i]] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited
            else:
                # no exploring possible, add just cells with infinity
                node.childs[self.tree_explored_actions_char[i]] = -np.inf

        if depth == self.max_depth:
            node.childs.clear()
        return node, visited

    def util_print_obs_subtree(self, tree: Node):
        """
        Utility function to print tree observations returned by this object.
        """
        self.print_node_features(tree, "root", "")
        for direction in self.tree_explored_actions_char:
            self.print_subtree(tree.childs[direction], direction, "\t")

    @staticmethod
    def print_node_features(node: Node, label, indent):
        print(indent, "Direction ", label, ": ", node.dist_own_target_encountered, ", ",
              node.dist_other_target_encountered, ", ", node.dist_other_agent_encountered, ", ",
              node.dist_potential_conflict, ", ", node.dist_unusable_switch, ", ", node.dist_to_next_branch, ", ",
              node.dist_min_to_target, ", ", node.num_agents_same_direction, ", ", node.num_agents_opposite_direction,
              ", ", node.num_agents_malfunctioning, ", ", node.speed_min_fractional, ", ",
              node.num_agents_ready_to_depart)

    def print_subtree(self, node, label, indent):
        if node == -np.inf or not node:
            print(indent, "Direction ", label, ": -np.inf")
            return

        self.print_node_features(node, label, indent)

        if not node.childs:
            return

        for direction in self.tree_explored_actions_char:
            self.print_subtree(node.childs[direction], direction, indent + "\t")

    def set_env(self, env: Environment):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)

    def _reverse_dir(self, direction):
        return int((direction + 2) % 4)
#endregion

class TreeObsForRailEnvUsingGraph(ObservationBuilder):
    """
    TreeObsForRailEnv object.

    This object returns observation vectors for agents in the RailEnv environment.
    The information is local to each agent and exploits the graph structure of the rail
    network to simplify the representation of the state of the environment for each agent.

    For details about the features in the tree observation see the get() function.
    """


    tree_explored_actions_char = ['L', 'F', 'R', 'B']

    def __init__(self, max_depth: int, 
                # map_graph: Graph, 
                predictor: PredictionBuilder = None,
                ):
        super().__init__()
        # self.map_graph = map_graph
        self.max_depth = max_depth
        self.observation_dim = 12
        # self.location_has_agent = {}
        # self.location_has_agent_direction = {}
        self.predictor = predictor
        # self.location_has_target = None

    # def reset(self):
    #     self.location_has_target = {tuple(agent.target): 1 for agent in self.env.agents}
    
    # Reset internal values
    def reset(self):
        self.map_graph = Graph(self.env)
        

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, Node]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """
        if self.predictor:
            #TODO: implement switch
            self.node_has_predicted_train = {}
            self.predictions = self.predictor.get()
            if self.predictions:
                for t in range(self.predictor.max_depth + 1):
                    for a in handles:
                        if self.predictions[a] is None:
                            continue
                        pos = self.predictions[a][t][1:3]
                        dir = self.predictions[a][t][3]
                        opposite_dir = (dir+2)%4
                        if pos[0] == pos[0] and pos[1] == pos[1]: 
                            pos = (int(pos[0]), int(pos[1]))
                            full_transitions = bin(self.env.rail.get_full_transitions(*pos))
                            for direction in Grid4TransitionsEnum:
                                if full_transitions.count('1') > 2 and direction == dir:
                                    continue
                                if dir != direction:
                                    if (pos, direction) in self.map_graph.cell_connected_to_node:
                                        node, distance = self.map_graph.cell_connected_to_node[(pos, direction)]
                                        if node not in self.node_has_predicted_train:
                                            self.node_has_predicted_train[node] = {}
                                        if a not in self.node_has_predicted_train[node]:
                                            
                                            self.node_has_predicted_train[node][a] = (t, t, distance)
                                        else:
                                            t_min, t_max, distance_min = self.node_has_predicted_train[node][a]
                                            if t < t_min:
                                                self.node_has_predicted_train[node][a] = (t, t_max, distance)
                                            elif t > t_max:
                                                self.node_has_predicted_train[node][a] = (t_min, t, distance_min)
                                            else:
                                                self.node_has_predicted_train[node][a] = (t_min, t_max, distance_min)                    
                    #     pos_list.append(self.predictions[a][t][1:3])
                    #     dir_list.append(self.predictions[a][t][3])
                    # self.predicted_pos.update({t: coordinate_to_position(self.env.width, pos_list)})
                    # self.predicted_dir.update({t: dir_list})

        self.node_has_agent_going_to_switch = {}
        self.node_has_agent_coming_from_switch = {}
        # self.node_has_agent_on_switch = {}
        self.node_has_malfunction_agent = {}
        self.node_has_slower_agent_speed_same_direction = {}
        self.node_has_agents_ready_to_depart = {}
        
        #dictionary of dictionaries: for every node, for every agent is given the distance from the target
        self.node_has_target_of_agent = {}     
        # suppose that has been implemented a dictionary in map_graph cell_connected_to_node which, given the cell, returns the node
        # suppose to have a dictionary in map_graph is_switch that, given a cell, is 1 if the cell is a switch
        for _agent in self.env.agents:
            # NOTE: should we include RailAgentStatus.DONE?
            if _agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and _agent.position:
                for direction in Grid4TransitionsEnum:
                    if (_agent.position, direction) in self.map_graph.cell_connected_to_node:
                        # opposite_agent_direction = _agent.direction+2%4
                        coming_from_switch = (direction != _agent.direction)
                        node, distance = self.map_graph.cell_connected_to_node[(_agent.position, direction)]
                        if coming_from_switch:
                            #if the agent is in the switch, it does not count as coming from it, as it is not sure where it will move (not true for going to switch)
                            if distance>0:
                                if not node in self.node_has_agent_coming_from_switch:
                                    self.node_has_agent_coming_from_switch[node] = []
                                self.node_has_agent_coming_from_switch[node].append((_agent.handle, distance)) 
                        else:
                            if not node in self.node_has_agent_going_to_switch:
                                self.node_has_agent_going_to_switch[node] = []
                            self.node_has_agent_going_to_switch[node].append((_agent.handle, distance))
                        
                        #copy pasted the already existing algorithm (adapting it a bit)
                        if _agent.malfunction_data['malfunction'] > 0:
                            if not node in self.node_has_malfunction_agent:
                                self.node_has_malfunction_agent[node] = []
                            self.node_has_malfunction_agent[node].append(_agent.malfunction_data['malfunction'], distance)
                        
                        # if not node in self.slower_agent_speed_same_direction:
                        #     self.slower_agent_speed_same_direction[node] = []
                        # self.slower_agent_speed_same_direction[node].append(_agent.speed_data['speed'], distance)
                        target_position = _agent.target
                        for direction in Grid4TransitionsEnum:
                            if (target_position, direction) in self.map_graph.cell_connected_to_node:
                                #FIXME: typo? _agent.position instead of target_position
                                # node, distance = self.map_graph.cell_connected_to_node[(_agent.position, int(direction))]
                                node, distance = self.map_graph.cell_connected_to_node[(target_position, int(direction))]
                                if not node in self.node_has_target_of_agent:
                                    self.node_has_target_of_agent[node] = {}
                                self.node_has_target_of_agent[node][_agent.handle] = distance
                 # target_position = _agent.target
                # for direction in Grid4TransitionsEnum:
                #     if (target_position, direction) in self.map_graph.cell_connected_to_node:
                #         node, distance = self.map_graph.cell_connected_to_node[(_agent.position, int(direction))]
                #         if not node in self.node_h(as_target_of_agent:
                #             self.node_has_target_of_agent[node] = {}
                #         self.node_has_target_of_agent[node][_agent.handle] = distance


                # if _agent.position in self.map_graph.is_switch:
                #     for direction in Grid4TransitionsEnum:
                #         self.node_has_agent_on_switch
            if _agent.status in [RailAgentStatus.READY_TO_DEPART] and _agent.initial_position:
                if (_agent.initial_position, _agent.direction) in self.map_graph.cell_connected_to_node:
                    #NOTE: should i use _agent.initial_direction?
                    node, distance = self.map_graph.cell_connected_to_node[(_agent.initial_position, _agent.direction)]
                    if not node in self.node_has_agents_ready_to_depart:
                        self.node_has_agents_ready_to_depart[node] = []
                        self.node_has_agents_ready_to_depart[node].append((_agent.handle, distance))
                    elif (_agent.handle, distance) not in self.node_has_agents_ready_to_depart[node]:
                        self.node_has_agents_ready_to_depart[node].append((_agent.handle, distance))
            # if _agent.status in [RailAgentStatus.ACTIVE, RailAgentStatus.DONE] and _agent.position:
            #     for direction in Grid4TransitionsEnum:
            #         if (_agent.initial_position, direction) in self.map_graph.cell_connected_to_node:
            #             node, distance = self.map_graph.cell_connected_to_node[(_agent.initial_position, direction)]
            #             if not node in self.node_has_agents_ready_to_depart:
            #                 self.node_has_agents_ready_to_depart[node] = []
            #             self.node_has_agents_ready_to_depart[node].append(_agent.handle, distance)
        





        observations = super().get_many(handles)

        return observations

    def get(self, handle: int = 0) -> Node:
        """
        Computes the current observation for agent `handle` in env

        The observation vector is composed of 4 sequential parts, corresponding to data from the up to 4 possible
        movements in a RailEnv (up to because only a subset of possible transitions are allowed in RailEnv).
        The possible movements are sorted relative to the current orientation of the agent, rather than NESW as for
        the transitions. The order is::

            [data from 'left'] + [data from 'forward'] + [data from 'right'] + [data from 'back']

        Each branch data is organized as::

            [root node information] +
            [recursive branch data from 'left'] +
            [... from 'forward'] +
            [... from 'right] +
            [... from 'back']

        Each node information is composed of 9 features:

        #1:
            if own target lies on the explored branch the current distance from the agent in number of cells is stored.

        #2:
            if another agents target is detected the distance in number of cells from the agents current location\
            is stored

        #3:
            if another agent is detected the distance in number of cells from current agent position is stored.

        #4:
            possible conflict detected
            tot_dist = Other agent predicts to pass along this cell at the same time as the agent, we store the \
             distance in number of cells from current agent position

            0 = No other agent reserve the same cell at similar time

        #5:
            if an not usable switch (for agent) is detected we store the distance.

        #6:
            This feature stores the distance in number of cells to the next branching  (current node)

        #7:
            minimum distance from node to the agent's target given the direction of the agent if this path is chosen

        #8:
            agent in the same direction
            n = number of agents present same direction \
                (possible future use: number of other agents in the same direction in this branch)
            0 = no agent present same direction

        #9:
            agent in the opposite direction
            n = number of agents present other direction than myself (so conflict) \
                (possible future use: number of other agents in other direction in this branch, ie. number of conflicts)
            0 = no agent present other direction than myself

        #10:
            malfunctioning/blokcing agents
            n = number of time steps the oberved agent remains blocked

        #11:
            slowest observed speed of an agent in same direction
            1 if no agent is observed

            min_fractional speed otherwise
        #12:
            number of agents ready to depart but no yet active

        #13:
            deadlock present in this node (if the current train will reach this node, it will cause a deadlock)

        Missing/padding nodes are filled in with -inf (truncated).
        Missing values in present node are filled in with +inf (truncated).


        In case of the root node, the values are [0, 0, 0, 0, distance from agent to target, own malfunction, own speed]
        In case the target node is reached, the values are [0, 0, 0, 0, 0].
        """

        if handle > len(self.env.agents):
            print("ERROR: obs _get - handle ", handle, " len(agents)", len(self.env.agents))
        agent = self.env.agents[handle]  # TODO: handle being treated as index

        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        transitions = bin(self.env.rail.get_full_transitions(*agent_virtual_position))
        possible_transitions = self.env.rail.get_transitions(*agent_virtual_position, agent.direction)
        num_transitions = transitions.count('1')

        # Here information about the agent itself is stored
        distance_map = self.env.distance_map.get()
        
        # was referring to TreeObsForRailEnv.Node
        root_node_observation = Node(dist_own_target_encountered=0, dist_other_target_encountered=0,
                                                       dist_other_agent_encountered=0, dist_potential_conflict=0,
                                                       dist_unusable_switch=0, dist_to_next_branch=0,
                                                       dist_min_to_target=distance_map[
                                                           (handle, *agent_virtual_position,
                                                            agent.direction)],
                                                       num_agents_same_direction=0, num_agents_opposite_direction=0,
                                                       num_agents_malfunctioning=agent.malfunction_data['malfunction'],
                                                       speed_min_fractional=agent.speed_data['speed'],
                                                       num_agents_ready_to_depart=0,
                                                       is_deadlock=0,
                                                       childs={})
        #print("root node type:", type(root_node_observation))

        visited = OrderedSet()
        
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # If only one transition is possible, the tree is oriented with this transition as the forward branch.
        orientation = agent.direction
        if num_transitions == 2:
            orientation = np.argmax(possible_transitions)
            # (node, node_distance) = ((agent_virtual_position[0], agent_virtual_position[1], orientation), 0)
        if ((agent_virtual_position), orientation) not in self.map_graph.cell_connected_to_node:
            pass
        (node, node_distance) = self.map_graph.cell_connected_to_node[((agent_virtual_position), agent.direction)]

        root_node_observation.childs[self.tree_explored_actions_char[0]] = -np.inf
        root_node_observation.childs[self.tree_explored_actions_char[1]] = -np.inf
        root_node_observation.childs[self.tree_explored_actions_char[2]] = -np.inf
        root_node_observation.childs[self.tree_explored_actions_char[3]] = -np.inf
        
        out_edges = self.map_graph.graph.out_edges(node, data=True)
        if len(out_edges) > 0:
            for (_, _, in_node_direction),(out_node_row, out_node_col, out_node_direction), c in out_edges:
                #only one possible transaction, direction is always forward
                if num_transitions == 2:
                    direction = 1
                else:
                    #converting the in and out cardinal point to a relative direction
                    direction = self._cardinal_to_action(in_node_direction, c['direction'])
                #the agent isn't in a switch, so it should start from the first switch
                if num_transitions == 2:
                    branch_observation, branch_visited = \
                        self._explore_branch(handle, orientation, node, 0, 1, node_distance)
                #the agent is already in a switch, so it should start exploring from the next one (avoid exploring the first switch two times)
                else:
                    branch_observation, branch_visited = \
                        self._explore_branch(handle, orientation, (out_node_row, out_node_col, out_node_direction), 0, 1, node_distance)
                
                root_node_observation.childs[self.tree_explored_actions_char[direction]] = branch_observation
                visited |= branch_visited    
        # for i, branch_direction in enumerate([(orientation + i) % 4 for i in range(-1, 3)]):
            
        #     if possible_transitions[branch_direction] and (agent_virtual_position , agent.direction) in self.map_graph.cell_connected_to_node:
        #         # new_cell = get_new_position(agent_virtual_position, branch_direction)
        #         (node, node_distance) = self.map_graph.cell_connected_to_node[((agent_virtual_position), agent.direction)]
        #         branch_observation, branch_visited = \
        #             self._explore_branch(handle, branch_direction, node, 1, 1, node_distance)
        #         root_node_observation.childs[self.tree_explored_actions_char[i]] = branch_observation
        #         # branch_observation, branch_visited = \
        #         #     self._explore_branch(handle, node, 1, 1, node_distance)
        #         # root_node_observation.childs[self.tree_explored_actions_char[i]] = branch_observation

        #         visited |= branch_visited
        #     else:
        #         # add cells filled with infinity if no transition is possible
        #         root_node_observation.childs[self.tree_explored_actions_char[i]] = -np.inf
        self.env.dev_obs_dict[handle] = visited

        return root_node_observation

    def _explore_branch(self, handle, direction, graph_node, tot_dist, depth, agent_to_node_distance = np.inf):
    # def _explore_branch(self, handle, graph_node, tot_dist, depth, agent_to_node_distance = np.inf):
        """
        Utility function to compute tree-based observations.
        We walk along the branch and collect the information documented in the get() function.
        If there is a branching point a new node is created and each possible branch is explored.
        
        ATTENTION!
        TODO: must also be considered the first case (between root node and the first switch)
        """

        # [Recursive branch opened]
        if depth >= self.max_depth + 1:
            return [], []

        # Continue along direction until next switch or
        # until no transitions are possible along the current direction (i.e., dead-ends)
        # We treat dead-ends as nodes, instead of going back, to avoid loops
        # exploring = True
        # last_is_switch = False
        # last_is_dead_end = False
        # last_is_terminal = False  # wrong cell OR cycle;  either way, we don't want the agent to land here
        # last_is_target = False

        visited = OrderedSet()
        agent = self.env.agents[handle]
        time_per_cell = np.reciprocal(agent.speed_data["speed"])
        own_target_encountered = np.inf
        other_agent_encountered = np.inf
        other_target_encountered = np.inf
        potential_conflict = np.inf
        unusable_switch = np.inf
        other_agent_same_direction = 0
        other_agent_opposite_direction = 0
        malfunctioning_agent = 0
        min_fractional_speed = 1.
        num_steps = 1
        other_agent_ready_to_depart_encountered = 0

        #total distance including also the distance of the current node
        in_edges = self.map_graph.graph.in_edges(graph_node, data=True)


        if len(in_edges) > 0:
            #there could be more than one in edge... to check if the graph is oriented
            for u,v,c in in_edges:
                #TODO: improve
                if tot_dist != 0:
                    tot_dist_next = tot_dist + c['distance']
                else:
                    tot_dist_next = agent_to_node_distance
                break
            agent_to_node_distance = tot_dist_next
        
        #region #1: 
        # if own target lies on the explored branch the current distance from the agent in number of cells is stored.
        if graph_node in self.node_has_target_of_agent:
            if handle in self.node_has_target_of_agent[graph_node]:
                distance = self.node_has_target_of_agent[graph_node][handle]
                if distance < agent_to_node_distance:
                    own_target_encountered = tot_dist_next - distance
        #endregion
        #region #2:
        #  if another agents target is detected the distance in number of cells from the agents current location\ is stored
            distances = list(self.node_has_target_of_agent[graph_node].values())
            distances = list(filter(lambda x: x < agent_to_node_distance, distances))
            if len(distances) > 0:
                other_target_encountered = tot_dist_next - np.max(distances)
        #endregion
        #region #3: 
        # if another agent is detected the distance in number of cells from current agent position is stored.
        # NOTE: only opposite direction or also same direction?
        if graph_node in self.node_has_agent_coming_from_switch:
        # if len(self.node_has_agent_coming_from_switch) > 0 and graph_node in self.node_has_agent_coming_from_switch:
            # handles, distances = zip(*self.node_has_agent_coming_from_switch[graph_node])
            handle_distances = list(filter(lambda x: x[0]!=handle and x[1] < agent_to_node_distance, self.node_has_agent_coming_from_switch[graph_node]))
            if len(handle_distances) > 0:
                _, distances = zip(*handle_distances)
                potential_conflict = tot_dist_next - np.max(distances)
        #endregion
        #region #4:
        #     possible conflict detected
        #     tot_dist = Other agent predicts to pass along this cell at the same time as the agent, we store the \
        #      distance in number of cells from current agent position

        #     0 = No other agent reserve the same cell at similar time
        predicted_time = int(tot_dist * time_per_cell)
        if self.predictor:
            # pre_step = int(tot_dist * time_per_cell)
            # post_step = int(tot_dist_next * time_per_cell)
            if graph_node in self.node_has_predicted_train:
                for a in self.env.agents:
                    if a.handle == handle:
                        continue
                    if a.handle in self.node_has_predicted_train[graph_node]:
                        time_min, time_max, distance = self.node_has_predicted_train[graph_node][a.handle]
                        #to exclude the case where the predicted train has already passed the root node
                        if distance < agent_to_node_distance:
                            if predicted_time in range(time_min, time_max):
                                potential_conflict = tot_dist_next - distance
                     
        # if self.predictor and predicted_time < self.max_prediction_depth:
        #     int_position = coordinate_to_position(self.env.width, [position])
        #     if tot_dist < self.max_prediction_depth:
        #         pre_step = max(0, predicted_time - 1)
        #         post_step = min(self.max_prediction_depth - 1, predicted_time + 1)

        #         # Look for conflicting paths at distance tot_dist
        #         if int_position in np.delete(self.predicted_pos[predicted_time], handle, 0):
        #             conflicting_agent = np.where(self.predicted_pos[predicted_time] == int_position)
        #             for ca in conflicting_agent[0]:
        #                 if direction != self.predicted_dir[predicted_time][ca] and cell_transitions[
        #                     self._reverse_dir(
        #                         self.predicted_dir[predicted_time][ca])] == 1 and tot_dist < potential_conflict:
        #                     potential_conflict = tot_dist
        #                 if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
        #                     potential_conflict = tot_dist
        #         # Look for conflicting paths at distance num_step-1
        #         elif int_position in np.delete(self.predicted_pos[pre_step], handle, 0):
        #             conflicting_agent = np.where(self.predicted_pos[pre_step] == int_position)
        #             for ca in conflicting_agent[0]:
        #                 if direction != self.predicted_dir[pre_step][ca] \
        #                     and cell_transitions[self._reverse_dir(self.predicted_dir[pre_step][ca])] == 1 \
        #                     and tot_dist < potential_conflict:  # noqa: E125
        #                     potential_conflict = tot_dist
        #                 if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
        #                     potential_conflict = tot_dist
        #         # Look for conflicting paths at distance num_step+1
        #         elif int_position in np.delete(self.predicted_pos[post_step], handle, 0):
        #             conflicting_agent = np.where(self.predicted_pos[post_step] == int_position)
        #             for ca in conflicting_agent[0]:
        #                 if direction != self.predicted_dir[post_step][ca] and cell_transitions[self._reverse_dir(
        #                     self.predicted_dir[post_step][ca])] == 1 \
        #                     and tot_dist < potential_conflict:  # noqa: E125
        #                     potential_conflict = tot_dist
        #                 if self.env.agents[ca].status == RailAgentStatus.DONE and tot_dist < potential_conflict:
        #                     potential_conflict = tot_dist
            
        #endregion
        #region  #5:
        #     if an not usable switch (for agent) is detected we store the distance.
        #   this point is useless, as we are changing the nodes such that also non usable switch is encoded as a node
        #endregion
        #region #6:
        #     This feature stores the distance in number of cells to the next branching  (current node)
        # NOTE: is this feature really useful?
        dist_to_next_branch = min(tot_dist_next - tot_dist, agent_to_node_distance)
        #endregion
        #region #7:
        #     minimum distance from node to the agent's target given the direction of the agent if this path is chosen
        # NOTE: is this feature really useful?
        if own_target_encountered != np.inf:
            dist_to_next_branch = own_target_encountered
            dist_min_to_target = 0
        else:
            # dist_to_next_branch = tot_dist
            #TODO: direction?
            row_num, col_num, direction = graph_node
            dist_min_to_target = self.env.distance_map.get()[handle, row_num, col_num, direction]
        #endregion
        #region #8:
        #     agent in the same direction
        #     n = number of agents present same direction \
        #         (possible future use: number of other agents in the same direction in this branch)
        #     0 = no agent present same direction
        if graph_node in self.node_has_agent_going_to_switch:
            distances = list(zip(*self.node_has_agent_going_to_switch[graph_node]))[1]
            distances = list(filter(lambda x: x < agent_to_node_distance, distances))
            other_agent_same_direction = len(distances)
        #endregion
        #region #9:
        #     agent in the opposite direction
        #     n = number of agents present other direction than myself (so conflict) \
        #         (possible future use: number of other agents in other direction in this branch, ie. number of conflicts)
        #     0 = no agent present other direction than myself
        if graph_node in self.node_has_agent_coming_from_switch:
            distances = list(zip(*self.node_has_agent_coming_from_switch[graph_node]))[1]
            distances = list(filter(lambda x: x < agent_to_node_distance, distances))
            other_agent_opposite_direction = len(distances)
        #endregion
        #region #10:
        #     malfunctioning/blokcing agents
        #     n = number of time steps the oberved agent remains blocked
        if len(self.node_has_malfunction_agent) > 0:
            malfunctions = filter(lambda malfunction_val, distance: distance < agent_to_node_distance, self.node_has_malfunction_agent[graph_node])
            malfunctioning_agent = self.node_has_malfunction_agent[graph_node]
        #endregion
        #region #11:
        #     slowest observed speed of an agent in same direction
        #     1 if no agent is observed
        if len(self.node_has_slower_agent_speed_same_direction) > 0:
            min_fractional_speed = self.node_has_slower_agent_speed_same_direction[graph_node]
        #     min_fractional speed otherwise
        #endregion
        #region #12:
        #     number of agents ready to depart but no yet active
        if len(self.node_has_agents_ready_to_depart) > 0:
            other_agent_ready_to_depart_encountered = self.node_has_agents_ready_to_depart.get(graph_node, 0)
        #endregion
        #region #13:
        #    deadlock present in this node (if the current train will reach this node, it will cause a deadlock)
        #   can be true or false (0 or 1)
        out_edges = self.map_graph.graph.out_edges(graph_node, data=True)
        deadlock = 1
        # if a train is in current node, there is deadlock
        if other_agent_opposite_direction == 0:
            #check connected nodes, if all of them have a train, it is a deadlock
            x_switch, y_switch, direction_switch = graph_node
            position_switch = (x_switch, y_switch)
            iherent_directions = self.get_deadlock_inherent_directions(position_switch, direction_switch, [0, 0, 0 ,0])
            for direction in Grid4TransitionsEnum:
                # get_new_position(position, direction)
                # xxx
                if direction!= (direction_switch + 2) % 4 and iherent_directions[direction]:
                    new_position = get_new_position(position_switch, direction)
                    if (new_position, direction) not in self.map_graph.cell_connected_to_node:
                        raise Exception("this should not be a possibl situation")
                    next_node, _ = self.map_graph.cell_connected_to_node[(new_position, direction)]
                    #must check that the agent is not "blocking itself" and the blocking agent are "other agents"
                    other_agent_found = False
                    if next_node in self.node_has_agent_coming_from_switch:
                        for other_handle, _ in self.node_has_agent_coming_from_switch[next_node]:
                            if other_handle != handle:
                                other_agent_found = True
                                continue
                        if not other_agent_found:
                            deadlock = 0
                            continue
                    else:
                        deadlock = 0
                        continue
        #TODO: add new features to normalize_observation


       
        # TreeObsForRailEnv.Node
        node = Node(dist_own_target_encountered=own_target_encountered,
                                      dist_other_target_encountered=other_target_encountered,
                                      dist_other_agent_encountered=other_agent_encountered,
                                      dist_potential_conflict=potential_conflict,
                                      dist_unusable_switch=unusable_switch,
                                      dist_to_next_branch=dist_to_next_branch,
                                      dist_min_to_target=dist_min_to_target,
                                      num_agents_same_direction=other_agent_same_direction,
                                      num_agents_opposite_direction=other_agent_opposite_direction,
                                    #   num_agents_opposite_direction=deadlock,
                                      num_agents_malfunctioning=malfunctioning_agent,
                                      speed_min_fractional=min_fractional_speed,
                                      num_agents_ready_to_depart=other_agent_ready_to_depart_encountered,
                                      is_deadlock=deadlock,
                                      childs={})

        # #############################
        # #############################
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right, back], relative to the current orientation
        # Get the possible transitions
        visited.add(graph_node)
        
        node.childs[self.tree_explored_actions_char[0]] = -np.inf
        node.childs[self.tree_explored_actions_char[1]] = -np.inf
        node.childs[self.tree_explored_actions_char[2]] = -np.inf
        node.childs[self.tree_explored_actions_char[3]] = -np.inf

        out_edges = self.map_graph.graph.out_edges(graph_node, data=True)
        if len(out_edges) > 0:
            for (in_node_row, in_node_col, in_node_direction),(out_node_row, out_node_col, out_node_direction), c in out_edges:
                
                direction = self._cardinal_to_action(in_node_direction, c['direction'])
                # possible_transitions = self.env.rail.get_transitions(*(in_node_row, in_node_col), direction)
                branch_observation, branch_visited = self._explore_branch(handle,c['direction'], (out_node_row, out_node_col, out_node_direction), tot_dist_next, depth+1, tot_dist_next)
                # branch_observation, branch_visited = self._explore_branch(handle, (out_node_row, out_node_col, out_node_direction), tot_dist_next, depth+1, tot_dist_next)
                node.childs[self.tree_explored_actions_char[direction]] = branch_observation
                if len(branch_visited) != 0:
                    visited |= branch_visited
                      
        if depth == self.max_depth:
            node.childs.clear()
        return node, visited
        # possible_transitions = self.env.rail.get_transitions(*position, direction)
        # for i, branch_direction in enumerate([(direction + 4 + i) % 4 for i in range(-1, 3)]):
        #     if last_is_dead_end and self.env.rail.get_transition((*position, direction),
        #                                                          (branch_direction + 2) % 4):
        #         # Swap forward and back in case of dead-end, so that an agent can learn that going forward takes
        #         # it back
        #         new_cell = get_new_position(position, (branch_direction + 2) % 4)
        #         branch_observation, branch_visited = self._explore_branch(handle,
        #                                                                   new_cell,
        #                                                                   (branch_direction + 2) % 4,
        #                                                                   tot_dist + 1,
        #                                                                   depth + 1)
        #         node.childs[self.tree_explored_actions_char[i]] = branch_observation
        #         if len(branch_visited) != 0:
        #             visited |= branch_visited
        #     elif last_is_switch and possible_transitions[branch_direction]:
        #         new_cell = get_new_position(position, branch_direction)
        #         branch_observation, branch_visited = self._explore_branch(handle,
        #                                                                   new_cell,
        #                                                                   branch_direction,
        #                                                                   tot_dist + 1,
        #                                                                   depth + 1)
        #         node.childs[self.tree_explored_actions_char[i]] = branch_observation
        #         if len(branch_visited) != 0:
        #             visited |= branch_visited
        #     else:
        #         # no exploring possible, add just cells with infinity
        #         node.childs[self.tree_explored_actions_char[i]] = -np.inf

        # if depth == self.max_depth:
        #     node.childs.clear()
        # return node, visited

    def get_deadlock_inherent_directions(self, switch_position, direction, inherent_directions):
        """
        Given the position of a switch and an initial direction, 
        gives all directions that interests the trains going from the given direction
        Example: if there is a crossing and the train is going north, all directions are connected to rails,
         but only trins coming from north or south could interest our train
        """
        possible_transitions = self.env.rail.get_transitions(*switch_position, direction)
        for new_direction in Grid4TransitionsEnum:
            if possible_transitions[new_direction]:
                if not inherent_directions[new_direction]:
                    inherent_directions[new_direction] = 1
                    inherent_directions = self.get_deadlock_inherent_directions(switch_position, (new_direction + 2) % 4 , inherent_directions)
        return inherent_directions
    
    def util_print_obs_subtree(self, tree: Node):
        """
        Utility function to print tree observations returned by this object.
        """
        self.print_node_features(tree, "root", "")
        for direction in self.tree_explored_actions_char:
            self.print_subtree(tree.childs[direction], direction, "\t")

    @staticmethod
    def print_node_features(node: Node, label, indent):
        print(indent, "Direction ", label, ": ", node.dist_own_target_encountered, ", ",
              node.dist_other_target_encountered, ", ", node.dist_other_agent_encountered, ", ",
              node.dist_potential_conflict, ", ", node.dist_unusable_switch, ", ", node.dist_to_next_branch, ", ",
              node.dist_min_to_target, ", ", node.num_agents_same_direction, ", ", node.num_agents_opposite_direction,
              ", ", node.num_agents_malfunctioning, ", ", node.speed_min_fractional, ", ",
              node.num_agents_ready_to_depart)

    def print_subtree(self, node, label, indent):
        if node == -np.inf or not node:
            print(indent, "Direction ", label, ": -np.inf")
            return

        self.print_node_features(node, label, indent)

        if not node.childs:
            return

        for direction in self.tree_explored_actions_char:
            self.print_subtree(node.childs[direction], direction, indent + "\t")

    def set_env(self, env: Environment):
        super().set_env(env)
        if self.predictor:
            self.predictor.set_env(self.env)

    def _reverse_dir(self, direction):
        return int((direction + 2) % 4)

    #TODO: how to improve?
    def _cardinal_to_action(self, in_dir, out_dir):
        if in_dir == out_dir:
            return 1
        elif (in_dir + 1) % 4 == out_dir:
            return 2
        elif (in_dir + 2) % 4 == out_dir:
            return 3
        else:
            return 0
#region GlobalsObsForRailEnv
class GlobalObsForRailEnv(ObservationBuilder):
    """
    Gives a global observation of the entire rail environment.
    The observation is composed of the following elements:

        - transition map array with dimensions (env.height, env.width, 16),\
          assuming 16 bits encoding of transitions.

        - obs_agents_state: A 3D array (map_height, map_width, 5) with
            - first channel containing the agents position and direction
            - second channel containing the other agents positions and direction
            - third channel containing agent/other agent malfunctions
            - fourth channel containing agent/other agent fractional speeds
            - fifth channel containing number of other agents ready to depart

        - obs_targets: Two 2D arrays (map_height, map_width, 2) containing respectively the position of the given agent\
         target and the positions of the other agents targets (flag only, no counter!).
    """

    def __init__(self):
        super(GlobalObsForRailEnv, self).__init__()

    def set_env(self, env: Environment):
        super().set_env(env)

    def reset(self):
        self.rail_obs = np.zeros((self.env.height, self.env.width, 16))
        for i in range(self.rail_obs.shape[0]):
            for j in range(self.rail_obs.shape[1]):
                bitlist = [int(digit) for digit in bin(self.env.rail.get_full_transitions(i, j))[2:]]
                bitlist = [0] * (16 - len(bitlist)) + bitlist
                self.rail_obs[i, j] = np.array(bitlist)

    def get(self, handle: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        agent = self.env.agents[handle]
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            agent_virtual_position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            agent_virtual_position = agent.target
        else:
            return None

        obs_targets = np.zeros((self.env.height, self.env.width, 2))
        obs_agents_state = np.zeros((self.env.height, self.env.width, 5)) - 1

        # TODO can we do this more elegantly?
        # for r in range(self.env.height):
        #     for c in range(self.env.width):
        #         obs_agents_state[(r, c)][4] = 0
        obs_agents_state[:, :, 4] = 0

        obs_agents_state[agent_virtual_position][0] = agent.direction
        obs_targets[agent.target][0] = 1

        for i in range(len(self.env.agents)):
            other_agent: EnvAgent = self.env.agents[i]

            # ignore other agents not in the grid any more
            if other_agent.status == RailAgentStatus.DONE_REMOVED:
                continue

            obs_targets[other_agent.target][1] = 1

            # second to fourth channel only if in the grid
            if other_agent.position is not None:
                # second channel only for other agents
                if i != handle:
                    obs_agents_state[other_agent.position][1] = other_agent.direction
                obs_agents_state[other_agent.position][2] = other_agent.malfunction_data['malfunction']
                obs_agents_state[other_agent.position][3] = other_agent.speed_data['speed']
            # fifth channel: all ready to depart on this position
            if other_agent.status == RailAgentStatus.READY_TO_DEPART:
                obs_agents_state[other_agent.initial_position][4] += 1
        return self.rail_obs, obs_agents_state, obs_targets
#endregion

#region LocalsObsForRailEnv
class LocalObsForRailEnv(ObservationBuilder):
    """
    !!!!!!WARNING!!! THIS IS DEPRACTED AND NOT UPDATED TO FLATLAND 2.0!!!!!
    Gives a local observation of the rail environment around the agent.
    The observation is composed of the following elements:

        - transition map array of the local environment around the given agent, \
          with dimensions (view_height,2*view_width+1, 16), \
          assuming 16 bits encoding of transitions.

        - Two 2D arrays (view_height,2*view_width+1, 2) containing respectively, \
        if they are in the agent's vision range, its target position, the positions of the other targets.

        - A 2D array (view_height,2*view_width+1, 4) containing the one hot encoding of directions \
          of the other agents at their position coordinates, if they are in the agent's vision range.

        - A 4 elements array with one hot encoding of the direction.

    Use the parameters view_width and view_height to define the rectangular view of the agent.
    The center parameters moves the agent along the height axis of this rectangle. If it is 0 the agent only has
    observation in front of it.

    .. deprecated:: 2.0.0
    """

    def __init__(self, view_width, view_height, center):

        super(LocalObsForRailEnv, self).__init__()
        self.view_width = view_width
        self.view_height = view_height
        self.center = center
        self.max_padding = max(self.view_width, self.view_height - self.center)

    def reset(self):
        # We build the transition map with a view_radius empty cells expansion on each side.
        # This helps to collect the local transition map view when the agent is close to a border.
        self.max_padding = max(self.view_width, self.view_height)
        self.rail_obs = np.zeros((self.env.height,
                                  self.env.width, 16))
        for i in range(self.env.height):
            for j in range(self.env.width):
                bitlist = [int(digit) for digit in bin(self.env.rail.get_full_transitions(i, j))[2:]]
                bitlist = [0] * (16 - len(bitlist)) + bitlist
                self.rail_obs[i, j] = np.array(bitlist)

    def get(self, handle: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        agents = self.env.agents
        agent = agents[handle]

        # Correct agents position for padding
        # agent_rel_pos[0] = agent.position[0] + self.max_padding
        # agent_rel_pos[1] = agent.position[1] + self.max_padding

        # Collect visible cells as set to be plotted
        visited, rel_coords = self.field_of_view(agent.position, agent.direction, )
        local_rail_obs = None

        # Add the visible cells to the observed cells
        self.env.dev_obs_dict[handle] = set(visited)

        # Locate observed agents and their coresponding targets
        local_rail_obs = np.zeros((self.view_height, 2 * self.view_width + 1, 16))
        obs_map_state = np.zeros((self.view_height, 2 * self.view_width + 1, 2))
        obs_other_agents_state = np.zeros((self.view_height, 2 * self.view_width + 1, 4))
        _idx = 0
        for pos in visited:
            curr_rel_coord = rel_coords[_idx]
            local_rail_obs[curr_rel_coord[0], curr_rel_coord[1], :] = self.rail_obs[pos[0], pos[1], :]
            if pos == agent.target:
                obs_map_state[curr_rel_coord[0], curr_rel_coord[1], 0] = 1
            else:
                for tmp_agent in agents:
                    if pos == tmp_agent.target:
                        obs_map_state[curr_rel_coord[0], curr_rel_coord[1], 1] = 1
            if pos != agent.position:
                for tmp_agent in agents:
                    if pos == tmp_agent.position:
                        obs_other_agents_state[curr_rel_coord[0], curr_rel_coord[1], :] = np.identity(4)[
                            tmp_agent.direction]

            _idx += 1

        direction = np.identity(4)[agent.direction]
        return local_rail_obs, obs_map_state, obs_other_agents_state, direction

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[
        int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Called whenever an observation has to be computed for the `env` environment, for each agent with handle
        in the `handles` list.
        """

        return super().get_many(handles)

    def field_of_view(self, position, direction, state=None):
        # Compute the local field of view for an agent in the environment
        data_collection = False
        if state is not None:
            temp_visible_data = np.zeros(shape=(self.view_height, 2 * self.view_width + 1, 16))
            data_collection = True
        if direction == 0:
            origin = (position[0] + self.center, position[1] - self.view_width)
        elif direction == 1:
            origin = (position[0] - self.view_width, position[1] - self.center)
        elif direction == 2:
            origin = (position[0] - self.center, position[1] + self.view_width)
        else:
            origin = (position[0] + self.view_width, position[1] + self.center)
        visible = list()
        rel_coords = list()
        for h in range(self.view_height):
            for w in range(2 * self.view_width + 1):
                if direction == 0:
                    if 0 <= origin[0] - h < self.env.height and 0 <= origin[1] + w < self.env.width:
                        visible.append((origin[0] - h, origin[1] + w))
                        rel_coords.append((h, w))
                    # if data_collection:
                    #    temp_visible_data[h, w, :] = state[origin[0] - h, origin[1] + w, :]
                elif direction == 1:
                    if 0 <= origin[0] + w < self.env.height and 0 <= origin[1] + h < self.env.width:
                        visible.append((origin[0] + w, origin[1] + h))
                        rel_coords.append((h, w))
                    # if data_collection:
                    #    temp_visible_data[h, w, :] = state[origin[0] + w, origin[1] + h, :]
                elif direction == 2:
                    if 0 <= origin[0] + h < self.env.height and 0 <= origin[1] - w < self.env.width:
                        visible.append((origin[0] + h, origin[1] - w))
                        rel_coords.append((h, w))
                    # if data_collection:
                    #    temp_visible_data[h, w, :] = state[origin[0] + h, origin[1] - w, :]
                else:
                    if 0 <= origin[0] - w < self.env.height and 0 <= origin[1] - h < self.env.width:
                        visible.append((origin[0] - w, origin[1] - h))
                        rel_coords.append((h, w))
                    # if data_collection:
                    #    temp_visible_data[h, w, :] = state[origin[0] - w, origin[1] - h, :]
        if data_collection:
            return temp_visible_data
        else:
            return visible, rel_coords
#endregion