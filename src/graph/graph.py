from tkinter.constants import NO
from attr import dataclass
from flatland.core.grid.grid4_utils import get_direction, get_new_position
from flatland.envs.rail_env import RailEnv
from flatland.core.grid.grid4 import Grid4TransitionsEnum
import networkx as nx
import numpy as np
from pyglet.window.key import D
class Graph:
    #instantiate this object at each episode, to read the whole map
    def __init__(self,env, grid):
        self.graph = nx.DiGraph()
        self.grid = grid
        self.grid_shape = grid.shape
        self.env = env
        # self._create_graph()
    

    def _explore_branch(self, all_in_nodes, out_node):
        pass
        

    def create_graph(self):
        # Save the env's rail grid as a Networkx graph

        for row in range(self.grid_shape[0]):
            for column in range(self.grid_shape[1]):
                current_cell = self.grid[row, column]  # is a uint16
                if current_cell:
                    # transition = boolean array
                    transitions = current_cell.format(current_cell, 'b')
                    total_transitions = np.count_nonzero(transitions)
                    #if it's a swtich
                    if total_transitions > 2:
                        #contains the next switch position and direction, relative to the current position and direction
                        already_explored = {}

                        # direction where I came from
                        for direction in Grid4TransitionsEnum:
                            # a node is identified by row, column and direction
                            possible_transitions = self.env.rail.get_transitions(*[row,column], direction)
                            
                            # direction where I'm going
                            for direction_transition in Grid4TransitionsEnum:
                                if possible_transitions[direction_transition]:
                                    if (row, column, direction_transition) in already_explored:
                                        # next_direction = direction where I'll be coming from the next node
                                        (next_row, next_column, next_direction, distance) = already_explored[(row, column, direction)]
                                    else:
                                        (next_row, next_column, next_direction, distance) = self.explore_branch(self, row, column, direction, 0)
                                    self.add_edge((row, column, direction), (next_row, next_column, next_direction), direction = direction_transition, distance = distance)
                                    
                              

    # explore the branch cell by cell
    def _explore_branch(self, row, column, direction, distance):
        (y,x) = get_new_position([row, column], direction)
        next_cell = self.grid[y][x]
        transitions = next_cell.format(next_cell, 'b')
        total_transitions = np.count_nonzero(transitions)
        # if it's a switch
        if total_transitions > 2:
            return (y, x, direction, distance + 1)
        possible_transitions = self.env.rail.get_transitions(*[y,x], direction)
        new_direction = np.argmax(possible_transitions)
        return self._explore_branch(y, x, new_direction, distance+1)
