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
    def __init__(self, grid):
        self.graph = nx.DiGraph()
        self.grid = grid
        self.grid_shape = grid.shape
        # self._create_graph()
    
        


    def _create_graph(self):
        pass

    def _explore_branch(self, all_in_nodes, out_node):
        
        

    def _find_first_rail(self):
        # Save the env's rail grid as a Networkx graph
        for row in range(self.grid_shape[0]):
            for column in range(self.grid_shape[1]):
                current_cell = self.grid[row, column]  # is a uint16
                if current_cell:
                    for direction in Grid4TransitionsEnum:
                        possible_transitions = env.rail.get_transitions(*current_cell, direction)
                        np.count
                    transitions = TransitionMatrix(current_cell)  # is a 4x4 matrix

                    self.graph.add_node((row, column), transitions=transitions)  # Add or update
                    for connection, increment in zip(transitions.exits, GRID_INCREMENTS):
                        if connection:
                            self.graph.add_edge((row, column), (row + increment[0], column + increment[1]))

        for row in range(self.grid_shape[0]):
            for column in range(self.grid_shape[1]):
                current_cell = self.grid[row, column]  # is a uint16
                if current_cell:
                    transitions = current_cell.format(current_cell, 'b')
                    total_transitions = np.count_nonzero(transitions)
                    if total_transitions > 2:
                        for i, bit in enumerate(transitions):
                              if direction[0]:
                                self.explore_branch(self, row, column, direction)
                              if direction[1]:
                                self.explore_branch(self, row, column, direction)
                              if direction[2]:
                                self.explore_branch(self, row, column, direction)
                              if direction[3]:
                                self.explore_branch(self, row, column, direction)



        def _explore_branch(self, row, column, direction):
            (y, x) = get_new_position([row, column], direction)
            next_cell = self.grid[y][x]
            transitions = next_cell.format(current_cell, 'b')
            total_transitions = np.count_nonzero(transitions)
            new_direction = get_direction(row, column, y,x)
            if total_transitions > 2:
                return self._convert_transitions(transitions)
            return self._explore_branch(y, x, new_direction)

        def _convert_transitions(self, cell):
            ...
            return array[] = (y, x ,direction)