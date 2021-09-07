from flatland.core.grid.grid4_utils import get_direction, get_new_position
from flatland.envs.rail_env import RailEnv
from flatland.core.grid.grid4 import Grid4TransitionsEnum
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
class Graph:
    #instantiate this object at each episode, to read the whole map
    def __init__(self,env):
        self.graph = nx.DiGraph()
        self.env = env
        self.cell_connected_to_node = dict()
        self.create_graph()
        
    def create_graph(self):
        # Save the env's rail grid as a Networkx graph
        
        grid = self.env.rail.grid
        grid_shape = grid.shape
        for row in range(grid_shape[0]):
            for column in range(grid_shape[1]):
                current_cell = grid[row, column]  # is a uint16
                if current_cell:
                    # transition = boolean array
                    # NOTE: debug only
                    if current_cell == 20994:
                        print()
                    transitions = bin(current_cell)
                    total_transitions = transitions.count('1')
                    #if it's a swtich
                    if total_transitions > 2:

                        # for i,bit in enumerate(transitions):
                        #     if bit == '1':
                        #         in_direction = int(i/4)
                        #         out_direction = i%4
                        #         if out_direction in already_explored:
                        #         (next_row, next_column, next_direction, distance, path) = self._explore_branch(row, column, out_direction, 1, [])
                        #         for cell_y, cell_x, cell_direction, distance_from_previous_node in path:
                        #                     self.cell_connected_to_node[((cell_y, cell_x), cell_direction)] = ((next_row, next_column, int(next_direction)), distance - distance_from_previous_node)
                                
                        #         self.graph.add_edge((row, column, int(in_direction)), (next_row, next_column, int(next_direction)), direction = int(out_direction), distance = distance)
                        #         already_explored.append(out_direction)
                        
                        # direction where I came from
                        for direction in Grid4TransitionsEnum:
                            # a node is identified by row, column and direction
                            possible_transitions = self.env.rail.get_transitions(*[row,column], direction)
                            # direction where I'm going
                            if max(possible_transitions) != 0:
                                for direction_transition in Grid4TransitionsEnum:
                                    if possible_transitions[direction_transition]:
                                        # if (row, column, direction_transition) not in already_explored:
                                            # next_direction = direction where I'll be coming from the next node
                                            # (next_row, next_column, next_direction, distance) = already_explored[(row, column, direction_transition)]
                                        (next_row, next_column, next_direction, distance, path) = self._explore_branch(row, column, direction_transition, 1, [])
                                        self.cell_connected_to_node[((row, column), int(direction))] = ((row, column, int(direction)), 0)
                                        for cell_y, cell_x, cell_direction, distance_from_previous_node in path:
                                            self.cell_connected_to_node[((cell_y, cell_x), cell_direction)] = ((next_row, next_column, int(next_direction)), distance - distance_from_previous_node)
                                        self.graph.add_edge((row, column, int(direction)), (next_row, next_column, int(next_direction)), direction = int(direction_transition), distance = distance)                                            

        # self._draw()
                                    
                              

    # explore the branch cell by cell
    def _explore_branch(self, row, column, direction, distance, path):
        (y,x) = get_new_position([row, column], direction)
        next_cell = self.env.rail.grid[y][x]
        transitions = bin(next_cell)
        total_transitions = transitions.count('1')
        # if it's a switch
        if total_transitions > 2:
            return (y, x, direction, distance + 1, path)
        possible_transitions = self.env.rail.get_transitions(*[y,x], direction)
        new_direction = np.argmax(possible_transitions)
        path.append((y, x, direction, distance))
        return self._explore_branch(y, x, new_direction, distance+1, path)

    def _draw(self):
        pos = nx.spring_layout(self.graph, seed = 1)
        nx.draw(self.graph, pos, with_labels=True)
        direction_labels = nx.get_edge_attributes(self.graph,'direction')
        distance_labels = nx.get_edge_attributes(self.graph,'distance')
        # nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=direction_labels)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=distance_labels)
        plt.show()