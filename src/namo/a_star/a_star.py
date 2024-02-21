import math
import numpy as np
from itertools import product


class Node:
    def __init__(self, parent=None, position=None):
        self.par = parent
        self.pos = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.pos == other.pos

    def __hash__(self):
        return hash(self.pos)


def eucledian_distance(pos, goal_pos):
    return math.sqrt((pos[0] - goal_pos[0]) ** 2 + (pos[1] - goal_pos[1]) ** 2)


def a_star(maze, init_pos, goal_pos):
    init_node = Node(None, init_pos)
    goal_node = Node(None, goal_pos)
    
    frontier = []
    explored = set()
    
    frontier.append(init_node)

    while frontier:
        current_node = min(frontier, key=lambda node: node.f)
        
        frontier.remove(current_node)
        explored.add(current_node)

        if current_node == goal_node:
            path = []

            while current_node:
                path.append(current_node.pos)
                current_node = current_node.par

            path = np.array(path)

            path[:, [0, 1]] = path[:, [1, 0]]
            return path[::-1], explored

        for move in product([-1, 0, 1], repeat=2):
            node_position = (current_node.pos[0] + move[0], current_node.pos[1] + move[1])

            if node_position == current_node.pos:
                continue
            
            if (0 <= node_position[1] < len(maze)) and (0 <= node_position[0] < len(maze[0])):
                if maze[node_position[1]][node_position[0]] == 0:  # Adjusted indexing for width and height
                    child_node = Node(current_node, node_position)

                    child_node.g = current_node.g + 1
                    child_node.h = eucledian_distance(current_node.pos, goal_node.pos)
                    child_node.f = child_node.g + child_node.h

                    if child_node in explored and child_node.g >= current_node.g:
                        continue

                    if child_node in frontier and child_node.g >= current_node.g:
                        continue

                    frontier.append(child_node)

    return None, explored
