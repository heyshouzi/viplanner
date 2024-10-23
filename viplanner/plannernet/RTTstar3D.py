from cost_maps import SemCostMap
from typing import Tuple, List
import numpy as np

class Node:
    def __init__(self, position: Tuple[float, float, float], parent=None):
        self.position = position
        self.parent = parent

class RTTStarPlanner:
    def __init__(self, cost_map: SemCostMap):
        self.cost_map = cost_map
        self.path = []

    def plan_trajectory(self, start: Tuple[float, float, float], goal: Tuple[float, float, float]):
        self.start = Node(start)
        self.goal = Node(goal)
        
        # Step 1: Create the cost map
        grid_loss, points, height_map = self.cost_map.create_costmap()
        
        # Step 2: Implement RTT* planning algorithm
        self.path = self.rtt_star(self.start, self.goal, grid_loss)

    def rtt_star(self, start: Node, goal: Node, grid_loss: np.ndarray) -> List[Node]:
        open_set = [start]  # 优先队列
        closed_set = set()

        while open_set:
            current = self.get_lowest_cost_node(open_set, goal, grid_loss)
            if self.reached_goal(current, goal):
                return self.reconstruct_path(current)

            open_set.remove(current)
            closed_set.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                # 计算成本
                cost = self.calculate_cost(neighbor.position, grid_loss)

                # 如果是新的节点，或者新的路径更优
                if neighbor not in open_set or cost < self.get_cost(neighbor):
                    open_set.append(neighbor)

        return []

    def get_neighbors(self, node: Node) -> List[Node]:
        neighbors = []
        x, y, z = node.position
        directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        
        for dx, dy, dz in directions:
            neighbor_position = (x + dx, y + dy, z + dz)
            neighbors.append(Node(neighbor_position, parent=node))

        return neighbors

    def calculate_cost(self, position: Tuple[float, float, float], grid_loss: np.ndarray) -> float:
        x, y = int(position[0]), int(position[1])
        return grid_loss[x][y]  # 假设grid_loss是个二维数组

    def get_lowest_cost_node(self, open_set: List[Node], goal: Node, grid_loss: np.ndarray) -> Node:
        return min(open_set, key=lambda node: self.calculate_cost(node.position, grid_loss) + np.linalg.norm(np.array(node.position) - np.array(goal.position)))

    def reached_goal(self, current: Node, goal: Node) -> bool:
        return np.linalg.norm(np.array(current.position) - np.array(goal.position)) < 1.0

    def reconstruct_path(self, current: Node) -> List[Tuple[float, float, float]]:
        path = []
        while current:
            path.append(current.position)
            current = current.parent  # 通过parent属性指向上一个节点
        return path[::-1]  # 返回正向路径
