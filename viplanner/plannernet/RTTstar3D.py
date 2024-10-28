from typing import List
import torch
from traj_cost_opt import TrajCost
torch.set_default_dtype(torch.float32)

class Node:
    def __init__(self, position: torch.Tensor, parent=None):
        self.position = position
        self.parent = parent

    def __hash__(self):
        return hash(tuple(self.position.numpy()))

    def __eq__(self, other):
        return torch.equal(self.position, other.position)

class RTTStarPlanner:
    def __init__(self, traj_cost: TrajCost):
        self.traj_cost = traj_cost
        self.paths = []

    def plan_trajectories(self, starts: torch.Tensor, goals: torch.Tensor):
        batch_size = start.shape[0]
        self.paths = []
        for i in range(batch_size):
        
            start = Node(starts[i])
            goal = Node(goals[i])
            path = self.rtt_star(start,goal)
            self.paths.append(path)
        
        return self.paths
            

    def rtt_star(self, start: Node, goal: Node) -> List[Node]:
        open_set = [start]  # 优先队列
        closed_set = set()

        while open_set:
            current = self.get_lowest_cost_node(open_set, goal)
            if self.reached_goal(current, goal):
                return self.reconstruct_path(current)

            open_set.remove(current)
            closed_set.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                # 计算成本
                cost = self.calculate_cost(neighbor.position)

                # 如果是新的节点，或者新的路径更优
                if neighbor not in open_set:
                    open_set.append(neighbor)
                else:
                    # 在已有节点中更新成本
                    existing_cost = self.calculate_cost(neighbor.position)
                    if cost < existing_cost:
                        open_set.remove(neighbor)
                        open_set.append(neighbor)

        return []

    def get_neighbors(self, node: Node) -> List[Node]:
        neighbors = []
        x, y, z = node.position
        directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        
        for dx, dy, dz in directions:
            neighbor_position = torch.tensor([x + dx, y + dy, z + dz])
            neighbors.append(Node(neighbor_position, parent=node))

        return neighbors

    def calculate_cost(self, position: torch.Tensor) -> float:
        return self.traj_cost._compute_oloss(position, 1).item()  # 确保返回float

    def get_lowest_cost_node(self, open_set: List[Node], goal: Node) -> Node:
        return min(open_set, key=lambda node: self.calculate_cost(node.position) + torch.norm(node.position - goal.position))

    def reached_goal(self, current: Node, goal: Node) -> bool:
        return torch.norm(current.position - goal.position) < 1.0

    def reconstruct_path(self, current: Node) -> List[torch.Tensor]:
        path = []
        while current:
            path.append(current.position)  # 直接使用torch.Tensor
            current = current.parent  # 通过parent属性指向上一个节点
        return path[::-1]  # 返回正向路径
