


class RTTStarPlanner:
    def __init__(self, cost_map: SemCostMap):
        self.cost_map = cost_map
        self.path = []

    def plan_trajectory(self, start: Tuple[float, float, float], goal: Tuple[float, float, float]):
        self.start = start
        self.goal = goal
        
        # Step 1: Create the cost map
        grid_loss, points, height_map = self.cost_map.create_costmap()
        
        # Step 2: Implement RTT* planning algorithm
        self.path = self.rtt_star(start, goal, grid_loss)

    def rtt_star(self, start, goal, grid_loss):
        # 这里实现RTT*算法
        open_set = []  # 优先队列
        closed_set = set()

        # 将起始点加入开放集合
        open_set.append((start, 0))  # (坐标, 当前成本)

        while open_set:
            current = self.get_lowest_cost_node(open_set)
            if self.reached_goal(current, goal):
                return self.reconstruct_path(current)

            open_set.remove(current)
            closed_set.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                # 计算成本
                cost = self.calculate_cost(neighbor, grid_loss)

                # 如果是新的节点，或者新的路径更优
                if neighbor not in open_set or cost < self.get_cost(neighbor):
                    open_set.append((neighbor, cost))

        return []

    def get_neighbors(self, node):
        # 返回邻近节点
        pass

    def calculate_cost(self, node, grid_loss):
        # 计算从节点到目标的成本
        x, y = node[:2]
        return grid_loss[int(x)][int(y)]  # 假设grid_loss是个二维数组

    def get_lowest_cost_node(self, open_set):
        # 获取开放集合中成本最低的节点
        return min(open_set, key=lambda x: x[1])

    def reached_goal(self, current, goal):
        # 判断是否到达目标
        return np.linalg.norm(np.array(current) - np.array(goal)) < 1.0

    def reconstruct_path(self, current):
        # 重建路径
        path = []
        while current:
            path.append(current)
            current = current.parent  # 假设有parent属性指向上一个节点
        return path[::-1]  # 返回正向路径
