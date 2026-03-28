import heapq

graph = {
    'A': [(3, 'C'), (4, 'B')],
    'B': [(4, 'A'), (2, 'C'), (6,'E'), (3, 'D')],
    'C': [(3, 'A'), (2, 'B'), (1, 'D')],
    'D': [(3, 'F'), (1, 'C'), (3, 'B')],
    'E': [(6, 'B'), (4, 'I')],
    'F': [(3, 'D'), (2, 'H')],
    'H': [(2, 'F'), (4, 'I')],
    'I': [(4, 'H'), (4, 'E')],
}
start_node = 'A'
goal_node = 'I'

class GoalBasedAgent:
    def __init__(self, goal):
        self.goal = goal

    def formulate_goal(self, percept):
        if percept == self.goal:
            return "Goal Reached"
        return "Searching"
    
    def get_path(self, current_node, came_from):
        path = []
        while current_node:
            path.append(current_node)
            current_node = came_from[current_node]
        path.reverse()
        return path
    
    def UCS(self, graph, start_node, goal_node):
        frontier = []
        heapq.heappush(frontier, (0, start_node))
        visited = set()
        cost_so_far = {
            start_node: 0
        }
        came_from = {
            start_node: None
        }
        while frontier:
            cost, current_node = heapq.heappop(frontier)
            if current_node in visited:
                continue
            visited.add(current_node)
            print(f"Visiting {current_node}, cost: {cost}")
            if current_node == goal_node:
                print(f"Goal Reached! Total Cost: {cost}")
                return self.get_path(current_node, came_from)
            for neighbor_cost, neighbor in graph[current_node]:
                if neighbor not in cost_so_far or cost + neighbor_cost < cost_so_far[neighbor]:
                    heapq.heappush(frontier, (cost + neighbor_cost, neighbor))
                    came_from[neighbor] = current_node
                    cost_so_far[neighbor] = cost + neighbor_cost
                    
                    
        print(f"Goal not found")
        return None
    
    def act(self, percept, graph):
        goal_status = self.formulate_goal(percept)
        if goal_status == "Goal Reached":
            return f"Goal {self.goal} found!"
        return self.UCS(graph, percept, self.goal)
    
class Environment:
    def __init__(self, graph):
        self.graph = graph
    def get_percept(self, node):
        return node

agent = GoalBasedAgent(goal_node)
environment = Environment(graph)

def run_agent(agent=agent, environment=environment, start_node=start_node):
    percept = environment.get_percept(start_node)
    action = agent.act(percept, graph)
    print(action)
run_agent(agent, environment, start_node)
