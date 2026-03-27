
class GoalBasedAgent:
    def __init__(self, goal):
        self.goal = goal
    
    def formulate_goal(self,percept):
        if percept == self.goal:
            return "Goal Reached"
        return "Searching"
    
    def IDDFS(self, graph, percept, goal, max_depth=50):

        for depth in range(max_depth):
            print(f"Current Depth: {depth}")
            path = []
            if self.dls(graph, percept, goal, depth, path):
                path.reverse()
                print(f"Path: {path}")
                return True
        
        print(f"Maximum Depth Limit Reached - Path not found")
        return False
    
    def dls(self, graph, current_node, goal, depth, path):
        if depth == 0:
            return False
        if current_node == goal:
            path.append(current_node)
            return True
        if current_node not in graph:
            return False
        for neighbor in graph[current_node]:
            if neighbor not in path:
                if self.dls(graph, neighbor, goal, depth-1, path):
                    path.append(neighbor)
                    return True
        return False


    def act(self, percept, graph):
        goal_status = self.formulate_goal(percept)
        if goal_status == "Goal reached":
            return f"Goal {self.goal} found!"
        return self.IDDFS(graph, percept, self.goal)


class Environment:
    def __init__(self, graph):
        self.graph = graph
    def get_percept(self, node):
        return node
    
graph = {
    'R1' : ['R2', 'R4'],
    'R2' : ['R1', 'R3', 'R5'],
    'R3' : ['R2', 'R6'],
    'R4' : ['R1', 'R5', 'R7'],
    'R5' : ['R2', 'R4', 'R6', 'R8'],
    'R7' : ['R4', 'R8'],
    'R8' : ['R5', 'R7', 'R9'],
    'R9' : ['R6', 'R8'],
}
start_node = 'R1'
goal_node = 'R8'
agent = GoalBasedAgent(goal_node)
environment = Environment(graph)

def run_agent(agent, environment=environment, start_node=start_node):
    percept = environment.get_percept(start_node)
    action = agent.act(percept, environment.graph)
    print(action)

run_agent(agent, environment, start_node)
