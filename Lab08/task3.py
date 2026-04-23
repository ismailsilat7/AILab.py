import math

A = 1 # agent
B = 2 # blocker
G = 3 # goal
X = 4 # obstacle

N = 5 # NxN graph
graph = [
    [A,0,X,0,0],
    [0,0,0,0,0],
    [0,0,0,B,0],
    [0,0,0,X,0],
    [0,X,0,0,G]
]

class Node:
    def __init__(self):
        self.children = []
        self.minmax_value = None
    
class MinimaxAgent:
    def __init__(self, depth):
        self.depth = depth
    
    def formulate_goal(self, node):
        return "Goal reached" if node.minmax_value is not None else "Searching"
    
    def act(self, node, environment):
        goal_status = self.formulate_goal(node)
        if goal_status == "Goal reached":
            return f"Minimax value for root node: {node.minmax_value}"
        else:
            return environment.compute_minimax(node, self.depth)

class Environment:
    def __init__(self, graph=graph):
        self.graph = graph
        self.goal_node = (4,4)
        self.start_node = (0,0)
        self.blocker_node = (2,3)

    def get_percept(self, node):
        return node
    
    def get_children(self, node, blocker=False):
        children = []
        x,y = node
        moves = [(0,1), (1,0), (0,-1), (-1, 0)]
        for dx, dy in moves:
            cx = x + dx
            cy = y + dy
            if 0 <= cx < N and 0 <= cy < N and (
                (not blocker and self.graph[cx][cy] not in [X, B]) 
                or 
                (blocker and self.graph[cx][cy] not in [X, G, A])
            ):
                children.append((cx,cy))
        return children
    

    def get_heuristic(self, agent, blocker):
        ax, ay = agent
        gx, gy = self.goal_node
        bx, by = blocker
        goal_dist = abs(ax-gx) + abs(ay-gy)
        block_dist = abs(ax-bx) + abs(ay-by)
        return -goal_dist + block_dist
    
    def compute_minimax(self, state, depth, maximizing_player=True):
        agent, blocker = state

        if depth == 0:
            value = self.get_heuristic(agent, blocker)
            return value
        
        if agent == self.goal_node:
            return 10
        
        if maximizing_player:
            value = -math.inf
            children = self.get_children(agent)
            if not children:
                return -10
            for new_agent in children:
                child_state = (new_agent, blocker)
                child_value = self.compute_minimax(child_state, depth - 1, False)
                value = max(value, child_value)
            return value
        else:
            value = math.inf
            children = self.get_children(blocker, True)

            if not children:
                return self.get_heuristic(agent, blocker)

            for new_blocker in children:
                if new_blocker == agent:
                    return -10

                child_state = (agent, new_blocker)
                child_value = self.compute_minimax(child_state, depth - 1, True)
                value = min(value, child_value)

            return value
        

def run_agent(environment, depth):
    agent = environment.start_node
    blocker = environment.blocker_node

    for step in range(3):
        print("State", step + 1)
        print_grid(environment, agent, blocker)

        best_move = None
        best_value = -math.inf

        for move in environment.get_children(agent):
            value = environment.compute_minimax(
                (move, blocker),
                depth - 1,
                False
            )

            print("Possible move:", move, "Value:", value)

            if value > best_value:
                best_value = value
                best_move = move

        print("Agent's chosen move:", best_move)
        if agent == best_move:
            break
        agent = best_move

        worst_value = math.inf
        best_blocker = blocker

        for move in environment.get_children(blocker, True):
            if move == agent:
                best_blocker = move
                break

            value = environment.compute_minimax(
                (agent, move),
                depth - 1,
                True
            )

            if value < worst_value:
                worst_value = value
                best_blocker = move
        print("   Blocker moved to:", best_blocker)
        print()
        blocker = best_blocker

        if agent == (4,4):
            print("Goal reached!")
            break
        

def print_grid(environment, agent, blocker):
    for i in range(N):
        for j in range(N):
            if (i, j) == agent:
                print("A", end=" ")
            elif (i, j) == blocker:
                print("B", end=" ")
            elif (i, j) == environment.goal_node:
                print("G", end=" ")
            elif environment.graph[i][j] == X:
                print("X", end=" ")
            else:
                print(".", end=" ")
        print()
    print()

env = Environment()

print("-" * 10)
print("Depth = 2")
print("-" * 10)
run_agent(env, 2)
 
print("-" * 10)
print("Depth = 3")
print("-" * 10)
run_agent(env, 3)

