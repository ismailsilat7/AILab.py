import math

class Node:
    def __init__(self, value=None):
        self.value = value
        self.children = []
        self.minmax_value = None

class MinimaxAgent:
    def __init__(self, depth):
        self.depth = depth

    def formulate_goal(self, node):
        # In Minimax, the goal is to compute the minimax value for the root node
        return "Goal reached" if node.minmax_value is not None else "Searching"

    def act(self, node, environment):
        goal_status = self.formulate_goal(node)

        if goal_status == "Goal reached":
            return f"Minimax value for root node: {node.minmax_value}"
        else:
            return environment.compute_minimax(node, self.depth)

class Environment:
    def __init__(self, tree):
        self.tree = tree
        self.computed_nodes = []

    def get_percept(self, node):
        return node

    def compute_minimax(self, node, depth, maximizing_player=True):
        if depth == 0 or not node.children:
            self.computed_nodes.append(node.value)
            return node.value

        if maximizing_player:
            value = -math.inf

            for child in node.children:
                child_value = self.compute_minimax(child, depth - 1, False)
                value = max(value, child_value)

            node.minmax_value = value
            print(f"Backtracking at MAX node {node.value}: chooses {value}")
            self.computed_nodes.append(node.value)
            return value

        else:
            value = math.inf

            for child in node.children:
                child_value = self.compute_minimax(child, depth - 1, True)
                value = min(value, child_value)

            node.minmax_value = value
            print(f"Backtracking at MIN node {node.value}: chooses {value}")
            self.computed_nodes.append(node.value)
            return value


def run_agent(agent, environment, start_node):
    percept = environment.get_percept(start_node)
    agent.act(percept, environment)

# Sample tree
root = Node('A')
n1 = Node('B')
n2 = Node('C')
root.children = [n1, n2]

n3 = Node('D')
n4 = Node('E')
n5 = Node('F')
n6 = Node('G')
n1.children = [n3, n4]
n2.children = [n5, n6]

n7 = Node(3)
n8 = Node(5)
n9 = Node(6)
n10 = Node(9)
n3.children = [n7, n8]
n4.children = [n9, n10]

n11 = Node(1)
n12 = Node(2)
n13 = Node(0)
n14 = Node(-1)
n5.children = [n11, n12]
n6.children = [n13, n14]


# Define depth for Minimax
depth = 3
agent = MinimaxAgent(depth)
environment = Environment(root)

run_agent(agent, environment, root)


print("Computed Nodes:", environment.computed_nodes)
print("Minimax values:")
print("A:", root.minmax_value)
print("B:", n1.minmax_value)
print("C:", n2.minmax_value)
print("D:", n3.minmax_value)
print("E:", n4.minmax_value)
print("F:", n5.minmax_value)
print("G:", n6.minmax_value)