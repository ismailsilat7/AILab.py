import random

MAX_POPULATION = 200
MAX_GENERATION = 200
MUTATION_RATE = 1.5

""" AI GENERATED PROBLEM DATA """
# The hidden data provided by your instructor
weights = [10, 20, 30, 40, 50, 15, 25, 35, 45, 55] # Weight of each package
values = [60, 100, 120, 150, 200, 75, 90, 130, 170, 210] # Value of each package
max_capacity = 150 # The truck cannot hold more than 150 total weight
# The number of items dictates the length of your chromosome!
num_items = len(weights) # In this case, 10

# population
def create_random_individual(num_items=num_items):
    string = ""
    for i in range(num_items):
        bit = random.randint(0,1)
        string += f"{bit}"
    return string
def initialize_population(pop_size=MAX_POPULATION, num_items=num_items):
    population = []
    for i in range(pop_size):
        population.append(create_random_individual(num_items))
    return population

# fitness
def fitness_func(individual):
    total_weight = 0
    total_value = 0

    for i in range(len(individual)):
        if individual[i] == '1':
            total_value += values[i]
            total_weight += weights[i]
    
    if total_weight > 150:
        return 0
    
    return total_value

# selection
def select_parents(population, fitness_scores):
    sorted_population = [x for _,x in sorted(zip(fitness_scores, population), key=lambda x:x[0])]
    return sorted_population[len(sorted_population)//2:]

# crossover
def cross_over(parent1, parent2):
    i = random.randint(0, len(parent1) - 1)
    child1 = parent1[:i] + parent2[i:]
    child2 = parent2[:i] + parent1[i:]
    return child1, child2

# mutation
def mutation(individual):
    if random.random() < MUTATION_RATE:
        index = random.randint(0, len(individual) - 1)
        individual = individual[:index] + ('0' if individual[index] == '1' else '1') + individual[index+1:]
    return individual

def genetic_algo(popsize=MAX_POPULATION, num_items=num_items):
    population = initialize_population()
    for generation in range(MAX_POPULATION):
        fitness_scores = [fitness_func(population[i]) for i in range(len(population))]
        best_fitness = max(fitness_scores)
        print(f"Generation: {generation + 1}, Best Fitness: {best_fitness}")

        # no predetermined best value so cant converge on basis of that

        parents = select_parents(population, fitness_scores)
        best_index = fitness_scores.index(best_fitness)
        new_population = [population[best_index]]
        while len(new_population) < popsize:
            p1, p2 = random.sample(parents, 2)
            child1, child2 = cross_over(p1, p2)
            child1 = mutation(child1)
            child2 = mutation(child2)
            new_population.append(child1)
            new_population.append(child2)
        population = new_population
    fitness_scores = [fitness_func(population[i]) for i in range(len(population))]
    best_fitness = max(fitness_scores)
    best_individiual = population[fitness_scores.index(best_fitness)]
    return best_individiual

def print_solution(individual):
    print("Selected Packages:")
    
    total_weight = 0
    total_value = 0
    
    for i in range(len(individual)):
        if individual[i] == '1':
            print(f"Item {i+1}: Weight = {weights[i]}, Value = {values[i]}")
            total_weight += weights[i]
            total_value += values[i]
    
    print("Summary:")
    print(f"Total Weight: {total_weight} / {max_capacity}")
    print(f"Total Value : {total_value}")
    print(f"Chromosome  : {individual}")

most_valuable_set = genetic_algo()
print_solution(most_valuable_set)