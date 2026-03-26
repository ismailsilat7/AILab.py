import random

MAX_POPULATION = 10
MAX_GENERATIONS = 50
MUTATION_RATE = 0.15

# population

def get_chromosome():
    bit_string = ""
    for i in range(5):
        bit = random.randint(0,1)
        bit_string += f"{bit}"
    return bit_string

def generate_population():
    population = []
    for i in range(MAX_POPULATION):
        population.append(get_chromosome())
    return population

# fitness func
def fitness_func(chromosome):
    num = 0
    for power in range(5):
        bit = int(chromosome[len(chromosome) - 1 - power])
        num += bit*pow(2, power)
    return num


# selection
def selection(population, fitness_values):
    sorted_population = sorted(zip(fitness_values, population), key=lambda x:x[0])
    selected = sorted_population[len(sorted_population)//2:]
    return [chromosome for _, chromosome in selected]

# crossover 
def crossover(p1, p2):
    child1 = ""
    for i in range(5):
        if random.random() > 0.5:
            child1 += p1[i]
        else:
            child1 += p2[i]
    return child1

# mutation
def mutation(chromosome):
    if random.random() < MUTATION_RATE:
        i = random.randint(0,4)
        chromosome = chromosome[:i] + ('0' if chromosome[i]=='1' else '1') + chromosome[i+1:]
    return chromosome

def genetic_algo():
    # population
    population = generate_population()
    for generation in range(MAX_GENERATIONS):
        #fitness
        fitness_values = [fitness_func(population[i]) for i in range(len(population))]
        best_fitness = max(fitness_values)
        print(f"Generation: {generation + 1}, Best Fitness: {best_fitness}")
        if best_fitness == 31:
            print("Max value found")
            break
        parents = selection(population, fitness_values)
        best_index = fitness_values.index(best_fitness)
        # crossover & mutation
        new_population = [population[best_index]]
        while len(new_population) < MAX_POPULATION:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2)
            child = mutation(child)
            new_population.append(child)
        population = new_population
    fitness_values = [fitness_func(population[i]) for i in range(len(population))]
    best_fitness = max(fitness_values)
    best_chromosome = population[fitness_values.index(best_fitness)]
    return best_chromosome

max_value = genetic_algo()
print(f"Best Max Value: {max_value}")

