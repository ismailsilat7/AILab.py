import random
# 5 teachers
# 5 courses
# 5 time slots
# 5 days per week

# 1. A teacher cannot teach two classes at the same time.
# 2. Each course must be scheduled exactly 3 times per week.
# 3. No teacher should teach more than 3 consecutive classes.

# 15 sessions <--- 5 courses * 3 sessions
teachers = ['T1', 'T2', 'T3', 'T4', 'T5']
courses = ['C1', 'C2', 'C3', 'C4', 'C5']
days = ['Mon', 'Tue', 'Wed', 'Thu', "Fri"]
slots = [1, 2, 3, 4, 5]

MAX_GENERATIONS = 200
POPULATION_SIZE = 20
MUTATION_RATE = 0.15

def create_chromosome():
    # 25 total slots in a week
    chromosome = [None] * 25 
    
    # Randomly assign 15 classes without guaranteeing exact course counts.
    # The GA will have to figure out how to balance them!
    indices = random.sample(range(25), 15)
    for i in indices:
        chromosome[i] = (random.choice(courses), random.choice(teachers))
        
    return chromosome

def population_initialization():
    population = []
    for i in range(POPULATION_SIZE):
        population.append(create_chromosome())
    return population

def fitness_func(chromosome):
    penalty = 0

    num_slots = 5
    num_days = 5

    # A teacher cannot teach two classes at the same time.
    # already handled - as fixed slots
    # assumes only 1 session per slot
    
    # No teacher should teach more than 3 consecutive classes.
    for teacher in teachers:
        for day in range(num_days):
            count = 0
            for slot in range(num_slots):
                index = day * num_slots + slot
                session = chromosome[index]
                if session and session[1] == teacher:
                    count += 1
                    if count > 3:
                        penalty += abs(count - 3) * 10
                else:
                    count = 0
            
    # Each course must be scheduled exactly 3 times per week.
    course_counts = {course: 0 for course in courses}
    for session in chromosome:
        if session:
            course_counts[session[0]] += 1
    for course, count in course_counts.items():
        if count != 3:
            penalty += 5

    return penalty

def selection(population, fitness_scores):
    sorted_population = [x for _,x in sorted(zip(fitness_scores, population), key=lambda x:x[0])]
    return sorted_population[:len(sorted_population)//2]

def crossover(p1, p2):
    child1 = []
    child2 = []

    for i in range(len(p1)):
        if random.random() > 0.5:
            child1.append(p1[i])
            child2.append(p2[i])
        else:
            child1.append(p2[i])
            child2.append(p1[i])
    return child1, child2

def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        i = random.randint(0,len(chromosome) - 1)
        chromosome[i], chromosome[len(chromosome)-1-i] = chromosome[len(chromosome)-1-i], chromosome[i]
    return chromosome

def genetic_algo():
    # Population
    population = population_initialization()
    for generation in range(MAX_GENERATIONS):
        # Selection
        fitness_scores = [fitness_func(c) for c in population]
        best_fitness = min(fitness_scores)
        print(f"Generation: {generation + 1}, Best Fitness: {best_fitness}")
        if best_fitness == 0:
            print("Perfect timetable found!")
            break
        parents = selection(population, fitness_scores)
        # Crossover & Mutation
        best_index = fitness_scores.index(best_fitness)
        new_population = [population[best_index]]
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = random.sample(parents, 2)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_population.append(c1)
            new_population.append(c2)
        population = new_population
    fitness_scores = [fitness_func(c) for c in population]
    best_fitness = min(fitness_scores)
    best_chromosome = population[fitness_scores.index(best_fitness)]
    return best_chromosome

best_timetable = genetic_algo()

for day in range(len(days)):
    print(f"Day {days[day]}: ")
    for slot in range(len(days)):
        index = len(days) * day + slot
        print(f"\tSlot {slots[slot]}: {best_timetable[index] if best_timetable[index] else "No class"}")
