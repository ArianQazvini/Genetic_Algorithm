import math
from enum import Enum
from game import Game
import random
import matplotlib.pyplot as plt
import numpy as np
class SelectionType(Enum):
    ONLY_BEST = 1
    ROULETTE = 2
class CrossOver(Enum):
    SINGLE_POINT = 1
    TWO_POINT = 2
class GeneticAlgorithm:
    def __init__(self,initial_population_number,game, win_reward_allowed ,selection_type,crossover_type,mutation_prob):
        self.initial_population_number = initial_population_number
        self.game = game
        self.game.load_next_level()
        self.chromosome_length = game.current_level_len
        self.win_reward_allowed = win_reward_allowed
        self.selection_type = selection_type
        self.crossover_type = crossover_type
        self.mutation_prob= mutation_prob
        # self.selection_size =selection_size
        self.population = []
        self.parents = []
        self.children = []
        self.next_gen=[]

    def create_initial_population(self):
        for i in range(self.initial_population_number):
            chromosome = ""
            for j in range(self.chromosome_length):
                chromosome+= str(random.randint(0,2))
            self.population.append(chromosome)
        self.fix(self.population)

    def fix(self, arr):
        for i in range(len(arr)):
            # temp = list(arr[i])
            for j in range(len(arr[i])-1):
                if(arr[i][j] == "1" and (arr[i][j+1] == "1" or arr[i][j+1]=='2')):
                   newchar = "0"
                   index = j +random.randint(0,1)
                   string = arr[i]
                   arr[i]=string[:index] + newchar+string[index+1:]

    def get_scores(self,arr):
        temp = [self.game.get_score(x)[1] for x in arr]
        return temp

    def get_possibilities(self,arr):
        scores = self.get_scores(arr)
        scaled_scores = [math.exp(x) for x in scores]
        sum = np.sum(scaled_scores)
        probs = [x/sum for x in scaled_scores]
        return probs


    def get_best(self,arr,n):
        arr_with_score = []
        for i in arr:
            arr_with_score.append((i,self.game.get_score(i)[1]))
        arr_with_score.sort(key=lambda x:x[1])
        return [x[0] for x in arr_with_score[-n:]]

    def select_parents(self):

        if(self.selection_type==SelectionType.ONLY_BEST):
               self.parents= self.get_best(self.population,len(self.population)//2)
        elif(self.selection_type==SelectionType.ROULETTE):

            possibilities = self.get_possibilities(self.population)
            index = np.random.choice(len(self.population), len(self.population) // 2, p=possibilities)
            self.parents = [self.population[x] for x in index]


    def crossover(self):
        self.children=[]
        if(self.crossover_type==CrossOver.SINGLE_POINT):
            point = random.randint(0, self.chromosome_length)
            for i in range(len(self.parents)-1):
                child1_half1 = self.parents[i][0:point]
                child1_half2 = self.parents[i+1][point:]
                child1 = child1_half1 + child1_half2



                child2_half1 = self.parents[i+1][0:point]
                child2_half2 = self.parents[i][point:]
                child2 = child2_half1+child2_half2
                self.children.append(child1)
                self.children.append(child2)

        elif(self.crossover_type==CrossOver.TWO_POINT):
            point1 = random.randint(0, self.chromosome_length)
            point2=random.randint(0, self.chromosome_length)
            while (point1 == point2):
                point2 = random.randint(0, self.chromosome_length)
            if (point1 > point2):
                for i in range(len(self.parents)-1):
                    child1_part1 = self.parents[i][0:point2]
                    child1_part2=self.parents[i+1][point2:point1]
                    child1_part3 = self.parents[i][point1:self.chromosome_length]
                    child1 = child1_part1+child1_part2+child1_part3


                    child2_part1 = self.parents[i+1][0:point2]
                    child2_part2=self.parents[i][point2:point1]
                    child2_part3 = self.parents[i+1][point1:self.chromosome_length]
                    child2 = child2_part1+child2_part2+child2_part3

                    self.children.append(child1)
                    self.children.append(child2)
            else:
                for i in range(len(self.parents) - 1):
                    child1_part1 = self.parents[i][0:point1]
                    child1_part2 = self.parents[i + 1][point1:point2]
                    child1_part3 = self.parents[i][point2:self.chromosome_length]
                    child1 = child1_part1 + child1_part2 + child1_part3

                    child2_part1 = self.parents[i + 1][0:point1]
                    child2_part2 = self.parents[i][point1:point2]
                    child2_part3 = self.parents[i + 1][point2:self.chromosome_length]
                    child2 = child2_part1 + child2_part2 + child2_part3

                    self.children.append(child1)
                    self.children.append(child2)
        self.fix(self.children)

    def mutation(self,arr):
        if(random.random()<=self.mutation_prob):
            for i in range(len(arr)):
                temp = list(arr[i])
                index = np.random.choice(len(arr[i]), int(np.log2(len(arr[i]))), replace=False)
                for c in index:
                    if(arr[i][c]=='1'):
                        temp[c] = random.choice(['0','2'])
                    elif(arr[i][c]=='2'):
                        temp[c]=random.choice(['0','1'])
                    elif(arr[i][c]=='0'):
                        temp[c]=random.choice(['1','2'])
                arr[i] = ''.join(temp)
            self.fix(arr)

    def mutation2(self,arr):
        if(random.random()<=self.mutation_prob):
            for i in range(len(arr)):
                temp = list(arr[i])
                index = np.random.choice(len(arr[i]), int(np.log2(len(arr[i]))), replace=False)
                for c in index:
                    if (arr[i][c] == "1"):
                        if (random.random() > 0.75):
                            temp[c] = "0"
                        else:
                            temp[c] = str(random.randint(1, 2))
                    else:
                        temp[c] = str(random.randint(0, 2))
                arr[i] = ''.join(temp)
            self.fix(arr)
    def avg_score(self):
        sum = np.sum(self.get_scores(self.population))
        return sum/(len(self.population))

    def new_generation(self):
        self.children=self.get_best(self.children,len(self.parents))
        self.next_gen=[]

        possibilities = self.get_possibilities(self.population)
        index = np.random.choice(len(self.population), len(self.population) // 2, p=possibilities)
        self.next_gen = [self.population[x] for x in index]

        # self.mutation(self.next_gen)
        self.population= self.next_gen+self.children

    def get_best_agent(self):
        scores = self.get_scores(self.population)
        index = scores.index(max(scores))
        best_agent = self.population[index]
        score = max(scores)
        return best_agent,score
    def get_worst_agent(self):
        scores = self.get_scores(self.population)
        index = scores.index(min(scores))
        worst_agent = self.population[index]
        score = min(scores)
        return worst_agent,score
    def run(self):
        avg_list = []
        self.create_initial_population()
        epsilon = 0.001
        best_arr=[]
        worst_arr=[]
        for i in range(400):
            new_avg = self.avg_score()
            avg_list.append(new_avg)
            best_arr.append(self.get_best_agent()[1])
            worst_arr.append(self.get_worst_agent()[1])
            self.select_parents()
            self.crossover()
            self.mutation(self.children)
            self.new_generation()


        x = [i for i in range(len(avg_list))]
        x_arr= np.array(x)
        y_arr= np.array(avg_list)

        for x in self.population:
            if(self.game.get_score(x)[0]==True and self.game.get_score(x)[1]==self.get_best_agent()[1]):
                print(x)
                # print("Test:",test.get_score(x))
                print(self.game.levels[self.game.current_level_index][1])
                break


        plt.plot(x_arr,y_arr,'g')
        plt.plot(x_arr,best_arr,'r')
        plt.plot(x_arr,worst_arr,'b')
        plt.legend(["Average","Best Agent","Worst Agent"],loc="lower right")
        plt.show()

game = Game([('level2', '____G_____')],win_reward_allowed=True)

ga  = GeneticAlgorithm(200,game,True,SelectionType.ONLY_BEST,CrossOver.SINGLE_POINT,0.1)
ga.run()


