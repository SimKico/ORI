# myAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from game import Agent, Directions
from searchProblems import PositionSearchProblem

import util
import time
import search
import random

"""
IMPORTANT
`agent` defines which agent you will use. By default, it is set to ClosestDotAgent,
but when you're ready to test your own agent, replace it with MyAgent
"""
def createAgents(num_pacmen, agent='MyAgent'):
    return [eval(agent)(index=i) for i in range(num_pacmen)]

class MyAgent(Agent):
    """
    Implementation of your agent.
    """

    # Da bi mogli da razmenjuju informacije između više Pacman-ova,
    # kreiramo promenljivu

    # Broj PacMan-ova
    pacmanAmount = 0
    # Hrana koja je postavljena za cilj nekog Pacman, kako bi je drugi Pacman-ovi ignorisali
    # da bi izbjegli situaciju da vise PacMana ganja istu hranu
    chasingGoal = []

    def getAction(self, state):
        """
            Returns the next action the agent will take
        """
        #Zaustavi Pacmana
        if self.isFinished:
            return Directions.STOP
        else:
            # ako nema vise koraka u listi akcija,
            # onda se generise nova lista akcija
            if len(self.actions) == 0:
                actions = search.ucs(MyFoodSearchProblem(state, self.index))
                self.actions = actions
                print(actions)
            # Sve dok lista akcija nije prazna,
            # vracaj prvu akciju i azuriraj listu akcija
            if len(self.actions) > 0:
                nextAction = self.actions[0]
                del self.actions[0]
                return nextAction
            # Ako je lista akcija prazna,
            # zaustavi Pacmana
            else:
                self.isFinished = True
                return Directions.STOP

    def initialize(self):
        """
        Intialize anything you want to here. This function is called
        when the agent is first created. If you don't need to use it, then
        leave it blank
        """

        "*** YOUR CODE HERE"
        # Inicijalizacija podataka
        self.isFinished = False
        self.actions = []

        MyAgent.pacmanAmount += 1

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""


class MyFoodSearchProblem(PositionSearchProblem):

    def __init__(self, gameState, agentIndex):

        # Info o hrani
        self.food = gameState.getFood()

        # PositionSearchProblem
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0

        # dodjeljujemo svakom Pacmanu hranu koju ce on da jede
        self.agentIndex = agentIndex
        self.foodAll = self.food.asList()
        avgFood = len(self.foodAll) // MyAgent.pacmanAmount + 1
        self.foodByAgent = self.foodAll[agentIndex * avgFood : (agentIndex + 1) * avgFood]

    def isGoalState(self, state):
        #resavanje situacije kad je vise Pacmena nego hrane
        if len(self.foodAll) <= MyAgent.pacmanAmount:
            return state in self.foodAll
        #procenjujemo trenutnu poziciju ako ima hrane
        if state in self.foodAll:
            #ako je hrana u listi hrane tog agenta i ako nije vec dodeljena u ChasingGoal
            if (state in self.foodByAgent) and (state not in MyAgent.chasingGoal):
                #dodajemo stanje u chasingGoal za tog Pacmena
                MyAgent.chasingGoal.append(state)
                return True
            #ako je Pacmen blizu hrane, a hrana nije dodeljena drugom Pacmenu
            elif (util.manhattanDistance(state, self.startState) <= (1 + self.agentIndex) * (1 + self.agentIndex)) and (state not in MyAgent.chasingGoal):
                #dodajemo stanje u chasingGoal za tog Pacmena
                MyAgent.chasingGoal.append(state)
                return True
            else:
                return state in self.foodByAgent
        else:
            return False


class ClosestDotAgent(Agent):

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition(self.index)
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState, self.index)


        "*** YOUR CODE HERE ***"

        return search.breadthFirstSearch(problem)
        #util.raiseNotDefined()

    def getAction(self, state):
        return self.findPathToClosestDot(state)[0]

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"

        return self.food[x][y]
        #util.raiseNotDefined()

