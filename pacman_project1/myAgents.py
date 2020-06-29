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

    # Da bi mogli da razmenjuju informacije između više Pac-Man-ova,
    # kreiramo promenljivu

    # Indicates how many Pac-Man people there are
    pacmanAmount = 0
    # Indicate which food has been set as the target, so other Pac-Man can ignore it
    chasingGoal = []

    def getAction(self, state):
        """
        Returns the next action the agent will take
        """

        # If you pass the previous calculations and confirm that Doudou’s tasks have been completed,
        # in order to save computing resources, stop Doudou directly
        if self.isFinished:
            return Directions.STOP
        else:
            # If there is no next step in the action sequence,
            # then a new action sequence must be generated
            if len(self.actions) == 0:
                actions = search.ucs(MyFoodSearchProblem(state, self.index))
                self.actions = actions
                print(actions)
            #  As long as the action sequence is not empty,
            #  return the first action and update the action sequence
            if len(self.actions) > 0:
                nextAction = self.actions[0]
                del self.actions[0]
                return nextAction
            # If there are no further steps in the action sequence,
            # then the task of Doudou is completed
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
        # Initialize some Pac-Man information
        # isFinished indicates whether Pac-Man has completed his task,
        # initially is False
        self.isFinished = False
        self.actions = []

        MyAgent.pacmanAmount += 1

"""
Put any other SearchProblems or search methods below. You may also import classes/methods in
search.py and searchProblems.py. (ClosestDotAgent as an example below)
"""


class MyFoodSearchProblem(PositionSearchProblem):

    def __init__(self, gameState, agentIndex):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition(agentIndex)
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE


        self.agentIndex = agentIndex
        self.foodAll = self.food.asList()
        avgFood = len(self.foodAll) // MyAgent.pacmanAmount + 1
        self.foodByAgent = self.foodAll[agentIndex * avgFood: (agentIndex + 1) * avgFood]

    def isGoalState(self, state):
        if len(self.foodAll) <= MyAgent.pacmanAmount:
            return state in self.foodAll
        if state in self.foodAll:
            if (state in self.foodByAgent) and (state not in MyAgent.chasingGoal):
                MyAgent.chasingGoal.append(state)
                return True
            elif (util.manhattanDistance(state, self.startState) <= (1 + self.agentIndex) * (1 + self.agentIndex)) \
                    and (state not in MyAgent.chasingGoal):
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

