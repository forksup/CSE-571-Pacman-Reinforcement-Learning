# valueIterationAgents.py
# -----------------------
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

import mdp, util
import copy
from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        ValueEstimationAgent.__init__(self)
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.actions = ['north', 'south', 'west', 'east']
        self.actionRes = {'north': [0, 1], 'south': [0, -1], 'west': [-1, 0], 'east': [1, 0]}
        self.relatedActions = {'north': ['west', 'east'], 'south': ['west', 'east'],
                               'west': ['north', 'south'], 'east': ['north', 'south']}
        self.runValueIteration()

    def runValueIteration(self):
        for i in range(self.iterations):
            oValues = copy.deepcopy(self.values)
            for x in range(self.mdp.grid.width):
                for y in range(self.mdp.grid.height):
                    if isinstance(self.mdp.grid[x][y], str):
                        oValues[(x, y)] = self.mdp.livingReward \
                                + (max([self.computeQValueFromValues([x, y], a) for a in self.actions]))
                    elif isinstance(self.mdp.grid[x][y], int):
                        oValues[(x, y)] = self.mdp.grid[x][y]
            self.values = oValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        if action == 'exit':
            return self.mdp.grid[state[0]][state[1]]

        nextStates = [[x + y for x, y in zip(state, self.actionRes[action])],
                      [x + y for x, y in zip(state, self.actionRes[self.relatedActions[action][0]])],
                      [x + y for x, y in zip(state, self.actionRes[self.relatedActions[action][1]])]]

        result = []

        for nState in nextStates:
            if 0 <= nState[0] < self.mdp.grid.width and 0 <= nState[1] < self.mdp.grid.height \
                    and self.mdp.grid[nState[0]][nState[1]] != '#':
                result.append(nState)
            else:
                result.append(state)

        total = (1 - self.mdp.noise) * self.values[(result[0][0], result[0][1])]

        if self.mdp.noise > 0:
            total += self.mdp.noise/2 * self.values[(result[1][0], result[1][1])]
            total += self.mdp.noise/2 * self.values[(result[2][0], result[2][1])]

        return self.discount * total

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if state == self.mdp.grid.terminalState:
            return 'None'
        elif isinstance(self.mdp.grid.data[state[0]][state[1]], int):
            return 'exit'

        nextStates = [[sum(i) for i in zip(state, self.actionRes[a])] for a in self.actions]

        scores = []
        actions = []

        for i in range(len(nextStates)):
            if 0 <= nextStates[i][0] < self.mdp.grid.width and 0 <= nextStates[i][1] < self.mdp.grid.height:
                scores.append(self.computeQValueFromValues(state, self.actions[i]))
                actions.append(self.actions[i])
        return actions[scores.index(max(scores))]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
