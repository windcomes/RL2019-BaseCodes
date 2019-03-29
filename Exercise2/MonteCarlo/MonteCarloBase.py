#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
import random

class MonteCarloAgent(Agent):
	def __init__(self, discountFactor, epsilon, initVals=0.0):
		super(MonteCarloAgent, self).__init__()
		#self.possibleActions = ['DRIBBLE_UP','DRIBBLE_DOWN','DRIBBLE_LEFT','DRIBBLE_RIGHT','KICK']
		self.numActions = len(self.possibleActions)
		self.discountFactor = discountFactor 
		self.epsilonPassed = epsilon
		self.epsilon = 0
		self.timeStep = 0
		self.experience = []
		self.currentState = []
		self.G = 0

		self.states = [(x,y) for x in range(6) for y in range(5)] # generates all states
		self.states.append("GOAL")
		self.states.append("OUT_OF_BOUNDS")
		self.states.append("OUT_OF_TIME")
		self.qValue = {}
		self.returns ={}
		for state in self.states:
			for action in self.possibleActions:
				self.qValue[(state,action)] = initVals
				self.returns[(state,action)] = []

	def learn(self):
		qUpdatedInEpisode = []
		for t in range(self.timeStep-1,-1,-1):
			self.G = self.discountFactor*self.G + self.experience[t][2]
			state_t = tuple(self.experience[t][0][0])
			action_t = self.experience[t][1]
			for tt in range(0,t):
				if (state_t,action_t)==(tuple(self.experience[tt][0][0]),self.experience[tt][1]):
					break
			if tt == t: # It means the state-action pair occurs for the first time
				self.returns[(state_t,action_t)].append(self.G)
				self.qValue[(state_t,action_t)] = np.mean(self.returns[(state_t,action_t)])
				qUpdatedInEpisode.append(self.qValue[(state_t,action_t)])
		qUpdatedInEpisode.reverse()
		return (self.qValue,qUpdatedInEpisode)
		raise NotImplementedError

	def toStateRepresentation(self, state):
		return state 
		raise NotImplementedError

	def setExperience(self, state, action, reward, status, nextState):
		self.experience.append((state,action,reward,nextState))
		self.timeStep = self.timeStep+1
		#raise NotImplementedError

	def setState(self, state):
		self.currentState = state
		#raise NotImplementedError

	def reset(self):
		self.experience = []
		self.timeStep = 0
		self.G = 0
		#raise NotImplementedError

	def act(self):
		curState = tuple(self.currentState[0])
		qValue = []
		for action in self.possibleActions:
			qValue.append(self.qValue[(curState,action)])
		qValueMax = max(qValue)
		qValueMaxIndexAll = [i for i,j in enumerate(qValue) if j==qValueMax]
		qValueMaxIndex = random.choice(qValueMaxIndexAll)
		actionIndexAll = [i for i in range(self.numActions)]
		actionIndexAll.remove(qValueMaxIndex)

		proNotMaxA = self.epsilon/self.numActions
		proisMaxA = 1 - self.epsilon + proNotMaxA
		pro = random.random()
		if pro <= proisMaxA:
			choosedAction = self.possibleActions[qValueMaxIndex]
		else:
			choosedAction = self.possibleActions[random.choice(actionIndexAll)]
		return choosedAction
		raise NotImplementedError

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		#raise NotImplementedError

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		epsilonComputed = np.power(0.999,episodeNumber)
		return(epsilonComputed) 
		raise NotImplementedError


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	#Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Monte-Carlo Agent
	agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes
	numTakenActions = 0
	# Run training Monte Carlo Method
	for episode in range(numEpisodes):	
		agent.reset()
		observation = hfoEnv.reset()
		status = 0
        
		while status==0:
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy)) 
			action = agent.act()
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			observation = nextObservation
        
		agent.learn()
