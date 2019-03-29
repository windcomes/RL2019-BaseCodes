#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import random
import numpy as np

class QLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(QLearningAgent, self).__init__()
		self.numActions = len(self.possibleActions)
		self.learningRatePassed = learningRate
		self.discountFactor = discountFactor
		self.epsilonPassed = epsilon
		self.epsilon = 0

		self.states = [(x,y) for x in range(5) for y in range(6)] # 生成所有的state
		self.states.append("GOAL")
		self.states.append("OUT_OF_BOUNDS")
		self.states.append("OUT_OF_TIME")
		self.qValue = {}
		for state in self.states:
			for action in self.possibleActions:
				self.qValue[(state,action)] = initVals

		self.currentState = []
		self.currentExperience = []

	def learn(self):
		state_t = tuple(self.currentExperience[0][0])
		action_t = self.currentExperience[1]
		reward = self.currentExperience[2]
		state_next = tuple(self.currentExperience[3][0])
		
		qValue_next = []
		for action in self.possibleActions:
			qValue_next.append(self.qValue[(state_next,action)])
		qValue_nextMax = max(qValue_next)
		
		qValue = self.qValue[(state_t,action_t)]
		changeValue =  self.learningRate*(reward+self.discountFactor*(qValue_nextMax)-qValue)
		self.qValue[(state_t,action_t)] =qValue + changeValue

		return changeValue
		raise NotImplementedError

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

	def toStateRepresentation(self, state):
		return state
		raise NotImplementedError

	def setState(self, state):
		self.currentState = state
		#raise NotImplementedError

	def setExperience(self, state, action, reward, status, nextState):
		self.currentExperience = (state,action,reward,nextState)
		#raise NotImplementedError

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate
		#raise NotImplementedError

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		#raise NotImplementedError

	def reset(self):
		self.timeStep = 0
		#raise NotImplementedError
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		epsilonComputed = np.power(0.999,episodeNumber)
		#epsilonComputed = 0.2
		learningRateComputed = np.power(0.99954,episodeNumber)
		return(learningRateComputed,epsilonComputed)
		raise NotImplementedError

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	# Initialize connection with the HFO server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Q-Learning Agent
	agent = QLearningAgent(learningRate = 0.1, discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes

	# Run training using Q-Learning
	numTakenActions = 0 
	for episode in range(numEpisodes):
		status = 0
		observation = hfoEnv.reset()
		
		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			update = agent.learn()
			
			observation = nextObservation
	
