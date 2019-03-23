#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import random
import numpy as np

class SARSAAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(SARSAAgent, self).__init__()
		self.learningRatePassed = learningRate
		self.discountFactor = discountFactor
		self.epsilonPassed = epsilon
		self.S = [(x,y) for x in range(6) for y in range(5)] # 生成所有的state
		self.S.append("GOAL")
		self.S.append("OUT_OF_BOUNDS")
		self.curState = (0,0)
		self.experience = 0
		self.timeStep = 0
		self.qValue = {}
		self.curAction = 'None'
		for state in self.S:
			for action in self.possibleActions:
				self.qValue[(state,action)] = initVals

	def learn(self):
        t = self.timeStep - 1
		state_t = self.expeirence[t][0]
		action_t = self.expeirence[t][1]
		reward = self.experience[t][2]
		stata_next = self.experience[t+1][0]
		action_next = self.expeirence[t+1][1]
		qValue = self.qValue[(state_t,action_t)]
		changeValue =  self.learningRate*(reward+self.discountFactor*(self.qValue[(stata_next,action_next)])-qValue)
		self.qValue[(state_t,action_t)] =qValue + changeValue
		return changeValue
		raise NotImplementedError

	def act(self):
		q_value = []
		for action in self.possibleActions:
			q_value.append(self.qValue[(self.curState,action)])
		q_valueMax = max(q_value)
		q_valueMaxIndexAll = [i for i,j in enumerate(q_value) if j==q_valueMax]
		q_valueMaxIndex = random.choice(q_valueMaxIndexAll)
		actionIndexAll = [i for i in range(self.possibleActions)]
		actionIndexAll.remove(q_valueMaxIndex)
		proNotMaxA = self.epsilon/self.amountActions
		proisMaxA = 1 - self.epsilon + proNotMaxA
		pro = random.random()
		if pro <= proisMaxA:
			self.curAction = self.possibleActions[q_valueMaxIndex]
		else:
			self.curAction = self.possibleActions[random.choice(actionIndexAll)]
		return self.curAction
		raise NotImplementedError

	def setState(self, state):
		self.curState = state
		raise NotImplementedError

	def setExperience(self, state, action, reward, status, nextState):
		self.expeirence.append((self.timeStep,state,action,reward,nextState))
		self.timeStep = self.timeStep+1
		raise NotImplementedError

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return(self.learningRatePassed,self.epsilonPassed)
		raise NotImplementedError

	def toStateRepresentation(self, state):
		return state
		raise NotImplementedError

	def reset(self):
		self.experience = []
		self.timeStep = 0
		raise NotImplementedError

	def setLearningRate(self, learningRate):
		self.learningRate = learningRate
		raise NotImplementedError

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		raise NotImplementedError

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()
	
	numEpisodes = args.numEpisodes
	# Initialize connection to the HFO environment using HFOAttackingPlayer
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a SARSA Agent
	agent = SARSAAgent(0.1, 0.99,1)

	# Run training using SARSA
	numTakenActions = 0 
	for episode in range(numEpisodes):	
		agent.reset()
		status = 0

		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True

		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1

			nextObservation, reward, done, status = hfoEnv.step(action)
			print(obsCopy, action, reward, nextObservation)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			
			if not epsStart :
				agent.learn()
			else:
				epsStart = False
			
			observation = nextObservation

		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()

	
