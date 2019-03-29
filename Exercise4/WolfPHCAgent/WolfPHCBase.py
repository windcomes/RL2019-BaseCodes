#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
		
class WolfPHCAgent(Agent):
	def __init__(self, learningRate, discountFactor, winDelta=0.01, loseDelta=0.1, initVals=0.0):
		super(WolfPHCAgent, self).__init__()
		self.numActions = len(self.possibleActions)
		self.learningRatePassed = learningRate
		self.learningRate = 0.0
		self.discountFactor = discountFactor
		self.winDeltaPassed = winDelta
		self.winDelta = 0
		self.loseDeltaPassed = winDelta
		self.loseDelta = 0
		self.delta = 0

		self.states = [(x,y) for x in range(5) for y in range(5)] # 生成所有的state
		self.specialStates = []
		self.specialStates.append("GOAL")
		self.specialStates.append("OUT_OF_BOUNDS")
		self.specialStates.append("OUT_OF_TIME")
		# generate the Q-value table
		self.qValue = {}
		# generate the policy table
		self.policy = {} 
		for action in self.possibleActions:
			for state_1 in self.states:
				for state_2 in self.states:
					self.qValue[((state_1,state_2),action)] = initVals
					self.policy[((state_1,state_2),action)] = 1/self.numActions
			for speicalState in self.specialStates:
				self.qValue[(speicalState,action)] = initVals
				self.policy[(speicalState,action)] = 1/self.numActions
		self.avgPolicy = self.policy.copy()
		
		self.C = {}
		for state_1 in self.states:
				for state_2 in self.states:
					self.C[(state_1,state_2)] = 0
		for speicalState in self.specialStates:
			self.C[speicalState] = 0

		self.currentState = []
		self.currentExperience = []
		
		
	def setExperience(self, state, action, reward, status, nextState):
		self.currentExperience = (state, action,reward,nextState)
		return
		#raise NotImplementedError

	def learn(self):
		state_t = (tuple(self.currentExperience[0][0][0]),tuple(self.currentExperience[0][0][1]))
		action_t = self.currentExperience[1]
		reward = self.currentExperience[2]
		if self.currentExperience[3] == "GOAL" or self.currentExperience[3] == "OUT_OF_BOUNDS" or self.currentExperience[3]=="OUT_OF_TIME":
			state_next = self.currentExperience[3]
		else:
		    state_next = (tuple(self.currentExperience[3][0][0]),tuple(self.currentExperience[3][0][1]))
		qValue_nextState = []
		for action in self.possibleActions:
			qValue_nextState.append(self.qValue[(state_next,action)])
		qValue_nextStateMax = max(qValue_nextState)
		qValue = self.qValue[(state_t,action_t)]
		changeValue =  self.learningRate*(reward+self.discountFactor*(qValue_nextStateMax)-qValue)
		self.qValue[(state_t,action_t)] =qValue + changeValue
		return changeValue
		raise NotImplementedError

	def act(self):
		curstate = (tuple(self.currentState[0][0]),tuple(self.currentState[0][1]))
		pos = []
		for action in (self.possibleActions):
			pos.append(self.policy[(curstate,action)])
		if sum(pos) <=1:
			pos[-1] = pos[-1] + 1-sum(pos)
			if pos[-1]<=0.00000000000001:
				pos[-1] = 0
		choosedAction = np.random.choice(self.possibleActions,p=pos)
		return choosedAction
		#raise NotImplementedError

	def calculateAveragePolicyUpdate(self):
		curState = (tuple(self.currentState[0][0]),tuple(self.currentState[0][1]))
		self.C[curState] = self.C[curState]+1
		for action in self.possibleActions:
			originalValue = self.avgPolicy[(curState,action)]
			self.avgPolicy[(curState,action)] = originalValue + (1/self.C[curState])*(self.policy[(curState,action)]-originalValue)
		avg_return = []
		for action in self.possibleActions:
			avg_return.append(self.avgPolicy[(curState,action)])
		return avg_return
		#raise NotImplementedError

	def calculatePolicyUpdate(self):
		curState = (tuple(self.currentState[0][0]),tuple(self.currentState[0][1]))
		qValue = []
		for action in self.possibleActions:
			qValue.append(self.qValue[(curState,action)])
		qvalueMax = max(qValue)
		optimalActionsIndex = [i for i,j in enumerate(qValue) if j==qvalueMax]
		suboptimalActionsIndex = [i for i,j in enumerate(qValue) if j!=qvalueMax]
		optimalActions=  []
		suboptimalActions = []
		for i in range(self.numActions):
			if i in optimalActionsIndex:
				optimalActions.append(self.possibleActions[i])
			else:
				suboptimalActions.append(self.possibleActions[i])
		sum_pq_1 = 0
		sum_pq_2 = 0
		for action in self.possibleActions:
			sum_pq_1 = sum_pq_1 + self.policy[(curState,action)] * self.qValue[(curState,action)]
			sum_pq_2 = sum_pq_2 + self.policy[(curState,action)] * self.qValue[(curState,action)]
		if sum_pq_1>=sum_pq_2:
			self.delta = self.winDelta
		else:
			self.delta = self.loseDelta
		
		p_moved = 0
		for action in suboptimalActions:
			p_moved = p_moved + min(self.delta/(len(suboptimalActions)),self.policy[(curState,action)])
			self.policy[(curState,action)] = self.policy[(curState,action)] - min(self.delta/(len(suboptimalActions)),self.policy[(curState,action)])
		for action in optimalActions:
			self.policy[(curState,action)]= self.policy[(curState,action)]+ p_moved/(self.numActions-len(suboptimalActions))
		policy_return = []
		for action in self.possibleActions:
			policy_return.append(self.policy[(curState,action)])
		return policy_return
		raise NotImplementedError

	
	def toStateRepresentation(self, state):
		return state
		raise NotImplementedError

	def setState(self, state):
		self.currentState = state
		#raise NotImplementedError

	def setLearningRate(self,lr):
		self.learningRate = lr
		#raise NotImplementedError
		
	def setWinDelta(self, winDelta):
		self.winDelta = winDelta
		#raise NotImplementedError
		
	def setLoseDelta(self, loseDelta):
		self.loseDelta = loseDelta
		#raise NotImplementedError
	
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		#epsilonComputed = 0.2
		learningRateComputed = np.power(0.999954,episodeNumber)
		winDeltaComputed = 1/(5000+numTakenActions)
		loseDeltaComputed = 2*winDeltaComputed
		return(loseDeltaComputed,winDeltaComputed,learningRateComputed)
		raise NotImplementedError

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=100000)

	args=parser.parse_args()

	numOpponents = args.numOpponents
	numAgents = args.numAgents
	MARLEnv = DiscreteMARLEnvironment(numOpponents = numOpponents, numAgents = numAgents)

	agents = []
	for i in range(args.numAgents):
		agent = WolfPHCAgent(learningRate = 0.2, discountFactor = 0.99, winDelta=0.01, loseDelta=0.1)
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0
	totalReward_record = []
	totalTimestep = []
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		totalReward = 0.0
		timeSteps = 0
		
		while status[0]=="IN_GAME":
			for agent in agents:
				loseDelta, winDelta, learningRate = agent.computeHyperparameters(numTakenActions, episode)
				agent.setLoseDelta(loseDelta)
				agent.setWinDelta(winDelta)
				agent.setLearningRate(learningRate)
			actions = []
			perAgentObs = []
			agentIdx = 0
			for agent in agents:
				obsCopy = deepcopy(observation[agentIdx])
				perAgentObs.append(obsCopy)
				agent.setState(agent.toStateRepresentation(obsCopy))
				actions.append(agent.act())
				agentIdx += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)
			numTakenActions += 1

			agentIdx = 0
			for agent in agents:
				agent.setExperience(agent.toStateRepresentation(perAgentObs[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agent.learn()
				agent.calculateAveragePolicyUpdate()
				agent.calculatePolicyUpdate()
				agentIdx += 1
			totalReward = totalReward + reward[0]
			timeSteps = timeSteps + 1
			observation = nextObservation

		totalTimestep.append(timeSteps)
		totalReward_record.append(totalReward)
		if episode % 1000 == 0:
			print(episode)
			print(np.mean(totalReward_record))
			print(np.mean(totalTimestep))
			print(np.max(totalTimestep))
			totalReward_record = []
			totalTimestep = []
