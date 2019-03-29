#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import argparse
import numpy as np
		
class IndependentQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(IndependentQLearningAgent, self).__init__()
		self.amountActions = len(self.possibleActions)
		self.discountFactor = discountFactor
		self.learningRatePassed = learningRate
		self.learningRate = 0.0
		self.epsilonPassed = epsilon
		self.epsilon = 0.0
		self.states = [(x,y) for x in range(5) for y in range(5)] # generates all states
		self.states_special = []
		self.states_special.append("GOAL")
		self.states_special.append("OUT_OF_BOUNDS")
		self.states_special.append("OUT_OF_TIME")
		self.curState = []
		self.experience = []
		self.curEpisode =0
		self.numTakenActionsLastEpisode = 0
		self.qValue = {}
		for action in self.possibleActions:
			for state_1 in self.states:
				for state_2 in self.states:
						self.qValue[((state_1,state_2),action)] =initVals
			for state_special in self.states_special:
				self.qValue[(state_special,action)] = initVals

	def setExperience(self, state, action, reward, status, nextState):
		self.experience = (timeSteps,state,action,reward,nextState)
		#raise NotImplementedError
	
	def learn(self):
		state_t = (tuple(self.experience[1][0][0]),tuple(self.experience[1][0][1]))
		action_t = self.experience[2]
		reward = self.experience[3]
		if self.experience[4] == "GOAL" or self.experience[4] == "OUT_OF_BOUNDS" or self.experience[4]=="OUT_OF_TIME":
			state_next = self.experience[4]
		else:
		    state_next = (tuple(self.experience[4][0][0]),tuple(self.experience[4][0][1]))
		q_value_nextState = []
		for action in self.possibleActions:
			q_value_nextState.append(self.qValue[(state_next,action)])
		q_value_nextStateMax = max(q_value_nextState)
		qValue = self.qValue[(state_t,action_t)]
		changeValue =  self.learningRate*(reward+self.discountFactor*(q_value_nextStateMax)-qValue)
		self.qValue[(state_t,action_t)] =qValue + changeValue
		return changeValue
		raise NotImplementedError

	def act(self):
		q_value = []
		for action in self.possibleActions:
			q_value.append(self.qValue[((tuple(self.curState[0][0]),tuple(self.curState[0][1])),action)])
		q_valueMax = max(q_value)
		q_valueMaxIndexAll = [i for i,j in enumerate(q_value) if j==q_valueMax]
		q_valueMaxIndex = random.choice(q_valueMaxIndexAll)
		actionIndexAll = [i for i in range(self.amountActions)]
		actionIndexAll.remove(q_valueMaxIndex)
		pro = random.random()
		if pro > self.epsilon:
			curAction = self.possibleActions[q_valueMaxIndex]
		else:
			curAction = self.possibleActions[random.choice(actionIndexAll)]
		return curAction
		raise NotImplementedError

	def toStateRepresentation(self, state):
		return state #返回的state是一个list, 1st entry is own location, 2nd entry is opponent location
		raise NotImplementedError

	def setState(self, state):
		self.curState = state 
		#raise NotImplementedError

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		#raise NotImplementedError
		
	def setLearningRate(self, learningRate):
		self.learningRate = learningRate
		#raise NotImplementedError
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		'''if self.curEpisode != episodeNumber:
			self.curEpisode = episodeNumber
			self.numTakenActionsLastEpisode = numTakenActions
		numTakenActionsInepisode = numTakenActions - self.numTakenActionsLastEpisode'''
		epsilonComputed = self.epsilonPassed*(np.power(0.9996,episodeNumber))
		#learningRateComputed =1/np.power((numTakenActionsInepisode+1),0.7)
		learningRateComputed = np.power(0.9999954,numTakenActions)
		return(learningRateComputed,epsilonComputed)
		raise NotImplementedError

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	for i in range(args.numAgents):
		agent = IndependentQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0)
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
				learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
				agent.setEpsilon(epsilon)
				agent.setLearningRate(learningRate)
			actions = []
			stateCopies = []
			changes = []
			for agentIdx in range(args.numAgents):
				obsCopy = deepcopy(observation[agentIdx]) # 从环境中获取每个agent的state 每个agent接收到的state一样的
				stateCopies.append(obsCopy) #将这两个agent的state保存下来
				agents[agentIdx].setState(agent.toStateRepresentation(obsCopy)) #将获取的state传给相应的agent
				actions.append(agents[agentIdx].act())
			numTakenActions += 1
			nextObservation, reward, done, status = MARLEnv.step(actions) # 这里返回的每个东西都是一个list，每个list中分别记录了两个agent的相应信息

			for agentIdx in range(args.numAgents):
				agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
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
