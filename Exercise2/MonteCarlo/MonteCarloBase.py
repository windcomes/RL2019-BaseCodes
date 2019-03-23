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
		#同时继承它的父类agent的初始化 self.possibleActions = ['DRIBBLE_UP','DRIBBLE_DOWN','DRIBBLE_LEFT','DRIBBLE_RIGHT','KICK']
		self.amountActions = len(self.possibleActions)
		self.discountFactor = discountFactor # 初始化discountFactor
		self.epsilonPassed = epsilon  # 初始化epsilon
		self.epsilon = 0
		self.experience = []
		self.timeStep = 0
		self.curState = (0,0)
		self.G = 0
		self.S = [(x,y) for x in range(6) for y in range(5)] # 生成所有的state
		self.S.append("GOAL")
		self.S.append("OUT_OF_BOUNDS")
		self.returns ={}
		self.qValue = {}
		for state in self.S:
			for action in self.possibleActions:
				self.qValue[(state,action)] = 0
				self.returns[(state,action)] = []

	def learn(self):
		qUpdatedInEpisode = []
		for t in range(self.timeStep-1,-1,-1):
			state_t = self.experience[t][1]
			action_t = self.expeirence[t][2]
			self.G = self.discountFactor*self.G + self.experience[t][3]
			for tt in range(0,t):
				if (state_t,action_t)==(self.experience[tt][1],self.experience[tt][2]):
					break
			if tt == t: #说明是这个s-a对第一次出现
				self.returns[(state_t,action_t)].append(self.G)
				self.qValue[(state_t,action_t)] = np.mean(self.returns[(state_t,action_t)])
				qUpdatedInEpisode.append(self.qValue[(state_t,action_t)])
		qUpdatedInEpisode.reverse()
		return (self.qValue,qUpdatedInEpisode)
		raise NotImplementedError

	def toStateRepresentation(self, state):
		return state # 这里可以不用设置，与HFOAttackingPlayer中的state表示保持一致，agent以及opponents的state都用tuple表示
		raise NotImplementedError

	def setExperience(self, state, action, reward, status, nextState):
		self.expeirence.append((self.timeStep,state,action,reward,nextState))
		self.timeStep = self.timeStep+1
		raise NotImplementedError

	def setState(self, state):
		self.curState = state[0] #这里的state[0]是一个元组，代表的是当前的location(x,y)
		raise NotImplementedError

	def reset(self):
		self.experience = []
		self.timeStep = 0
		self.G =0
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
			return self.possibleActions[q_valueMaxIndex]
		else:
			return self.possibleActions[random.choice(actionIndexAll)]
		raise NotImplementedError

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon[0] #这里input的epsilon是下一个method返回的tuple格式，我们只取tuple中的第一个值
		raise NotImplementedError

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		return(self.epsilonPassed,numTakenActions) # 这里我们使得每个episode中，每个timestep下的epsilon值都不改变
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
        #下面这段代码在生成其中一个episode.
		while status==0:
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon) #设置了agent的epsilon
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy)) #把agent的当前位置以及defending players的位置都传递进来
			action = agent.act() #agent
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)#看到这里！！！！！！！！！
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			observation = nextObservation
        # 等待其中一个episode生成完毕后，开始进行mc的学习
		agent.learn()
