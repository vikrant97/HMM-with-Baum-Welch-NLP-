import sys,os,re
import numpy as np
import random
from nltk.corpus import brown
import time

def get_states():
	l=['tag1','tag2','tag3','tag4','tag5','tag6','tag7','tag8','tag9','tag10']
	return l

def fb_alg(states,vocab,A,B,pi,observation):
	alpha={}
	beta={}
	for i in states:
		alpha[i]={}
		beta[i]={}
		for t in xrange(len(observation)):
			alpha[i][t]=0
			beta[i][t]=0

	c=np.zeros(len(observation))
	###Initialize alpha(i,0) and compute others
	for i in states:
		alpha[i][0]=pi[i]*B[i][observation[0]]
		#c[0]+=alpha[i][0]

	# for i in states:
	# 	alpha[i][0]/=c[0]

	for t in range(1,len(observation)):
		for j in states:
			alpha[j][t]=0
			for i in states:
				alpha[j][t]+=alpha[i][t-1]*A[i][j]
			alpha[j][t]*=B[j][observation[t]]
			#c[t]+=alpha[j][t]
		# for i in states:
		# 	alpha[i][t]/=c[t]

	###Initialize beta(i,t-1)
	T=len(observation)-1
	for i in states:
		if alpha[i][T]!=0:
			beta[i][T]=1.0/alpha[i][T]

	for t in range(T-1,-1,-1):
		for i in states:
			beta[i][t]=0
			for j in states:
				beta[i][t]+=A[i][j]*B[j][observation[t+1]]*beta[j][t+1]
			#beta[i][t]/=c[t]
	return alpha,beta



def baum_welch(states,vocab,A,B,pi,observations):
	
	for itr in xrange(1):
		for observation in observations:
			alpha,beta=fb_alg(states,vocab,A,B,pi,observation)
			###Calculating theta values
			theta={}
			for i in states:
				theta[i]={}
				for j in states:
					theta[i][j]={}
					for t in xrange(len(observation)):
						theta[i][j][t]=0
			summ=0.0
			for i in states:
				for j in states:
					for t in xrange(len(observation)-1):
						theta[i][j][t]=alpha[i][t]*beta[j][t+1]*A[i][j]*B[j][observation[t+1]]
						summ+=theta[i][j][t]
			for i in states:
				for j in states:
					for t in xrange(len(observation)):
						if summ!=0:
							theta[i][j][t]/=summ
						else:
							theta[i][j][t]=0.0

			#####Calculating gamma values
			gamma={}
			for i in states:
				gamma[i]={}
				for t in xrange(len(observation)):
					gamma[i][t]=0.0
					for j in states:
						gamma[i][t]+=theta[i][j][t]
			####Re-estimation of pi
			for i in states:
				pi[i]=gamma[i][0]

			###Re-estimation of A
			for i in states:
				denom=0.0
				summ=0.0
				for t in xrange(len(observation)):
					denom+=gamma[i][t]
				for j in states:
					num=0.0
					for t in xrange(len(observation)):		
							num+=theta[i][j][t]
					if denom!=0:
						A[i][j]=num/denom
					summ+=A[i][j]
				for j in states:
					A[i][j]/=summ

			####Re-estimation of B

			for i in states:
				summ=0.0
				denom=0.0
				for t in xrange(len(observation)):
					denom+=gamma[i][t]
				for j in xrange(len(vocab)):
					num=0.0
					for t in xrange(len(observation)):
						if observation[t]==vocab[j]:
							num+=gamma[i][t]
					if denom!=0:
						B[i][vocab[j]]=num/denom
					summ+=B[i][vocab[j]]
				for j in xrange(len(vocab)):
					B[i][vocab[j]]/=summ
			# print B
			# time.sleep(1)
	return pi,A,B

if __name__=="__main__": 

	states=get_states()
	###Generating Vocab & observations
	vocab=[]
	observations=[]
	count=0
	for sent in brown.sents():
		if count>=10:
			break
		observations.append(sent)
		for word in sent:
			if word not in vocab:
				vocab.append(word)
		count+=1

	###Initializing The Transition Matrix A
	A={}
	for i in states:
		A[i]={}
		summ=0
		for j in states:
			A[i][j]=float(random.randint(1,100))
			summ+=A[i][j]
		for j in states:
			A[i][j]/=summ

	#### Initializing the emission matrix
	B={}
	for i in states:
		B[i]={}
		summ=0.0
		for word in vocab:
			B[i][word]=float(random.randint(1,100))
			summ+=B[i][word]
		for word in vocab:
			B[i][word]/=summ

	####Initializing pi
	pi={}
	summ=0.0
	for i in states:
		pi[i]=random.randint(1,100)
		summ+=pi[i]
	for i in states:
		pi[i]/=summ

	pi,A,B=baum_welch(states,vocab,A,B,pi,observations)
	print B
	output={}
	for i in states:
		count=0
		output[i]=[]
		for key in sorted(B[i].iterkeys()):
			if count>10:
				break
			output[i].append(key)
			count+=1
	for key in output:
		print key,":", output[key]