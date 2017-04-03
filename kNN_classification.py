# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 12:19:09 2017

@author: KANHAIYA
"""

import numpy as np
# import random 
import scipy.stats as ss

def distance(p1,p2):
    ''' It finds the distance between the two pints'''
    return np.sqrt(np.sum(np.power(p2-p1,2)))
    
def majority_vote(votes):
    
    '''Returns teh most common elemnt in an array'''
    '''vote_counts = {}
    for vote in votes:
        if vote in vote_counts:
            vote_counts[vote] += 1
        else:
            vote_counts[vote] = 1
    winners = []
    max_counts = max(vote_counts.values())
    for vote,counts in vote_counts.items():
        if counts == max_counts:
            winners.append(vote)'''
    mode, count = ss.mstats.mode(votes)
    return int(mode)
    

votes = [1,2,3,1,2,3,4,4,3,2,2,1,1]
int(majority_vote(votes))

#loop over all points
   # calculate teh distance between the point p and the other points
# sort the other points nearest to the point p
def find_nearest_neighbours(p, points, k=5):
    ''' find k nearest point to p and return their indices'''
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[0:k]


points = np.array([[1,1], [1,2], [1,3], [2,1], [2,2], [2,3], [3,1], [3,2] ,[3,3]])
p = np.array([2.5,2])
import matplotlib.pyplot as plt

plt.plot(points[:,0], points[:,1], "ro")
plt.plot(p[0],p[1], "bo")
plt.axis([0.5, 3.5, 0.5, 3.5])

def knn_predict(p, points, outcomes, k = 5 ):
    ind = find_nearest_neighbours(p, points, k)    
    # find k nearest neighbours
    #predict the category of the point on majority of vote
    return(majority_vote(outcomes[ind]))
    
outcomes = np.array([0,0,0,0,1,1,1,1,1])
def generate_synth_data(n = 50):
    '''Generate a synthetic data using bivariate normal distribution'''
    points = np.concatenate((ss.norm(0, 1).rvs((n,2)), ss.norm(1, 1).rvs((n,2))), axis = 0)
    outcomes = np.concatenate((np.repeat(0, n), np.repeat(1, n))) 
    return(points, outcomes)
n = 20   
plt.figure()
plt.plot(points[:n,0], points[:n,1], "ro")
plt.plot(points[n:,0], points[n:,1], "bo")
plt.savefig("bivardata.pdf")
ss.norm(0, 1).rvs((5,2))
ss.norm(1, 1).rvs((5,2))