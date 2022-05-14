#!/usr/bin/python

import collections
import math
import re
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    phi = collections.defaultdict(int)
    for word in re.sub("[^\w\s]", "", x).split():
        phi[word] = phi.get(word, 0) + 1
    return phi
    # END_YOUR_CODE

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    # feature extraction
    trainSize = len(trainExamples)
    # feature extraction
    phi = []
    y = []
    for i in range(trainSize):
        phi.append(featureExtractor(trainExamples[i][0]))
        y.append(trainExamples[i][1])

    # loss function
    def F(w, ind):
        return max(0, 1 - dotProduct(w, phi[ind]) * y[ind])

    # weights update function
    def wUpdate(w, ind):
        increment(w, y[ind] * eta, phi[ind])

    # learning
    for _ in range(numIters):
        for i in range(trainSize):
            loss = F(weights, i)
            if loss > 0:
                wUpdate(weights, i)
    # END_YOUR_CODE
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        l = len(weights)
        phi = {list(weights.keys())[random.randint(0, l-1)]: random.random() for _ in range(random.randint(1, l))}
        y = dotProduct(phi, weights)
        if y != 0:
            y = int(y / abs(y))
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        s = re.sub("[^\w]", "", x)
        phi = {}
        for i in range(len(s)+1-n):
            phi[s[i:i+n]] = phi.get(s[i:i+n], 0) + 1
        return phi
        # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples: list[dict], K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 25 lines of code, but don't worry if you deviate from this)
    centers: list[dict] = []
    assignments_temp = {}
    assignments = []
    # random centers initialization
    random.seed(42)
    for i in range(K):
        centers.append(examples[random.randint(0, len(examples)-1)])

    def dist(e1: dict, e2: dict):
        sDist = 0
        for key, value in e1.items():
            sDist += (value - e2.get(key, 0))**2
        for key, value in e2.items():
            if key not in e1:
                sDist += value**2
        return math.sqrt(sDist)

    def assign(example):
        sDists = []
        for center in centers:
            sDists.append(dist(example, center))
        return min((j, i) for i, j in enumerate(sDists))[1]

    def recenter(assignment):
        center = {}
        for eg in assignment:
            for key, value in eg.items():
                center[key] = center.get(key, 0) + value / len(assignment)
        return center

    def totalCost(c, a):
        return sum(dist(examples[i], c[a[i]]) for i in range(len(examples)))

    for i in range(maxIters):
        assignments_dict = {}
        assignments = []
        # assignment
        for example in examples:
            n = assign(example)
            assignments.append(n)
            assignments_dict.setdefault(n, []).append(example)
        # convergence check
        if i > 0:
            converged = min(assignments[j] == assignments_temp[j] for j in range(len(examples)))
            if converged:
                return centers, assignments, totalCost(centers, assignments)
        assignments_temp = assignments
        # recenter
        for k in range(K):
            centers[k] = recenter(assignments_dict[k])
    return centers, assignments, totalCost(centers, assignments)
    # END_YOUR_CODE
