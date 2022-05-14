import shell
import util
import wordsegUtil
from wordsegUtil import ACTION_FORWARD

############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):
    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0, 0  # (position, last segmentation)
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.query) + 1
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        result = []
        if state[0] != 0:
            # segmentation of [last segmentation, position]
            word = self.query[state[1]:state[0]]
            result.append((word, (state[0]+1, state[0]), self.unigramCost(word)))
        if state[0] != len(self.query):
            result.append((ACTION_FORWARD, (state[0]+1, state[1]), 0))
        return result
        # END_YOUR_CODE

def segmentWords(query, unigramCost):
    if len(query) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    return ' '.join(word for word in ucs.actions if word != ACTION_FORWARD)
    # END_YOUR_CODE

############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):
    def __init__(self, queryWords, bigramCost, possibleFills):
        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0, wordsegUtil.SENTENCE_BEGIN    # state: (position, previous word)
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.queryWords)
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
        pos = state[0]
        prev_word = state[1]
        query = self.queryWords[pos]
        potential_words = self.possibleFills(query)
        if not potential_words:
            return [(query, (pos+1, prev_word), self.bigramCost(prev_word, query))]
        return [(word, (pos+1, word), self.bigramCost(prev_word, word))
                for word in potential_words]
        # END_YOUR_CODE

def insertVowels(queryWords, bigramCost, possibleFills):
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost, possibleFills))
    return ' '.join(ucs.actions)
    # END_YOUR_CODE

############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):
    def __init__(self, query, bigramCost, possibleFills):
        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 0, 0, wordsegUtil.SENTENCE_BEGIN  # state: (position, last segmentation, last word)
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return state[0] == len(self.query) + 1
        # END_YOUR_CODE

    def succAndCost(self, state):
        # BEGIN_YOUR_CODE (our solution is 14 lines of code, but don't worry if you deviate from this)
        pos = state[0]
        last_seg = state[1]
        last_word = state[2]

        result = []
        if pos != 0:
            query = self.query[last_seg:pos]
            potential_words = self.possibleFills(query)
            result += [(word, (pos+1, pos, word), self.bigramCost(last_word, word))
                       for word in potential_words]
        if pos != len(self.query):
            result.append((ACTION_FORWARD, (pos+1, last_seg, last_word), 0))
        return result
        # END_YOUR_CODE

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    ucs = util.UniformCostSearch(verbose=1)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost, possibleFills))
    return ' '.join(word for word in ucs.actions if word != ACTION_FORWARD)
    # END_YOUR_CODE

############################################################

if __name__ == '__main__':
    shell.main()
