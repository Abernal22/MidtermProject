import numpy as np

class Node:
    #A node is supposed to be either a leaf node or a split node and have its own setup
    def __init__(self):
        pass
    #Each node has some kind of overall value. Either a split or data.
    def getVal(self):
        raise NotImplementedError("Subclass must implement getVal()")

#A leaf node will hold data or data indices and a mean label/output value.
class Leaf(Node):
    #Making a leaf needs the data to hold, assumed to be a 2d matrix of samples and features.
    def __init__(self, data):
        super().__init__()
        self.data = data
        #Compute mean of labels/output for the leaf for predictions. 
        self.mean = np.mean(self.data[:,-1])
    #Returns the mean of the label/output of leaf data.    
    def getVal(self):
        return self.mean    

#A split node must have two children and holds split information and two child nodes.
class Split(Node):
    def __init__(self, splitInfo, left=None, right=None):
        super().__init__()
        self.splitInfo = splitInfo
        self.left = left
        self.right = right
    #Returns information to represent the split value and dimension.    
    def getVal(self):
        return self.splitInfo  
    

class RegressionTree:
    def __init__(self, data, height=None, leafSize=None, limit='height'):
        self.limits = dict()
        self.height = height
        self.leafSize = leafSize
        self.limits['height'] = self.height
        self.limits['leaf'] = self.leafSize
        self.limit = self.limits[limit]
        #TODO: Setup tree

    def predict(self, sample):
        #TODO: traverse tree and get prediction
        return None
    
    def decision_path(self, sample):
        #TODO: traverse tree and display path with deduction rules.
        return None
    
#Tests
if __name__ == '__main__':
    data = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    leafTest = Leaf(data)
    print(f"Leaf mean is {leafTest.getVal()}")






