import numpy as np

class Node:
    #A node is supposed to be either a leaf node or a split node and have its own setup
    def __init__(self, parent):
        self.parent = parent
    #Each node has some kind of overall value. Either a split or data.
    def getVal(self):
        raise NotImplementedError("Subclass must implement getVal()")


#A leaf node will hold data or data indices and a mean label/output value.
class Leaf(Node):
    #Making a leaf needs the data to hold, assumed to be a 2d matrix of samples and features.
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        #Compute mean of labels/output for the leaf for predictions. 
        self.mean = np.mean(self.data[:,-1])
    #Returns the mean of the label/output of leaf data.    
    def getVal(self):
        return self.data
    def getMean(self):
        return self.mean
    def __str__(self):
        return f"Data: {self.data}, mean: {self.mean}"    

#A split node must have two children and holds split information and two child nodes.
class Split(Node):
    def __init__(self, splitInfo, left=None, right=None, parent=None):
        super().__init__(parent)
        self.splitInfo = splitInfo
        self.left = left
        self.right = right
    #Returns information to represent the split value and dimension.    
    def getVal(self):
        return self.splitInfo
    def __str__(self):
        return f"Split: {self.splitInfo}, Left: {str(self.left)}, Right: {str(self.right)}"  
    

class RegressionTree:
    def __init__(self, data, height=None, leafSize=None, limit='height'):
        self.limits = dict()
        self.height = height
        self.leafSize = leafSize
        self.limits['height'] = self.height
        self.limits['leaf'] = self.leafSize
        self.limit = self.limits[limit]
        self.root = Leaf(data)
        stack = [(self.root,True, 0)]
        hlim = limit == 'height'
        #Loop and build tree
        while len(stack) > 0:
            #remove from stack and update if needed.
            current, dir, height = stack.pop()
            #First determine varience.
            #If leaf has no varience we can continue to the next element
            if self.leafError(current) == 0:
                continue
            #Leaf limiter
            if not hlim:
                if(current.data.shape[0] <= self.limit):
                    continue
            else:
                if not (self.limit == None):
                    if self.limit >=0:
                        if height >= self.limit:
                            continue    
            #Determine best split available.
            dim, val, left, right = self.bestSplit(current)
            #Assign all parents and children then add to stack.
            #Change the current node to a split node.
            newNode = Split((dim,val), left, right)
            left.parent = newNode
            right.parent = newNode
            if current == self.root:
                self.root = newNode
            else:    
                parent = current.parent
                newNode.parent = parent
                if dir:
                    parent.right = newNode
                else:
                    parent.left = newNode
            stack.append((right, True, height+1))
            stack.append((left,False,height+1)) 
    def __str__(self):
        return str(self.root)


    #Sample is assumed to be a 1d array
    def predict(self, sample):
        #Loop starting at root to find sample.
        #If we reach a leaf we have found our prediction.
        current = self.root
        while current != None:
            if isinstance(current, Leaf):
                #We have a leaf and prediction, return mean as prediction
                return current.getMean()
            #If not leaf we can change the node to a child by looking at the split value.
            #If less than or equal split value go left otherwise right.
            #Split node value must be a tuple with feature and value to split at.
            feature, value = current.getVal()
            sVal = sample[feature]
            if sVal <= value:
                current = current.left
            else:
                current = current.right    

        #Should not return none unless there is a problem.        
        return None
    
    def decision_path(self, sample):
        #Loop starting at root to find sample and display path
        current = self.root
        print("Decision path:")
        while current != None:
            if isinstance(current, Leaf):
                #We have a leaf and prediction, return mean as prediction and print leaf
                print(f"Prediction: {current.getMean()}")
                return current.getMean()
            #If not leaf we can change the node to a child by looking at the split value.
            #If less than or equal split value go left otherwise right.
            #Split node value must be a tuple with feature and value to split at.
            feature, value = current.getVal()
            sVal = sample[feature]
            if sVal <= value:
                #Display feature and val path
                print(f"Sample: (feature {feature}, value {sVal}) <= Split: (feature {feature}, value {value})")
                current = current.left
            else:
                print(f"Sample: feature {feature}, value {sVal} > Split: (feature {feature}, value {value})")
                current = current.right 
        return None
    #Assumed to be a leaf node to determine varience. 
    def leafError(self, node):
        error = 0
        #Assumed to be samples.
        data = node.getVal()
        mean = node.getMean()    
        for sample in data:
            error += (sample[-1] - mean)**2
        #Within leaf error is returned not including sample size.    
        return error
    
    #returns best split with leafs and does not modify structures.
    def bestSplit(self, leaf):
        #Loop over all possible dimensions and values of given data to find split.
        data = leaf.getVal()
        #Excludes last column as output.
        maxDim = data.shape[1] - 1
        n = data.shape[0]
        #Loop over features.
        dim = -1
        val = -1
        leftEnd = None
        rightEnd = None
        #The original leaf error including sample size.
        lowestError = (n+1)*self.leafError(leaf)
        for i in range(n):
            sample = data[i]
            for x in range(maxDim):
                value = sample[x]
                left, right = self.splitLeaf(leaf, x, value)
                if left == None or right == None:
                    continue
                lSize = left.data.shape[0]
                rSize = right.data.shape[0]
                #sum of squared errors
                currentError = (lSize*self.leafError(left)) + (rSize*self.leafError(right))
                #compare error
                if currentError <= lowestError:
                    #update values if we have a lower or equal error.
                    lowestError = currentError
                    dim = x
                    val = value
                    leftEnd = left
                    rightEnd = right

        return dim, val, leftEnd, rightEnd            





    #Splits a leaf in a specific dimension and value and returns two new leafs.
    #Does not modify any structures.
    def splitLeaf(self, leaf, dimension, value):
        data = leaf.getVal()
        indices = data[:, dimension] <= value
        #Spliting the values based on a dimension and given value.
        left = data[indices]
        right = data[~indices]
        if len(left) == 0:
            return None, Leaf(right)
        if len(right) == 0:
            return Leaf(left), None
        return Leaf(left), Leaf(right)




    
    
#Tests
if __name__ == '__main__':
    data = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    leafTest = Leaf(data)
    print(f"Leaf mean is {leafTest.getMean()}")
    #Test tree setup
    tree = RegressionTree(data)
    print(tree.root)
    left, right = tree.splitLeaf(leafTest, 0, 1)
    print(left)
    print(right)
    #Trying best split.
    dim, val, left, right = tree.bestSplit(leafTest)
    print(f"Best split ({dim},{val}) Leaves: {left}, {right}")
    #Testing obvious split
    data = np.array([[1,10],[2,12],[3,11],[4,20],[5,22],[6,21]])
    leafTest = Leaf(data)
    tree = RegressionTree(data)
    dim, val, left, right = tree.bestSplit(leafTest)
    print(f"Best split ({dim},{val}) Leaves: {left}, {right}")
    print()
    print("Testing Full Tree")
    print(tree)
    data = np.array([[1,5],[2,6],[3,12],[4,10],[5,30],[6,29], [7,35], [8,36]])
    tree = RegressionTree(data)
    print()
    print("Testing different split")
    print(tree)
    print()
    print("Testing leaf limit 4:")
    print(RegressionTree(data, leafSize=4, limit = 'leaf'))
    print()
    print("Testing leaf limit 2:")
    print(RegressionTree(data, leafSize=2, limit = 'leaf'))
    print()
    print("Testing height limit 0")
    print(RegressionTree(data, height=0))
    print()
    print("Testing height limit 1")
    print(RegressionTree(data, height=1))

    #testing prediction
    data = np.array([[1,5],[2,6],[3,12],[4,10],[5,30],[6,29], [7,35], [8,36]])
    tree = RegressionTree(data, height=2)
    print()
    print("Testing prediction of [1,5], should be 5.5 with height of 2")
    print(tree.predict(data[0]))

    #Testing path
    tree.decision_path(data[2])














