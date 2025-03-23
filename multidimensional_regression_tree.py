import numpy as np
from sklearn.model_selection import train_test_split
from regressionTree import RegressionTree

def train_dimensional_model(X,Y,height = None, leafSize = None, limit = 'height'):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    d_out = Y.shape[1]
    trees = []

    #train one tree per the dimension of output
    for i in range(d_out):
        train_data = np.hstack((X_train, Y_train[:, i:i+1]))
        tree = RegressionTree(train_data, height=height, leafSize=leafSize, limit=limit)
        trees.append(tree)

    
    #evaluate
    predictions = []
    for x in X_test:
        pred = [tree.predict(x) for tree in trees]
        predictions.append(pred)

    predictions = np.array(predictions)
    mse = np.mean((predictions - Y_test) ** 2)

    return trees, mse

def generate_vehicle_data(N=500):
    # System: x_k+1 = x_k + 0.1 * v_k, v_k+1 = v_k (velocity constant)
    X = np.random.uniform(low=[-10, 10], high=[10, 10], size=(N, 2))  # state: [x, v]
    Y = np.zeros_like(X)
    Y[:, 0] = X[:, 0] + 0.1 * X[:, 1]
    Y[:, 1] = X[:, 1]
    return X, Y

#simple testing
if __name__ == "__main__":
    X, Y = generate_vehicle_data()
    trees, mse = train_dimensional_model(X, Y, height=5, limit='height')
    print("Test MSE:", mse)

for i in range(5):
    print("Input state:", X[i])
    pred = [tree.predict(X[i]) for tree in trees]
    print("Predicted next state:", pred)
    print("Actual next state:", Y[i])
    print("----")