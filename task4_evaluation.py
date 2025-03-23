import numpy as np
import matplotlib.pyplot as plt
from regressionTree import RegressionTree
from multidimensional_regression_tree import train_dimensional_model

def generate_linear_system_data(N=500):
    X = np.random.uniform(low=[-5, -5], high=[5, 5], size=(N, 2))
    Y = np.zeros_like(X)
    Y[:, 0] = 0.9 * X[:, 0] - 0.2 * X[:, 1]
    Y[:, 1] = 0.2 * X[:, 0] + 0.9 * X[:, 1]
    
    return X, Y

def simulate_trajectory(trees, initial_state, steps = 20):
    state = np.array(initial_state)
    trajectory = [state]
    for _ in range(steps):
        next_state = np.array([tree.predict(state) for tree in trees])
        trajectory.append(next_state)
        state = next_state
    return np.array(trajectory)

def true_trajectory(initial_state, steps = 20):
    x = np.array(initial_state)
    traj = [x]
    for _ in range(steps):
        x1 = 0.9 * x[0] - 0.2 * x[1]
        x2 = 0.2 * x[0] + 0.9 * x[1]
        x = np.array([x1, x2])
        traj.append(x)
    return np.array(traj)

def plot_trajectories(true_traj, pred_traj):
    plt.figure()
    plt.plot(true_traj[:, 0], true_traj[:, 1], label='True', marker='o')
    plt.plot(pred_traj[:, 0], pred_traj[:, 1], label='Predicted', marker='x')
    plt.xlabel("x(1)")
    plt.ylabel("x(2)")
    plt.title("Trajectory Comparison (Task 4.1)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__" :
    X, Y = generate_linear_system_data()
    trees, mse = train_dimensional_model(X, Y, height = 6, limit = 'height')
    print("Task 4.1 - Test MSE: ", mse)

    initial_state = [0.5, 1.5]
    true_traj = true_trajectory(initial_state)
    pred_traj = simulate_trajectory(trees, initial_state)

    plot_trajectories(true_traj, pred_traj)

