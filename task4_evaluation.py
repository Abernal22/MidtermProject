import numpy as np
import matplotlib.pyplot as plt
from regressionTree import RegressionTree
from multidimensional_regression_tree import train_dimensional_model

np.random.seed(42)

def generate_linear_system_data(N=500):
    X = np.random.uniform(low=[-5, -5], high=[5, 5], size=(N, 2))
    Y = np.zeros_like(X)
    Y[:, 0] = 0.9 * X[:, 0] - 0.2 * X[:, 1]
    Y[:, 1] = 0.2 * X[:, 0] + 0.9 * X[:, 1]
    
    return X, Y

def generate_program_data(N =3000):
    X = np.random.uniform(low =[-3, 0], high = [3,15], size = (N, 2))
    Y = np.zeros_like(X)
    for i in range(N):
        x, z = X[i]
        if x > 1:
            x_new = 0
        else:
            x_new = x + 0.2
        z_new = z + x_new
        Y[i] = [x_new, z_new]
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

def true_program_trajectory(initial_state, steps = 20):
    x, z = initial_state
    trajectory = [[x,z]]
    for _ in range(steps):
        if x > 1:
            x = 0
        else:
            x += 0.2
        z += x
        trajectory.append([x, z])
    return np.array(trajectory)
    


def plot_trajectories(true_traj, pred_traj, title="Trajectory Comparison"):
    plt.figure()
    plt.plot(true_traj[:, 0], true_traj[:, 1], label='True', marker='o')
    plt.plot(pred_traj[:, 0], pred_traj[:, 1], label='Predicted', marker='x')
    plt.xlabel("x")
    plt.ylabel("z" if "Program" in title else "x(2)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Task 4.1
    X, Y = generate_linear_system_data()
    trees, mse = train_dimensional_model(X, Y, height=6, limit='height')
    print("Task 4.1 - Test MSE: ", mse)

    initial_state = [0.5, 1.5]
    true_traj = true_trajectory(initial_state)
    pred_traj = simulate_trajectory(trees, initial_state)

    plot_trajectories(true_traj, pred_traj)

    # Task 4.2
    print("\n=== Task 4.2: Program Simulation ===")
    X2, Y2 = generate_program_data()

    import time
    start = time.time()
    trees2, mse2 = train_dimensional_model(X2, Y2, height=12, limit='height')
    trees2, mse2 = train_dimensional_model(X2, Y2, leafSize=2, limit='leaf')

    end = time.time()

    print("Task 4.2 - Test MSE:", mse2)
    print("Training time:", round(end - start, 4), "seconds")

    initial_state2 = [2, 0]
    true_traj2 = true_program_trajectory(initial_state2)
    pred_traj2 = simulate_trajectory(trees2, initial_state2)

    plot_trajectories(true_traj2, pred_traj2, title="Trajectory Comparison (Task 4.2 - Program)")

