# Midterm: Regression Trees

## Overview

This project focuses on understanding and implementing regression trees using the Classification and Regression Tree (CART) algorithm. The objectives include:

- Gaining a comprehensive understanding of regression trees.
- Studying the CART algorithm to construct binary regression trees from training datasets.
- Implementing a basic CART algorithm to build binary regression trees with bounded height.
- Approximating continuous functions using regression trees.
- Approximating dynamical systems using regression trees.

## Implementation Tasks

### 1. Study of Regression Trees and CART Algorithm (10 points)

- **Review the CART Algorithm**: Examine the provided lecture notes to understand the simplified CART algorithm, which does not involve pruning.

- **Implement the `RegressionTree` Class**: Develop a Python class named `RegressionTree` with the following features:

  - **Constructor**: Builds a binary regression tree from a given dataset. The constructor should accept parameters for the dataset, maximum tree height, leaf size, and a variable indicating which limit (height or leaf size) controls the tree's complexity. Both limits can be set to `None` to allow the tree to expand fully, resulting in each node containing exactly one sample.

  - **`predict` Function**: Returns predictions for input samples.

  - **`decision_path` Function**: Displays the sequence of rules leading to a prediction for a given input.

  The "best" feature for a split and the corresponding split value should be determined by measuring the reduction in the sum of squared errors, similar to the strategy used by scikit-learn.

### 2. Function Approximation Using Regression Trees (6 points)

- **Test the Implementation with a Continuous Function**: Evaluate the `RegressionTree` implementation using the function \( y = 0.8 \sin(x - 1) \) over the domain \( x \in [-3, 3] \). Generate approximately 100 uniformly distributed training samples within this domain and compute their corresponding \( y \) values. Split the data into 80% training and 20% testing subsets.

- **Conduct Tests Under Different Conditions**:

  - **Without Any Limitations**: Build the regression tree without imposing any constraints. Record the resulting tree's height, test error, and the time taken to build the tree.

  - **With Height Limitations**: Limit the tree height to 1/2 and 3/4 of the height obtained in the unconstrained case. Record the test results for each scenario.

  - **With Leaf Size Limitations**: Restrict the leaf size to 2, 4, and 8 samples per leaf. Record the test results for each scenario.

  Compile the results into a table to compare the impact of height and leaf size limitations on prediction accuracy.

### 3. Approximating Dynamical Systems (3 points)

- **Design a Method to Approximate Multidimensional Functions**: Using regression trees, approximate functions in discrete-time dynamical systems defined by the difference equation \( \mathbf{x}_{k+1} = f(\mathbf{x}_k) \), where \( \mathbf{x}_i \) represents the system state at time \( t = i \). For example, consider a system where the motion of a vehicle is described by:

  \[
  \begin{align*}
  x_{k+1} &= x_k + 0.1v_k \\
  v_{k+1} &= 10
  \end{align*}
  \]

  Here, \( x \) denotes the position and \( v \) is the constant velocity of 10 m/s. If the exact system model is unknown but a set of paired data in the form \(((x_k, v_k), (x_{k+1}, v_{k+1}))\) is available, explain how a regression tree-based model can be used to predict the system's next state.

### 4. Evaluation of the Regression Tree Model

- **Case Study 1**: Evaluate a system with two state variables, \( x^{(1)} \) and \( x^{(2)} \), defined by the equations:

  \[
  \begin{align*}
  x^{(1)}_{k+1} &= 0.9x^{(1)}_k - 0.2x^{(2)}_k \\
  x^{(2)}_{k+1} &= 0.2x^{(1)}_k + 0.9x^{(2)}_k
  \end{align*}
  \]

  Generate a dataset of samples in the form \(((x^{(1)}, x^{(2)}), (x^{(1)\prime}, x^{(2)\prime}))\), where \((x^{(1)\prime}, x^{(2)\prime})\) represents the next state of \((x^{(1)}, x^{(2)})\). Consider the range \( x^{(1)}, x^{(2)} \in [-5, 5] \). To visualize the approximation quality, create figures showing the trajectories of \( x^{(1)} \) and \( x^{(2)} \) over time. Use the initial state \( x^{(1)}_0 = 0.5 \), \( x^{(2)}_0 = 1.5 \) to predict future states for \( t = 1, 2, \ldots, 20 \), and compare the predicted trajectories with the actual trajectories.

- **Case Study 2**: Analyze the following program:

  ```python
  def func(x):
      z = 0
      for _ in range(20):
          if x > 1:
              x = 0
          else:
              x += 0.2
          z += x


