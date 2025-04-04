import numpy as np
import time
from sklearn.model_selection import train_test_split

from regressionTree import RegressionTree
# Function
def target_function(x):
    return 0.8 * np.sin(x - 1)


# Generate dataset
np.random.seed(42)
x = np.random.uniform(-3, 3, 100)
y = target_function(x)
data = np.column_stack((x, y))

# Split data into training (80%) and testing (20%)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Test different limits

# First train an unlimited tree to determine its actual height
unlimited_tree = RegressionTree(train_data)
actual_height = unlimited_tree.treeHeight()

# Compute 1/2 and 3/4 of the actual tree height
height_half = actual_height // 2
height_three_fourths = (3 * actual_height) // 4


# 1/2 of log(100) about 3, 3/4 is about 5.
limits = [(None, None), (height_half, None), (height_three_fourths, None), (None, 2), (None, 4), (None, 8)]
results = []

for height, leaf_size in limits:
    start_time = time.time()
    tree = RegressionTree(train_data, height=height if height is not None else None,
                          leafSize=leaf_size if leaf_size is not None else None,
                          limit='height' if height is not None else 'leaf')
    end_time = time.time()
    print(tree)
    print()

    # Test error
    predictions = np.array([tree.predict(sample) for sample in test_data])
    test_error = np.mean((predictions - test_data[:, 1]) ** 2)

    # Store results
    results.append((height, leaf_size, test_error, end_time - start_time))

# Print results
table_header = "| Height Limit | Leaf Size Limit | Test Error | Time Taken (s) |"
table_divider = "|-------------|----------------|------------|----------------|"
print(table_header)
print(table_divider)
for height, leaf_size, error, time_taken in results:
    print(
        f"| {height if height is not None else 'None':^11} | {leaf_size if leaf_size is not None else 'None':^14} | {error:^10.4f} | {time_taken:^14.4f} |")
