import matplotlib.pyplot as plt
import numpy as np

# Data
data = [1, 5, 7, 3, 5, 7, 8, 8]

# Create a plot
plt.figure(figsize=(10, 5))
plt.plot(data, marker='o')
plt.title('Graph of Given Array')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)

# Save the plot to a file
plt.savefig('array_graph.png')
plt.close()