import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
"""
Demo 2: Introduction to the different plot types

In this demo, we want to load some data with x and y coordinate values and plot them.
We will experiment with different ways for plotting the same data.
"""

if __name__ == '__main__':

    x = [1, 2, 3, 4, 5]
    y = [3, 6, 1, 3, 5]

    # Line plot

    plt.plot(x, y)

    plt.show()

    # Scatter plot

    plt.scatter(x, y)

    plt.show()

    # Bar plot

    plt.bar(x, y)

    plt.show()

    # Heatmap / Image plot
    # Note: the input matrix to imshow can be three things
    # 1) (M, N): standard 2D matrix, where M is the number of rows anb N the number of columns,
    # in which case the entry are scalar values, eg. spike rate of particular neuron at a particular time bin
    # 2) (M, N, 3): image with RGB values (useful for showing images), RGB values vary from 0 - 1 or 0 - 255
    # 3) (M, N, 4): image with RGB values and a further transparency value that vary from 0 - 1 or 0 - 255

    example_matrix = np.load('../data/example_matrix.npy')
    print('Shape of numpy array: ')
    print(np.shape(example_matrix))

    plt.imshow(example_matrix)

    plt.show()
