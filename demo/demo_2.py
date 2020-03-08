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

    example_matrix = np.load('/home/timsit/matplotlib-tutorial/data/example_matrix.npy')
    print('Shape of numpy array: ')
    print(np.shape(example_matrix))

    plt.imshow(example_matrix)

    plt.show()

