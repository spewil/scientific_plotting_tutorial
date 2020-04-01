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

    # Histogram plot
    normally_distributed_data = np.random.normal(loc=0, scale=1, size=1000)
    plt.hist(normally_distributed_data)

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


    # What are the different plot types in matplotlib?
    # See: https://matplotlib.org/tutorials/introductory/sample_plots.html

    # Exercise: Make a figure with a different plot type than the one shown here
    # Eg. make a venn diagram, or a pie chart etc.

    ###### Different plot types have different plotting attributes #####

    # Line plot
    # Main attributes to know: linwidth, alpha, linestyle
    # Documentation always useful to check what are the options
    # https://matplotlib.org/gallery/lines_bars_and_markers/line_styles_reference.html

    plt.plot(x, y, linewidth=2, alpha=0.5, linestyle='--')

    plt.show()

    # Scatter plot
    # Main attributes to know: s (size), alpha, marker, facecolor, edgecolor

    plt.scatter(x, y, s=400, alpha=0.5, facecolor='red', edgecolors='black',
               marker='o')
    # maker options: https://matplotlib.org/3.1.1/api/markers_api.html

    plt.show()

    # Bar plot
    # Main attributes to know: color, edgecolor, linewidth,

    plt.bar(x, y, color='blue', edgecolor='green', linewidth=3)

    plt.show()

    # Histogram plot
    # Main attributes to know: bins

    normally_distributed_data = np.random.normal(loc=0, scale=1, size=1000)
    plt.hist(normally_distributed_data, bins=100)

    plt.show()

