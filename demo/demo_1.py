import matplotlib.pyplot as plt

"""
Demo 1: Introducition to the matplotlib plt function

Here I provide a minimum example of using matplotlib for plotting.
matplotlib.pyplot takes a minimum of 1 argument, 
corresponding to the y coordinate values of the points you want to plot.


"""

if __name__ == '__main__':

    y_values = [3, 10, 4]

    plt.plot(y_values)

    plt.show()
