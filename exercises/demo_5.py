import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
"""
DEMO 5: Subplots using plt
There are multiple ways of making subplots in matplotlib.
Here we demonstrate a matlab-like way to make subplots, and note that
there are inconviniences that can be resolved by thinking of plots
as objects instead.
"""

if __name__ == '__main__':

    year = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]

    margarine_consumption = [8.2, 7, 6.5, 5.3, 5.2, 4, 4.6, 4.5, 4.2, 3.7]
    divorce_rate = [5, 4.7, 4.6, 5.3, 5.2, 4, 4.6, 4.5, 4.2, 3.7]

    plt.subplot(121)
    plt.plot(margarine_consumption)

    plt.subplot(122)
    plt.plot(divorce_rate)

    # Let's say we now want to change some property of the first plot
    # We then need to declare that we are editing the first subplot
    plt.subplot(121)
    plt.plot(margarine_consumption)

    plt.subplot(122)
    plt.plot(divorce_rate)

    plt.subplot(1, 2, 1)
    plt.ylabel('Margarine consumption')

    plt.subplot(1, 2, 2)
    plt.ylabel('Divorce rate')

    plt.show()
    plt.close()

    # Subplots in a loop

    x0 = [1, 2, 3, 4, 5]
    y0 = [1, 2, 3, 4, 5]
    y1 = [2, 4, 6, 8, 10]
    y2 = [3, 5, 7, 9, 11]
    y3 = [4, 6, 8, 10, 12]
    y_list = [y0, y1, y2, y3]
    num_row = len(y_list)

    for n, y_val in enumerate(y_list):
        y_val += np.random.random(len(y_val))
        plt.subplot(1, num_row, n + 1)
        plt.title('Plot ' + str(n + 1))
        plt.scatter(x0, y_val)
        plt.plot(x0, y_val)

    plt.show()

    # Exercise, re-do the above plot, but instead have two rows and two columns
    # Notice how matplotlib orders the plots
