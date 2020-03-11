import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
"""
DEMO 4: Overlaying multiple plot objects on a single plot.

"""

if __name__ == '__main__':

    year = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]

    margarine_consumption = [8.2, 7, 6.5, 5.3, 5.2, 4, 4.6, 4.5, 4.2, 3.7]
    divorce_rate = [5, 4.7, 4.6, 5.3, 5.2, 4, 4.6, 4.5, 4.2, 3.7]

    # Combining line plot and scatter
    plt.plot(year, margarine_consumption)
    plt.scatter(year, margarine_consumption)

    plt.show()

    # Plotting multiple lines and scatter
    plt.plot(year, margarine_consumption)
    plt.scatter(year, margarine_consumption)

    plt.plot(year, divorce_rate)
    plt.scatter(year, divorce_rate)

    plt.show()

    # Having multiple y-axis (?)
