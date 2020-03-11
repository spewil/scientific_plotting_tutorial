"""
OO Version of Demo 3

In this demo, we demonstrate how you can set the many attributes of a plot.
"""
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

if __name__ == '__main__':

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    year = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
    y = [8.2, 7, 6.5, 5.3, 5.2, 4, 4.6, 4.5, 4.2, 3.7]

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    plot_handle = ax.plot(x, y)
    ax.set_xlabel('Year', size=10, color='blue')
    ax.set_ylabel('Margarine consumed (pounds)', size=10, color='red')
    ax.set_title('Per capita consumption of margarine in Maine')
    # ax.set_xticks(ticks=x[::2])
    # ax.set_xticklabels([str(y) for y in year[::2]])
    # plot_handle[0].set_color('red')
    # plot_handle[0].set_alpha(0.5)
    # plot_handle[0].set_linewidth(10)
    # ax.set_xlim([0, 6])

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    plt.show()
