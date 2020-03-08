"""
DEMO 3 : Setting plot attributes

In this demo, we demonstrate how you can set the many attributes of a plot.
"""
import matplotlib.pyplot as plt

if __name__ == '__main__':

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    year = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
    y = [8.2, 7, 6.5, 5.3, 5.2, 4, 4.6, 4.5, 4.2, 3.7]

    # Original plane plot
    plt.plot(x, y)
    plt.show()


    # Set x label
    plt.plot(x, y)
    plt.xlabel('Time')
    plt.show()

    # Set y label
    plt.plot(x, y)
    plt.xlabel('Time')
    plt.ylabel('Margarine consumed (pounds)')
    plt.show()

    # Set title of plot
    plt.plot(x, y)
    plt.xlabel('Year')
    plt.ylabel('Margarine consumed (pounds)')
    plt.title('Per capita consumption of margarine in Maine')
    plt.show()


    # Set custom xticks
    plt.plot(x, y)
    plt.xlabel('Year')
    plt.ylabel('Margarine consumed (pounds)')
    plt.title('Per capita consumption of margarine in Maine')
    plt.xticks(ticks=x[::2], labels=year[::2])
    plt.show()


    # Set the color of the line
    plt.plot(x, y, color='red')
    plt.xlabel('Year')
    plt.ylabel('Margarine consumed (pounds)')
    plt.title('Per capita consumption of margarine in Maine')
    plt.xticks(ticks=x[::2], labels=year[::2])
    plt.show()


    # Set the transparency of the line
    plt.plot(x, y, color='red', alpha=0.5)
    plt.xlabel('Year')
    plt.ylabel('Margarine consumed (pounds)')
    plt.title('Per capita consumption of margarine in Maine')
    plt.xticks(ticks=x[::2], labels=year[::2])
    plt.show()


    # Set the thickness of the line
    plt.plot(x, y, color='red', alpha=0.5, linewidth=3)
    plt.xlabel('Year')
    plt.ylabel('Margarine consumed (pounds)')
    plt.title('Per capita consumption of margarine in Maine')
    plt.xticks(ticks=x[::2], labels=year[::2])
    plt.show()


    # Set the font size of labels and titles
    plt.plot(x, y, color='red', alpha=0.5, linewidth=3)
    plt.xlabel('Year', size=10, color='blue')
    plt.ylabel('Margarine consumed (pounds)', size=10, color='red')
    plt.title('Per capita consumption of margarine in Maine', size=20)
    plt.xticks(ticks=x[::2], labels=year[::2])
    plt.show()


    # Set the axis limits
    plt.plot(x, y, color='red', alpha=0.5, linewidth=3)
    plt.xlabel('Year', size=10, color='blue')
    plt.ylabel('Margarine consumed (pounds)', size=10, color='red')
    plt.title('Per capita consumption of margarine in Maine', size=20)
    # plt.xticks(ticks=x[::2], labels=year[::2])
    plt.xlim([0, 6])
    plt.show()


    # Summary: anatomy of a matplotlib plot

