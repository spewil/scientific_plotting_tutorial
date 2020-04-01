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

    # Part 2: Some useful overlays and annotations
    # Often you want to highlight particular parts of your plot, such as shading
    # a particular area or adding a shape / arrow somewhere in the plot
    # Here I show you some common functions that you may find useful.

    plt.plot(year, margarine_consumption)
    plt.scatter(year, margarine_consumption)

    plt.plot(year, divorce_rate)
    plt.scatter(year, divorce_rate)

    # Let's say we want to highlight a particular set of years
    plt.fill_betweenx(y=[0, 10], x1=2001, x2=2005, color='gray', alpha=0.2)

    plt.show()


    # Or let's say you want to use shading to indicate your error bar /
    # confidence range of your estimate

    margarine_consumption = [8.2, 7, 6.5, 5.3, 5.2, 4, 4.6, 4.5, 4.2, 3.7]
    margarine_consumption_error = np.array([5, 3, 3, 2, 1, 3, 3, 2, 1, 0.5])
    plt.plot(year, margarine_consumption)
    plt.scatter(year, margarine_consumption)
    plt.fill_between(x=year, y1=margarine_consumption-margarine_consumption_error,
                     y2=margarine_consumption+margarine_consumption_error, color='gray', alpha=0.2)
    plt.show()

    # Text and annotation
    plt.plot(year, margarine_consumption)
    plt.scatter(year, margarine_consumption)
    plt.text(x=2002, y=5, s='Here are some text')
    plt.show()

    plt.plot(year, margarine_consumption)
    plt.scatter(year, margarine_consumption)
    plt.annotate('This was a good year', xy=(2004, 5.2), xytext=(2005, 6),
                 arrowprops=dict(facecolor='black', shrink=0.05)
                 )
    plt.show()


    # Adding shapes: Introducing the idea of Patches
    # Shapes are generally called Patch objects in matplotlib, which
    # have properties (color, edgecolor etc.) that you can modify)
    # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.Patch.html
    # you need to first define these patches as an object, then
    # add them to the plot (Spencer will go into more detail about
    # the objective-oriented view of matplotlib)

    plt.plot(year, margarine_consumption)
    plt.scatter(year, margarine_consumption)
    circle = plt.Circle(xy=(2003, 8), radius=1)
    rectangle = plt.Rectangle(xy=(2006, 3), width=1, height=1.5)
    plt.gca().add_patch(circle)
    plt.gca().add_patch(rectangle)
    plt.show()

    # More advanced detail: many of matplotlib's plots actually returns
    # a set of Patches. For example, the boxplot function of matplotlib
    # returns a set of Patches (corresponding to the boxes). If you want to
    # change their colors for exmaple, you can manually modify those
    # Patch objects

    plt.boxplot([margarine_consumption, divorce_rate], positions=[0, 1],
               widths=0.4, showmeans=False, meanline=True, patch_artist=True,
               medianprops=dict(color='white'), showfliers=False,
               whiskerprops=dict(color='blue'),
               boxprops=dict(facecolor='blue', color='blue',
                             alpha=0.5),
               capprops=dict(color='blue'),
               )
    plt.show()
