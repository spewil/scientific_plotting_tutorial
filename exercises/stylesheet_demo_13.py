import matplotlib.pyplot as plt
import numpy as np

"""
Demo: Custom stylesheet and functions for modifying plots.

References: 
 - Matplotlib style gallery: https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
"""

def make_figure():

    # Default Plot
    fig, ax = plt.subplots(dpi=50)
    # you should set dpi to at least 300 for saving
    # I have set it to 50 here for easier display on a small screen.
    ax.scatter(np.random.normal(size=100), np.random.normal(size=100))
    ax.set_xlabel('X title')
    ax.set_ylabel('Y title')
    fig.set_size_inches(4, 4)

    return fig, ax

if __name__ == '__main__':

    # Show the default plotting behaviour

    fig, ax = make_figure()
    plt.show()

    # Some built in styles

    with plt.style.context(['dark_background']):
        fig, ax = make_figure()
        plt.show()

    # Some more built in styles
    with plt.style.context(['fivethirtyeight']):
        fig, ax = make_figure()
        plt.show()


    # Combining built in styles
    with plt.style.context(['fivethirtyeight', 'dark_background']):
        fig, ax = make_figure()
        plt.show()

    # Some special cases
    with plt.xkcd():
        fig, ax = make_figure()
        plt.show()



    # Writing your own custom stylesheet
    """
    In this part, we will make a copy of the default.mplstyle and 
    make some edits to modify the default plotting parameters of matplotlib.
    """

    # Running a custom stylesheet that I wrote

    stylesheet_path = '../stylesheets/nature-reviews.mplstyle'
    with plt.style.context(stylesheet_path):
        fig, ax = make_figure()
        plt.show()

