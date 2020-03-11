import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

# Optional
# import sciplotlib.style as splstyle
# import sciplotlib.polish as splpolish
"""
Pushing beyond stylesheet: writing custom functions to transform the look of your plots.
In this demo I want to show you examples of functions that you can write to semi-automatically transform the look of
your plots. Here we will play around with two functions that make you figure closer to those seen in the figures
found in Nature Review articles.

For convinience, I have also made a python package that contains these functions so you just need to call a package
to apply these styles, see sciplotlib at https://github.com/Timothysit/sciplotlib for more details. You can install it
via 'pip install sciplotlib'
"""


# Again we, start by writing a function that creates some generic plot
def make_figure():

    # Default Plot
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 4)
    num_category = 5
    num_points = 10
    for cat in np.arange(num_category):
        ax.scatter(
            np.random.normal(size=num_points),
            np.random.normal(size=num_points),
            label=cat)
    ax.set_xlabel('X title')
    ax.set_ylabel('Y title')

    return fig, ax


def set_bounds(fig, ax, handle_zero_zero=True):
    """
    Change the limit of axis spines so that they match the first and last tick marks

    Parameters
    -----------
    fig:
    ax:
    handle_zero_zero : bool
        whether to join axis if both have minimum take mark value of zero
        useful for creating unity plots / plots where (0, 0) has a special meaning.
    :return:
    """

    xmin, xmax = ax.get_xlim()
    all_xtick_loc = ax.get_xticks()
    visible_xtick = [t for t in all_xtick_loc if (t >= xmin) & (t <= xmax)]
    min_visible_xtick_loc = min(visible_xtick)
    max_visible_xtick_loc = max(visible_xtick)
    ax.spines['bottom'].set_bounds(min_visible_xtick_loc,
                                   max_visible_xtick_loc)

    ymin, ymax = ax.get_ylim()
    all_ytick_loc = ax.get_yticks()
    visible_ytick = [t for t in all_ytick_loc if (t >= ymin) & (t <= ymax)]
    min_visible_ytick_loc = min(visible_ytick)
    max_visible_ytick_loc = max(visible_ytick)
    ax.spines['left'].set_bounds(min_visible_ytick_loc, max_visible_ytick_loc)

    if handle_zero_zero:
        if min_visible_xtick_loc == 0 and min_visible_ytick_loc == 0:
            ax.set_xlim([0, xmax])
            ax.set_ylim([0, ymax])

    return fig, ax


def apply_gradient(ax,
                   extent,
                   direction=0.3,
                   cmap_range=(0, 1),
                   aspect='auto',
                   **kwargs):
    """
    Draw a gradient image based on a colormap.

    Parameters
    ----------
    ax : Axes
        The axes to draw on.
    extent
        The extent of the image as (xmin, xmax, ymin, ymax).
        By default, this is in Axes coordinates but may be
        changed using the *transform* kwarg.
    direction : float
        The direction of the gradient. This is a number in
        range 0 (=vertical) to 1 (=horizontal).
    cmap_range : float, float
        The fraction (cmin, cmax) of the colormap that should be
        used for the gradient, where the complete colormap is (0, 1).
    **kwargs
        Other parameters are passed on to `.Axes.imshow()`.
        In particular useful is *cmap*.
    """
    if extent is None:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        extent = (xmin, xmax, ymin, ymax)

    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    X = np.array([[v @ [1, 0], v @ [1, 1]], [v @ [0, 0], v @ [0, 1]]])
    a, b = cmap_range
    X = a + (b - a) / X.max() * X
    im = ax.imshow(
        X,
        extent=extent,
        interpolation='bicubic',
        vmin=0,
        vmax=1,
        aspect=aspect,
        **kwargs)
    return im


if __name__ == '__main__':
    with plt.style.context('../stylesheets/nature.mplstyle'):
        fig, ax = make_figure()
        fig, ax = set_bounds(fig, ax)
        apply_gradient(
            ax,
            direction=0.3,
            extent=None,
            cmap_range=(0.125, 0),
            cmap='Greys')
        plt.show()
        fig.savefig('nature-style-w-bells-and-whistles')

    # Doing the same thing in sciplotlib
    """
    with plt.style.context(splstyle.get_style('nature'):
        fig, ax = make_figure()
        fig, ax = splpolish.set_bounds(fig, ax)
        splpolish.apply_gradient(ax, direction=0.3, extent=None, cmap_range=(0.125, 0), cmap='Greys')
        plt.show()
        fig.savefig('nature-style-w-bells-and-whistles')

    """
