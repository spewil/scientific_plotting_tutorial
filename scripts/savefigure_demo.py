import os
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
"""
DEMO 5: Saving and exporting figures in matplotlib.
"""

if __name__ == '__main__':

    # Default Saving behaviour in matplotlib

    fig, ax = plt.subplots()
    num_points = 1000
    ax.scatter(
        np.random.normal(size=num_points), np.random.normal(size=num_points))
    fig_name = 'default'
    fig.savefig(fig_name)

    # Let's try opening the plot, we see two things
    # (1) The plot is saved as a .png file by default
    # (2) The resolution of the plot is not very good; zoom in and you see
    # things get pixelated rather quickly

    # Saving the plot with a specified size

    fig, ax = plt.subplots()
    figure_width = 6
    figure_height = 4
    fig.set_size_inches(figure_width, figure_height)
    num_points = 1000
    ax.scatter(
        np.random.normal(size=num_points), np.random.normal(size=num_points))
    fig_name = 'custom_size'
    fig.savefig(fig_name)

    # Saving the plot with a specified dots per inch

    fig, ax = plt.subplots()
    figure_width = 6
    figure_height = 4
    fig.set_size_inches(figure_width, figure_height)
    num_points = 1000
    ax.scatter(
        np.random.normal(size=num_points), np.random.normal(size=num_points))
    fig_name = 'custom_dpi'
    fig.savefig(fig_name, dpi=300)

    # Saving the plot so you can edit it elsewhere and don't have to
    # worrry about dpi

    fig, ax = plt.subplots()
    figure_width = 6
    figure_height = 4
    fig.set_size_inches(figure_width, figure_height)
    num_points = 1000
    ax.scatter(
        np.random.normal(size=num_points), np.random.normal(size=num_points))
    fig_name = 'scatter_w_svg'
    fig_ext = '.svg'
    fig.savefig(fig_name + fig_ext)

    # Making sure all plot elements gets saved
    # Sometimes your plot contains objects that are outside of the axis
    # For example
    # (1) your x and y-axis may be too big
    # (2) your legend / colorbar is outside of the figure axis
    # This can be resolved via a simple parameter: bbox_inches


    def make_oversize_plot(num_cateogry=10, num_points=100):
        fig, ax = plt.subplots()
        for cat in np.arange(num_cateogry):
            x = np.random.normal(size=num_points)
            y = np.random.normal(size=num_points)
            ax.scatter(x, y, label=cat)

        ax.set_xlabel(r'$\mathcal{N}(0, 1)$')
        ax.set_ylabel(r'$\int\frac{\mathcal{N}(0, 1)}{e + \alpha}$')
        ax.set_title('A massive title', size=60)
        ax.legend(bbox_to_anchor=[1.04, 1.04], title='Legend title', ncol=2)

        return fig, ax

    # Original plot
    fig, ax = make_oversize_plot()
    fig_name = 'cropped_figure_example'
    fig_ext = '.png'
    fig.savefig(fig_name + fig_ext, dpi=300)

    # Not saved but ensuring all elements are kept inside the plot
    fig, ax = make_oversize_plot()
    fig_name = 'uncropped_figure_example'
    fig_ext = '.png'
    fig.savefig(fig_name, bbox_inches='tight')

    # Optional: save plot as pickle
    # Note that saving figure as pickle is normally not recommended, as there is no gurantee that you will be able
    # to load the plot in other versions of matplotlib.
    # However, it is useful if you somehow need to save some plot in the short term and edit it later,.
    # Normally, it is better practice to save the processed data just before making the plot, and load that
    # to make plots.

    fig, ax = make_oversize_plot()
    ax.set_title('Old title')
    fig.show()
    with open('figure.pkl', 'wb') as handle:
        pkl.dump((fig, ax), handle)

    # Load the pickle figure back
    with open('figure.pkl', 'rb') as handle:
        fig, ax = pkl.load(handle)

    ax.set_title('New title')
    fig.show()
    '''
        a function I often use for saving
    '''

    def save_fig(fig, save_dir, name):
        image_path = os.path.join(*[save_dir, name + ".png"])
        fig.savefig(
            image_path,
            dpi=150,
            facecolor='w',
            edgecolor='w',
            orientation='portrait',
            papertype=None,
            format=None,
            transparent=True,
            bbox_inches=None,
            pad_inches=0,
            frameon=False,
            metadata=None)
        return image_path
