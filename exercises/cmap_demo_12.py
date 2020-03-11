import matplotlib.pyplot as plt
import numpy as np

"""
Demo: Colormaps in matplotlib.
"""
if __name__ == '__main__':

    # Part 1: Why Jet is to be avoided unless you have a specific reason for using it
    num_data_points = 400
    ls = np.linspace(0, 10, num_data_points)
    x, y = np.meshgrid(ls, ls)

    noiseless_data = np.sin(x) * np.cos(y)
    noise_level = 0.1
    noisy_data = noiseless_data + noise_level * np.random.rand(num_data_points, num_data_points)

    colormap_name = 'Greys'
    fig, axs = plt.subplots()
    img1 = axs.imshow(noiseless_data, cmap=colormap_name)
    # img2 = axs[1].imshow(noisy_data, cmap=colormap_name)
    cbar2 = fig.colorbar(img2)
    plt.show()



    colormap_name = 'jet'
    fig, axs = plt.subplots()
    # img1 = axs[0].imshow(noiseless_data, cmap=colormap_name)
    img2 = axs.imshow(noisy_data, cmap=colormap_name)
    cbar2 = fig.colorbar(img2)
    plt.show()

    # Perceptually uniform colormaps
    colormap_name = 'viridis'
    # fig, axs = plt.subplots(1, 2)
    fig, ax = plt.subplots()
    # img1 = axs[0].imshow(noiseless_data, cmap=colormap_name)
    # img2 = axs[1].imshow(noisy_data, cmap=colormap_name)
    img = ax.imshow(noisy_data, cmap=colormap_name)
    cbar = fig.colorbar(img)
    plt.show()


    colormap_name = 'inferno'
    fig, axs = plt.subplots(1, 2)
    img1 = axs[0].imshow(noiseless_data, cmap=colormap_name)
    img2 = axs[1].imshow(noisy_data, cmap=colormap_name)
    plt.show()


    # Part 2: Different types of colormap in matplotlib
    # From part 1, we saw an example of a continous, sequential colormap, we now consider other types of
    # colormap you may find useful
    # You can find all the info here: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

    # For showing value ranges where negative and positive values have distinct meanings - for example
    # negative means that a neuron is suppressed, whereas positive means that a neuron is enhanced, it is useful
    # to use a divergent colormap to illustrate that there are two 'axis' being plotted:
    # (1) degree of enhacement and (2) degree of suppression

    fig, ax = plt.subplots()
    num_data_points = 30
    all_neuron_mean_response_matrix = np.random.normal(size=(num_data_points, num_data_points))
    heatmap = ax.imshow(all_neuron_mean_response_matrix, cmap='seismic')
    cbar = fig.colorbar(heatmap)


    # For showing categorical variables, we use a categorical colormap, here, the objective is to maximise how much
    # different categories can be distinguished, and therefore it is better to use something that maximises contrast
    # (rather than a perceptually uniform colormap)

    # This demo serves a secondary purpose, sometimes there are objects that you want to color that does not have
    # the cmap parameter, in that case, will need to obtain the list of colors  and manually apply them to the objects
    # that we want

    fig, ax = plt.subplots()
    # Box plot
    neuron_1 = [3, 2, 5, 7, 1]
    neuron_2 = [4, 2, 4, 6, 7]
    neuron_3 = [7, 12, 6, 8, 1]

    # setting patch_artist to true allows editing of specific properties of the boxplot boject
    my_box_plot = ax.boxplot([neuron_1, neuron_2, neuron_3], patch_artist=True)

    colormap_name = 'Set1'
    cmap = plt.get_cmap(colormap_name)
    for patch_number, patch in enumerate(my_box_plot['boxes']):
        patch.set_facecolor(cmap(patch_number))

    plt.show()
