"""
Colections of helper function for data visualization

some function were found online

credits:

Hvass-Labs - https://github.com/Hvass-Labs/TensorFlow-Tutorials
ageron - rep: https://github.com/ageron/handson-ml/ and book
"""
import matplotlib.pyplot as plt

def plot_mnist_images(images):
    """

    :param images: matrix (M,K) m = images number, K = images flated pixeis
    :return:
    """
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(28,28), cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()



def plot_mnist_images_conditonal(images,per_class=6):
    """

    :param images: matrix (M,K) m = images number, K = images flated pixeis
    :return:
    """
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(10, per_class)
    #fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(28,28), cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()