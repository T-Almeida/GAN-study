"""
Colections of helper function for data visualization

some function were found online

credits:

Hvass-Labs - https://github.com/Hvass-Labs/TensorFlow-Tutorials
ageron - rep: https://github.com/ageron/handson-ml/ and book
"""
import matplotlib.pyplot as plt

def plot_mnist_images(images, cls_true, cls_pred=None):
    """

    :param images:
    :param cls_true:
    :param cls_pred:
    :return:
    """
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(28,28), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()