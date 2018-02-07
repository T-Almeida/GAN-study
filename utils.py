"""
Colections of helper function for data visualization

some function were found online

credits:

Hvass-Labs - https://github.com/Hvass-Labs/TensorFlow-Tutorials
ageron - rep: https://github.com/ageron/handson-ml/ and book
"""
import matplotlib.pyplot as plt
import numpy as np
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


def plot_mnist_images_label(images, cls_true, cls_pred=None):
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


def one_hot(label):
    '''
    Apply one hot encoding to the label

    :param label: Vector with shape (Mx1)
    :return: Matrix with shape (Mx10)
    '''
    # print(len(label))
    m = np.zeros((len(label), 10))
    m[np.arange(len(label)), label] = 1
    return m


def random_Z(m, n=100):
    '''
    Random values for Z between -1 and 1

    :param m: number of samples
    :param n: dimension of Z
    :return: Vector with shape (m,n)
    '''
    return np.random.uniform(-1., 1., size=[m, n])


def random_y(m):
    '''
    Random labels y in one hot encoding

    :param m: number of samples
    :param n: dimension of Z
    :return: Vector with shape (m,n)
    '''
    return one_hot(np.random.randint(10, size=[m]))

def max_row_index(x):
    '''
    Given matrix (m,c) return the max index

    :param x:
    :return:
    '''
    return np.argmax(x,axis=1)