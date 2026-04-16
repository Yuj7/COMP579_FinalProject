import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(
        x : np.ndarray,
        y : np.ndarray,
        title : str,
        xlabel : str,
        ylabel : str,
        std_values : np.ndarray | None = None,
        legend_labels : list[str] | None = None,
        nb_ep_smoothing : int | None = None,
        save_plot_dir : str | None = None
) -> None:
    """
    Args:
        x (np.ndarray): Values for x axis ticks
        y (np.ndarray): Values for y axis to be plotted.
                        If array is of dimension 1, it will be assumed these are the strict values to plot.
                        If array is of dimension 2, it will be assumed that axis = 1 are different lines to be plotted.
        title (str): Title of plot
        xlabel (str): Label of the x axis
        ylabel (str): Label of the y axis
        std_values (np.ndarray | None, optional): Standard deviation curves to be plotted.
                                                  If array is of dimension 1, it will be assumed these are the strict values to plot.
                                                  If array is of dimension 2, it will be assumed that axis = 1 are different lines to be plotted.
                                                  Defaults to None which implies not to plot the std curves.
        legend_labels (list[str] | None, optional): Label associated to each line. Length of the list has to be equal to length of the second axis
                                                    of the y argument. Defaults to None.
        nb_ep_smoothing (int | None, optional): Number of episode to smoothe the curves. Defaults to None.
        save_plot_dir (str | None, optional): Directory to save plot. Defaults to None.
    """ 
    if nb_ep_smoothing: 
        for i in range(y.shape[1]):
            y[:, i] = smoothe_curve(nb_ep_smoothing, y[:, i])
            try:
                std_values[:,i] = smoothe_curve(nb_ep_smoothing, y[:,i])
            except:
                pass
        x = y.shape[0]

    if std_values:
        for i in range(y.shape[1]):
            plt.plot(x, y[:, i], label = legend_labels[i])
            plt.fill_between(x, y[:, i] - std_values, y[:, i] + std_values, alpha = 0.3)
    else:
        for i in range(y.shape[1]):
            plt.plot(x, y[:, i], label = legend_labels[i])
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if save_plot_dir:
        plt.savefig(save_plot_dir)
    plt.show()

def smoothe_curve(nb_ep : int, values : np.ndarray) -> tuple[np.ndarray, int]:
    moving_avg = np.convolve(values, np.ones(nb_ep)/nb_ep, mode='valid')
    return moving_avg