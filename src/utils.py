import math
import matplotlib.pyplot as plt

def plot_grid(columns, plot_func, ncols=None, figsize_scale=(5, 4)):
    n = len(columns)

    if ncols is None:
        ncols = math.ceil(math.sqrt(n))

    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(figsize_scale[0]*ncols, figsize_scale[1]*nrows))

    axes = axes.flatten()

    for i, col in enumerate(columns):
        plot_func(col, axes[i])

    # remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()