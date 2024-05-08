import pandas as pd
import numpy as np
import sys
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io

def plot_incidence_order(I, xlabel="Order (Unique 1Mb Bins)"):
    """Calculates degree statistics and generates a histogram.

    Args:
        I: A matrix representing connectivity information.
        xlabel: Label for the x-axis of the histogram.
    """

    degree = I.sum(axis=0)
    print(f"{degree.mean()=}")
    print(f"{np.median(degree)=}")
    
    sns.histplot(x=degree,  
                 discrete=True,
                 stat='percent')

    plt.xticks(list(range(1, 15)))  # Assuming 'degree' can range to 15
    plt.xlabel(xlabel)


def floats_to_colors(values, vmin=None, vmax=None, colormap='viridis'):
    """Converts a list of floating-point numbers to a list of Matplotlib color representations.

    Args:
        values (list of float): The list of floating-point numbers to convert.
        vmin (float, optional): The minimum value for normalization. If None, calculated from the input data.
        vmax (float, optional): The maximum value for normalization. If None, calculated from the input data.
        colormap (str or Colormap, optional): The colormap to use. Defaults to 'viridis'.

    Returns:
        list: A list of RGBA color tuples.
    """

    if vmin is None:
        vmin = float(min(values))
    if vmax is None:
        vmax = float(max(values))

    def _normalize(value, vmax, vmin):
        if vmin == vmax: # Handle potential division by zero
            return 0.5   # Return a neutral value if the input range is degenerate 
        return (float(value) - vmin) / (vmax - vmin)

    normalized_values = [_normalize(val, vmax, vmin) for val in values]
    cmap = plt.get_cmap(colormap)
    colors = cmap(normalized_values)
    return colors


def plot_incidence(df, ax=None, node_color='k', line_color='k', node_params={}, line_params={}):
    """Plots an incidence matrix with customizable colors and parameters.

    Args:
        df (pd.DataFrame): DataFrame representing the incidence matrix.
        ax (matplotlib.axes.Axes, optional): An existing axis to plot on. If None, creates a new figure.
        node_color (str or color-like, optional): Default color for nodes. Defaults to 'k' (black).
        line_color (str or color-like, optional): Default color for lines. Defaults to 'k' (black).
        node_params (dict, optional): Keyword arguments for plt.scatter (nodes).
        line_params (dict, optional): Keyword arguments for plt.plot (lines). 
    """
    if ax is None:
        fig, ax = plt.subplots()

    num_columns = len(df.columns)
    
    node_params.pop('c', None)  
    line_params.pop('c', None) 

    if isinstance(node_color, str):
        node_color = [node_color] * num_columns
    if isinstance(line_color, str):
        line_color = [line_color] * num_columns

    
    for i, column in enumerate(df.columns):
        hyperedge = df[column][df[column] > 0]
        order = len(hyperedge)
        x_ind = np.ones(order) * i
        node_cvec = [node_color[i]] * order
        line_cvec = [line_color[i]] * order

        # Plot nodes
        ax.scatter(x_ind, 
                   hyperedge.index, 
                   c=node_cvec,
                   **node_params)

        # Plot edges
        ax.plot(x_ind, hyperedge.index, 
                c=line_color[i],
                **line_params) 

    ax.invert_yaxis()
    

def scree_plot(matrix, normalize=False, grid=True, title="", **kwargs):
    """
    Calculates the singular values of a matrix and plots a scree plot,
    with the option to plot the optimal hard threshold (OHT) as a vertical line.

    Args:
        matrix (np.ndarray): The input matrix.
        normalize (bool, optional): If True, normalize singular values for cumulative variance.  
                                    Defaults to False.
        grid (bool, optional): If True, display a grid on the plot. Defaults to True. 
        title (str, optional): The title for the entire set of plots. Defaults to ''.
        **kwargs: Additional keyword arguments to pass to plt.plot for line customization. 
    """

    # Calculate singular values and OHT
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)

    # Normalize for cumulative variance, if desired
    if normalize:
        s = s / s.sum()

    fig, ax = plt.subplots()

    ax.plot(s, **kwargs) 
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value")
    ax.grid(grid)

    plt.suptitle(title)
    plt.tight_layout()


def plot_ml_metrics_over_epochs(df):
    """Plots model performance metrics over training epochs.

    Args:
        df (pd.DataFrame): A DataFrame containing columns for 'epoch', 'auc', 
                           'precision', 'accuracy', and 'f1score'.
    """

    metrics = ['AUC', 'Precision', 'Accuracy', 'F1 Score']
    fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True, sharex=True)

    # Iterate through metrics and plot
    for i, metric in enumerate(metrics):
        row = i // 2  # Calculate row index
        col = i % 2  # Calculate column index
        axes[row, col].plot(df['epoch'], df[metric])
        axes[row, col].set_title(metric.capitalize())  

    # Overall figure adjustments
    fig.suptitle('Learning Metrics over Epochs')
    sns.despine() 
    plt.tight_layout()
    plt.show()  # Add plt.show() to display the plot 
    

def fig2img(fig, dpi=300):
    """Converts a Matplotlib figure to a PIL Image with a white background.

    Args:
        fig: The Matplotlib figure to convert.
        dpi (int, optional): The resolution of the output image in dots per inch.
            Defaults to 100.

    Returns:
        PIL.Image: The converted image with a white background.
    """

    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", dpi=dpi)  # Specify PNG format and DPI
        buf.seek(0)
        img = Image.open(buf)

        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Paste with alpha channel

        return background


def plot_incidence_simple(ax, data, **kwargs):
    """
    Plots read codes against bins as a line graph with markers on the provided axes.
    
    Args:
      ax (matplotlib.axes.Axes): Existing axes object for plotting.
      data (dict): Dictionary containing data with keys 'read_code' (list) and 'bin' (list of lists).
      **kwargs: Additional keyword arguments passed to plt.plot().
    
    Returns:
      None
    """
    
    read_codes = data["read_code"]
    bins_list = data["bin"]
    
    for i, bins in enumerate(bins_list):
      sorted_bins = sorted(bins)
      ax.plot([i] * len(sorted_bins), sorted_bins,  **kwargs)
    
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', which='both', axis='both')
    ax.invert_yaxis()


def _plot_rectangle(ax, x, y, height, width, **kwargs):
  """A function to plot a rectangle.

  Args:
      ax: The matplotlib Axes object where the rectangle will be plotted.
      x: The x-coordinate of the bottom left corner of the rectangle.
      y: The y-coordinate of the bottom left corner of the rectangle.
      width: The width of the rectangle.
      height: The height of the rectangle.
      **kwargs: Additional keyword arguments to be passed to `plt.fill_between`.

  Returns:
      None
  """

  # Calculate top right corner coordinates
  top_right_x = x + width
  top_right_y = y + height

  # Plot the rectangle using fill_between
  plt.fill_between(
      [x, top_right_x],
      [y, y],
      [top_right_y, top_right_y],
      **kwargs
  )

    
def plot_alignments(ax, read_df, height=0.1):
    """Plots read alignments based on a pandas DataFrame.
    
    Args:
      ax (matplotlib.axes._axes.Axes): The matplotlib Axes object where the plot will be drawn.
      read_df (pandas.DataFrame): DataFrame containing read information.
          Expected columns:
              - read_start (int): Start position of the read on the reference.
              - read_end (int): End position of the read on the reference.
              - is_mapped (bool): Whether the read is mapped.
              - mapping_quality (int): Mapping quality of the read.
              - chrom (str): Chromosome name (optional).
              - bin (int): Bin number within the chromosome (optional).
    
      height (float, optional): Height of the rectangles representing alignments. Defaults to 0.1.
    
    Returns:
      None (shows the plot)
    """

    colormap = {
        True : 'darkgreen',
        False : 'red',
    }

    # plot the alignments 
    for idx, row in read_df.iterrows():
        alignment_length = row['read_end'] - row['read_start']
        midpoint = (alignment_length / 2) + row['read_start']
        c = colormap[row['is_mapped']]

        if (row['mapping_quality'] > 0) & (row['mapping_quality'] < 40):
            c = 'goldenrod'
        if row['mapping_quality'] < 1:
            c = 'red'
        
        _plot_rectangle(ax, 
                    x=row['read_start'], 
                    y=0.05, 
                    height=height, 
                    width=alignment_length, 
                    color=c, 
                    zorder=2,
                    alpha=0.3)


        if row['is_mapped']:                
            annot = f" --- {row['chrom']} ({int(row['bin'])})"
            ax.annotate(annot, 
                         ha='center', 
                         va='bottom', 
                         rotation=90,
                         xy=(midpoint, 0.17))

    # Optional: Set plot limits and labels (adjust as needed)
    ax.set_ylim(0,  height + 0.25)  # adjust y-axis limits
    ax.set_xlabel('Read Position (bp)')
    ax.set_yticks([])
    sns.despine(left=True)
