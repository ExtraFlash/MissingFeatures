import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


def set_plot_style(
        title_size=16,
        axis_label_size=14,
        tick_label_size=12,
        legend_size=10,
        font_family='Arial',
        font_weight='normal',
        dpi=300,
        grid_alpha=0.3,
        grid_linestyle='--',
        figure_style='seaborn-whitegrid'
):
    """
    Set global plot styling with extensive customization options.

    Parameters:
    -----------
    title_size : int, optional (default=16)
        Font size for plot titles
    axis_label_size : int, optional (default=14)
        Font size for x and y axis labels
    tick_label_size : int, optional (default=12)
        Font size for tick labels
    legend_size : int, optional (default=10)
        Font size for legend text
    font_family : str, optional (default='Arial')
        Font family for all text elements
    font_weight : str, optional (default='normal')
        Font weight ('light', 'normal', 'regular', 'book', 'medium', 'bold')
    dpi : int, optional (default=300)
        Dots per inch for high-quality rendering
    grid_alpha : float, optional (default=0.3)
        Transparency of grid lines
    grid_linestyle : str, optional (default='--')
        Line style for grid
    figure_style : str, optional (default='seaborn-whitegrid')
        Matplotlib/Seaborn style to use

    Returns:
    --------
    None
    """
    # Set matplotlib style
    # plt.style.use(figure_style)

    # Configure font
    font_prop = font_manager.FontProperties(
        family=font_family,
        weight=font_weight
    )
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.weight'] = font_weight

    # Title settings
    plt.rcParams['figure.titlesize'] = title_size
    plt.rcParams['axes.titlesize'] = title_size
    # plt.rcParams['axes.titleweight'] = 'bold'

    # Label settings
    plt.rcParams['axes.labelsize'] = axis_label_size
    # plt.rcParams['axes.labelweight'] = 'bold'

    # Tick settings
    plt.rcParams['xtick.labelsize'] = tick_label_size
    plt.rcParams['ytick.labelsize'] = tick_label_size

    # Legend settings
    plt.rcParams['legend.fontsize'] = legend_size

    # Grid settings
    plt.rcParams['grid.alpha'] = grid_alpha
    plt.rcParams['grid.linestyle'] = grid_linestyle

    # Other quality settings
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['savefig.dpi'] = dpi
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['path.simplify'] = True
    plt.rcParams['path.simplify_threshold'] = 1.0

    # Return the configured font properties if needed
    return font_prop