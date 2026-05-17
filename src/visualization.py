import matplotlib.pyplot as plt

def plot_metric_surface(df, x_col, y_col, split_col, line_col, std_col=None, title=None,
                        x_label=None, y_label=None):
    """
    Universal plotting function for ML experiment grids.

    Parameters:
    - df: pandas DataFrame
    - x_col: variable on X axis (e.g. n_features)
    - y_col: metric to plot (e.g. score)
    - split_col: creates subplots (e.g. C)
    - line_col: separates lines (e.g. threshold)
    - std_col: optional std deviation column
    """


    split_values = sorted(df[split_col].unique())
    line_values = sorted(df[line_col].unique())

    fig, axes = plt.subplots(1, len(split_values), figsize=(6 * len(split_values), 5), sharey=True)

    if len(split_values) == 1:
        axes = [axes]

    for i, split_val in enumerate(split_values):

        ax = axes[i]
        df_split = df[df[split_col] == split_val]

        for line_val in line_values:

            df_line = df_split[df_split[line_col] == line_val].sort_values(x_col)

            if df_line.empty:
                continue

            x = df_line[x_col]
            y = df_line[y_col]

            ax.plot(x, y, label=f"{line_col}={line_val}")

            if std_col is not None:
                yerr = df_line[std_col]
                ax.fill_between(
                    x,
                    y - yerr,
                    y + yerr,
                    alpha=0.15
                )

        ax.set_title(f"{split_col} = {split_val}")
        ax.set_xlabel(x_label or x_col)
        ax.grid(True)

    axes[0].set_ylabel(y_label or y_col)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()