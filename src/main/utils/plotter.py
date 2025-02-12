import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_on_one(title, xtitle, ytitle, data, show=True, logscale_axis=0, show_legend=True):
    """
    Plot multiple plots on one figure
    :param title:
    :param xtitle:
    :param ytitle:
    :param data:
        [x_array, y_array, label, style ...]
    :return:
    """
    # fig, ax = plt.subplots()

    handles = []

    for i in range(0, int(len(data)), 4):
        style = data[i + 3]
        style = '-' if style == '' else style

        if logscale_axis == 2:
            p = plt.semilogy(data[i], data[i + 1], style, label=data[i + 2])[0]
        elif logscale_axis == 1:
            p = plt.semilogx(data[i], data[i + 1], style, label=data[i + 2])[0]
        else:
            p = plt.plot(data[i], data[i + 1], style, label=data[i + 2])[0]

        handles.append(p)

    if show_legend:
        plt.legend(handles=handles)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    if show:
        plt.show()


def plot_silhouette_scores(X, cluster_labels, sample_silhouette_values, silhouette_avg, n_clusters, colors=None):
    fig, ax = plt.subplots()
    ax.set_xlim([-1, 1])
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ccolors = colors
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        if colors is not None:
            ith_colors = ccolors[cluster_labels == i] / 255
            score_color = list(np.c_[ith_cluster_silhouette_values, ith_colors])
            score_color.sort(key=lambda x: x[0])
            score_color = np.array(score_color)
            ccolors = score_color[:, 1:]
            ith_cluster_silhouette_values = score_color[:, 0]
        else:
            ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        if colors is None:
            ccolors = np.array(get_color(i, letters=False)) / 255
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=ccolors, edgecolor=ccolors, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])
    plt.show()


def plot_points(title='plot', xtitle='x', ytitle='y', points=None):
    """

    :param title:
    :param xtitle:
    :param ytitle:
    :param points:
        [
            [x, y, <fill color: (r, g, b)>, <edge color: (r, g, b)>]
        ]
    :return:
    """
    for p in points:
        x, y, fc, ec = p
        fc = np.array(fc) / 255
        ec = np.array(ec) / 255

        plt.plot(x, y, color=fc, marker='o', fillstyle='full',
                 markeredgecolor=ec,
                 markeredgewidth=0.3)

    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.show()


def plot_on_one_errorbars(title, xtitle, ytitle, data, only_ints_on_x=False, save_it=False):
    """
    Plot multiple plots on one figure with error bars
    :param title:
    :param xtitle:
    :param ytitle:
    :param data:
        [x_array, y_array, label, style, yerr, ...]
    :param only_ints_on_x:
    :param save_it:
    :return:
    """

    fig, ax = plt.subplots()
    if only_ints_on_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    handles = []

    for i in range(0, int(len(data)), 5):
        style = data[i + 3]
        style = '-' if style == '' else style
        yerr = data[i + 4]

        x, y = data[i], data[i + 1]
        p = plt.errorbar(x, y, fmt=style, yerr=yerr, label=data[i + 2])

        handles.append(p)

    plt.legend(handles=handles)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    if save_it:
        plt.savefig(title + '.png')
    else:
        plt.show()


def draw_plot_point_label_set2(title='plot', xtitle='x', ytitle='y', set=None):
    """
    Plots sets on one chart.
    Every label has different style.
    :param title:
    :param xtitle:
    :param ytitle:
    :param set:
        [[x1, y1, label1], [x2, y2, label1], ..., [x10, y10, label2], ...]
    :return:
    """
    if set is None:
        set = list()
    data = []
    cords = {}

    for index, vect in enumerate(set):
        x, y, label = vect
        label_key = str(int(label))
        if label_key in cords:
            cords[label_key]["x"].append(x)
            cords[label_key]["y"].append(y)
        else:
            cords[label_key] = {
                "x": [x],
                "y": [y],
                "style": get_style(int(label))
            }

    for k, v in cords.items():
        data.extend([v["x"], v["y"], str(k), v["style"]])

    plot_on_one(title, xtitle, ytitle, data, show_legend=False)


def merge_labels_2d(X, labels):
    """
    Can be used for transforming data for draw_plot_point_label_set2 and draw_plot_point_label_set
    :param X:
    :param labels:
    :return:
    """
    return np.c_[X[:, 0], X[:, 1], labels]


def draw_plot_point_label_set(title='plot', xtitle='x', ytitle='y', base=None, *sets):
    """
    Plots sets on one chart.
    Every set has one specific marker, every label has own color.
    :param title:
    :param xtitle:
    :param ytitle:
    :param base:
    :param sets:
        [
            [[x1, y1, label1], [x2, y2, label1], ..., [x10, y10, label2], ...], // one set
            ...
        ]
    :return:
    """
    data = []
    cords = {}

    for s_i, sett in enumerate(sets):
        marker = get_marker(s_i)
        for index, vect in enumerate(sett):
            x, y, label = vect
            label_key = str(int(label)) + marker
            if label_key in cords:
                cords[label_key]["x"].append(x)
                cords[label_key]["y"].append(y)
            else:
                cords[label_key] = {
                    "x": [x],
                    "y": [y],
                    "style": str(get_color(label)) + marker
                }

    for k, v in cords.items():
        data.extend([v["x"], v["y"], str(k), v["style"]])

    if base is None:
        plot_on_one(title, xtitle, ytitle, data)
    else:
        plot_on_one_with_vect(title, xtitle, ytitle, data, base)


glob_styles = []
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
colors_rgb = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (100, 100, 100), (100, 100, 0), (0, 100, 100)]
markers = ['.', 'o', '+', '2', 'x', 'v', '<', '>', '1']

for m in markers:
    for c in colors:
        glob_styles.append(c + m)


def get_color(x, letters=True):
    if letters:
        return colors[int(x) % len(colors)]
    else:
        return colors_rgb[int(x) % len(colors_rgb)]


def get_marker(x):
    return markers[int(x) % len(markers)]


def get_style(x):
    return glob_styles[int(x) % len(glob_styles)]


def newline(p1, p2):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], marker='o')


def get_rand_rgb(n=3):
    color = []
    for i in range(n):
        color.append(np.random.randint(0, 256))
    return color


def plot_on_one_with_vect(title, xtitle, ytitle, data, base):
    plot_on_one(title, xtitle, ytitle, data, show=False)
    newline(base[0][0], base[0][1])
    newline(base[1][0], base[1][1])
    plt.show()


def plot_gallery(title, images, n_col=3, n_row=2, image_shape=(64, 64)):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)

    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
    plt.show()


def histogram_bars(data_sequence, labels, ylabel='y', xlabel='x', title='histogram'):
    x = np.arange(len(data_sequence))
    plt.bar(x, height=data_sequence)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


def histogram_bars_grouped(seqences, xlabels, bar_labels, ylabel='y', xlabel='x', title='histogram',
                           bar_width=0.2, show_bar_values=True, figsize=None, save_fig=False,
                           legend_loc=0, show_legend=True):
    """

    :param seqences: sequences of data - they will be grouped by column
    [
        [1, 2, 3, 4],
        [2, 3, 4, 5]
    ]
    :param xlabels: labels on x axis
    :param bar_labels: labels for bars
    :param ylabel:
    :param xlabel: can be None
    :param title:
    :param bar_width:
    :param tight_layout:
    :return:
    """
    x = np.arange(len(xlabels))

    fig, ax = plt.subplots()
    rects = []

    seq_len = len(seqences)
    for i in range(seq_len):
        pos = i * bar_width + bar_width * (1 - seq_len) / 2
        rect = ax.bar(x + pos, seqences[i], bar_width, label=bar_labels[i])
        rects.append(rect)

    ax.set_ylabel(ylabel)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    if show_legend:
        ax.legend(loc=legend_loc)

    fig.tight_layout()
    if figsize is not None:
        fig.set_size_inches(figsize)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    if show_bar_values:
        for rect in rects:
            autolabel(rect)

    if save_fig:
        plt.savefig(title + '.png')
    else:
        plt.show()
