import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm


def plot_on_one(title, xtitle, ytitle, data, show=True, logscale_axis=0, show_legend=True):
    """
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


def plot_points(title='plot', xtitle='x', ytitle='y', points=[]):
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
        plt.savefig(title+'.png')
    else:
        plt.show()


def draw_plot_point_label_set2(title='plot', xtitle='x', ytitle='y', set=[]):
    """
    Plots sets one one chart.
    Every label has different style.
    :param title:
    :param xtitle:
    :param ytitle:
    :param set:
        [[x1, y1, label1], [x2, y2, label1], ..., [x10, y10, label2], ...], // one set
    :return:
    """
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
