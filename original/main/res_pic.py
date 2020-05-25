import matplotlib.pyplot as plt


def draw(image_path, xlist, ylist, xlabel, ylabel):
    """
    This file generates the graph of the recommendation results.

    :param image_path:
    :param xlist:
    :param ylist:
    :param xlabel:
    :param ylabel:
    :return:
    """
    lines = []
    titles = []
    colorList = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('recsys')

    line1, = plt.plot(xlist, ylist)
    plt.setp(line1, color=colorList[0], linewidth=2.0)

    titles.append(ylabel)
    lines.append(line1)

    plt.legend(lines, titles)
    plt.savefig(image_path, dpi=120)
    plt.close()
