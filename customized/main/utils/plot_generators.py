import time

import numpy as np

from customized.main.utils.plotter import plot_on_one_errorbars, get_color, histogram_bars_grouped


def gen_plots_with_errbars(plots_desc, compared, range_val, func, repeat=3, save_figs=False, verbose=False):
    """
    Generate series of figures with multiple plots on one piece with error bars.

    :param plots_desc: description of a plots
        Example:
        [
            {'key': 'time', 'name': 'Time vs k neighours', 'xlabel': 'k', 'ylabel': 't [s]'},
            {'key': 'prec', 'name': 'Prec vs k neighours', 'xlabel': 'k', 'ylabel': 't [s]'}
        ]
    :param compared: array of objects which we compare across others, has to contain name
        Example:
        [
            { 'model': Foo, 'name': 'Foo'},
            { 'model': Bar, 'name': 'Bar'}
        ]
    :param range_val: range of values which are our x values in a chart
    :param func: function which describe our logic. To the function 3 args are passed:
        sink - which is a dict with specified keys in plot_desc and corresponding list for a values
        compared - object from compared array
        j - which is a a value from specified range_val

        Example:
        def func(sink, model_desc, k):
            model_class = model_desc['model']
            model = model_class(a=k)

            start = time.time()
            model.do()
            end_time = time.time() - start
            sink['time'].append(end_time)

            r = model.get()
            sink['prec'].append(r)
    :param repeat: specyfy how many times repeat our function
    :param save_figs: True is you want to save plots to the filesystem
    :return: void
    """

    def _init_y_avg():
        y_val = dict()
        for desc in plots_desc:
            y_val[desc['key']] = {'y': list(), 'err': list()}

        return y_val

    def _init_list():
        y_val = dict()
        for desc in plots_desc:
            y_val[desc['key']] = list()

        return y_val

    def _calc_avg(y_val, y_avg):
        for desc in plots_desc:
            key = desc['key']
            y_avg[key]['y'].append(np.mean(y_val[key]))
            y_avg[key]['err'].append(np.std(y_val[key]))

    def _add_avg_to_data(index, data_lists, y_avg, case_name):
        for desc in plots_desc:
            key = desc['key']
            data_lists[key].extend([range_val, y_avg[key]['y'], case_name, get_color(index) + '-', y_avg[key]['err']])

    def _plot_data_lists(data_lists):
        for desc in plots_desc:
            key = desc['key']
            plot_on_one_errorbars(desc['name'], desc['xlabel'], desc['ylabel'],
                                  data_lists[key], only_ints_on_x=True, save_it=save_figs)

    def _print_after_process(j_value, y_avg):
        print('>>> results for value {}: '.format(j_value))
        for desc in plots_desc:
            key = desc['key']
            print('  >>> {} - y: {}, err: {}'.format(key, y_avg[key]['y'][-1], y_avg[key]['err'][-1]))

    data_lists = _init_list()
    for i, compared_obj in enumerate(compared):
        case_name = compared_obj['name']
        y_avg = _init_y_avg()
        if verbose:
            print('({}) started calculation for {} case'.format(i, case_name))
        for j in range_val:
            y_val = _init_list()

            for _ in range(repeat):
                func(y_val, compared_obj, j)
            _calc_avg(y_val, y_avg)

            if verbose:
                _print_after_process(j, y_avg)
        _add_avg_to_data(i, data_lists, y_avg, case_name)

    _plot_data_lists(data_lists)


def gen_grouped_histogram_bars(plots_desc, compared, range_val, func, repeat=3, save_figs=False, verbose=False,
                               bar_width=0.2, show_bar_values=True, figsize=None, legend_loc=0, show_legend=True):
    def _init_data():
        d = dict()
        for desc in plots_desc:
            d[desc['key']] = list()
            for i in range(len(range_val)):
                d[desc['key']].append([])

        return d

    def _init_list():
        y_val = dict()
        for desc in plots_desc:
            y_val[desc['key']] = list()

        return y_val

    def _calc_avg(y_val, data_lists, val_ind):
        for desc in plots_desc:
            key = desc['key']
            data_lists[key][val_ind].append(np.mean(y_val[key]))

    def _plot_data_lists(data_lists):
        xlabels = []
        for c in compared:
            xlabels.append(c['name'])
        for desc in plots_desc:
            key = desc['key']
            histogram_bars_grouped(data_lists[key], xlabels, bar_labels=range_val,
                                   ylabel=desc['ylabel'], xlabel=desc['xlabel'], title=desc['name'],
                                   bar_width=bar_width, show_bar_values=show_bar_values, figsize=figsize,
                                   save_fig=save_figs, legend_loc=legend_loc, show_legend=show_legend)

    def _print_after_process(j_value, j_idx, data_lists):
        print('>>> results for value {}: '.format(j_value))
        for desc in plots_desc:
            key = desc['key']
            print('  >>> {} - y: {}'.format(key, data_lists[key][j_idx][-1]))

    data_lists = _init_data()
    for i, compared_obj in enumerate(compared):
        case_name = compared_obj['name']
        if verbose:
            print('({}) started calculation for {} case'.format(i, case_name))

        for j, val in enumerate(range_val):

            y_val = _init_list()
            for _ in range(repeat):
                func(y_val, compared_obj, val)
            _calc_avg(y_val, data_lists, j)

            if verbose:
                _print_after_process(val, j, data_lists)

    _plot_data_lists(data_lists)


def measure_time(func, sink_list):
    start = time.time()
    r = func()
    end_time = time.time() - start
    sink_list.append(end_time)

    return r
