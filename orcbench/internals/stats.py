import random
import math
import numpy as np

from .orcglobals import CAPTURE_WINDOW

MAX_ROUND = 7


def inverse_poisson(start_min, mu):
    if mu == 0:
        return np.asarray([])
    interarrivals = np.random.exponential(1.0 / mu, mu)
    interarrivals = np.around(interarrivals, decimals=MAX_ROUND)
    interarrivals = np.cumsum(interarrivals)
    interarrivals = interarrivals + start_min
    return interarrivals


def resample_as_poisson(t, begin, end):
    my_values = []
    if sum(t[begin:end]) == 0:
        return np.asarray([end - begin])

    for min, val in enumerate(t[begin:end]):
        l = inverse_poisson(min, val)
        my_values.extend(l)

    my_values = np.asarray(my_values)
    my_values.sort()

    if begin == 0:
        prev = 0
    prev = 0
    i = begin
    while (i > 0) and (t[i - 1]) == 0:
        i -= 1
        prev -= 1

    final_values = [my_values[0] - prev]
    # Now go through this new interarrival timeline and calculate IA rate
    prev = my_values[0]
    for i in range(1, len(my_values)):
        next_value = my_values[i]

        # Take what the last IA value was so we double its weight
        if next_value == prev:
            next_value = prev + (1.0 / math.pow(10, -1 * MAX_ROUND))
        rounded = next_value - prev

        final_values.append(rounded)
        prev = next_value

    return np.asarray(final_values)


def calc_ia_time(func):
    t = func["timeline"]
    # Sunday + 12Hrs
    chunks = np.arange(1440 + 720, (5 * 1440), 1440)
    final_list = []
    for x in chunks:
        resampled = resample_as_poisson(t, x, x + CAPTURE_WINDOW)
        final_list.extend(resampled)

    func["ia"] = np.asarray(final_list, dtype="float32")


def calc_invocs_list(trace, run_for, unit=1):
    if run_for % unit == 0:
        invocs_list = [0] * int((run_for) / unit)
    else:
        invocs_list = [0] * int((run_for + unit) / unit)
    cur = 0
    invocs = 0
    ii = 0
    for t in trace:
        v = t[0]
        f = 1
        if int(v) >= run_for:
            break
        if int(v) > cur + unit:
            if ii == 0:
                i = 0
            else:
                i = int(trace[ii - 1][0] / unit)
            invocs_list[i] = invocs
            invocs = 0
            cur += unit
        if int(v) == cur + unit:
            if ii == 0:
                i = 0
            else:
                i = int(trace[ii - 1][0] / unit)
            invocs_list[i] = invocs
            invocs = f
            cur += unit
        else:
            invocs += f
        ii += 1

    ii -= 1
    if len(trace):
        invocs_list[int(trace[ii][0] / unit)] = invocs
    return invocs_list


def calculate_ia_freq(t, begin, end):
    return resample_as_poisson(t, begin, end)
