import six
from six.moves import xrange, zip
from matplotlib.colors import colorConverter
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Rectangle


# def candle_(ax, opens, closes, highs, lows, width=4, colorup='k', colordown='r', alpha=0.75):
#     candel(ax, opens, closes, highs, lows, width=4, colorup=colorup, colordown=colordown,
#         alpha=alpha)


def candle(ax, opens, highs, lows, closes, width=4, colorup='k', colordown='r', alpha=0.75):
    delta = width / 2.
    barVerts = [((i - delta, open), (i - delta, close), (i + delta, close), (i + delta, open)) for
                i, open, close in zip(xrange(len(opens)), opens, closes) if
                open != -1 and close != -1]
    rangeSegments = [((i, low), (i, high)) for i, low, high in zip(xrange(len(lows)), lows, highs)
                     if low != -1]
    r, g, b = colorConverter.to_rgb(colorup)
    colorup = r, g, b, alpha
    r, g, b, = colorConverter.to_rgb(colordown)
    colordown = r, g, b, alpha
    colord = {True: colorup, False: colordown, }
    colors = [colord[open < close] for open, close in zip(opens, closes) if
              open != -1 and close != -1]
    assert (len(barVerts) == len(rangeSegments))
    useAA = 0
    lw = 0.5
    rangeCollection = LineCollection(rangeSegments, colors=((0, 0, 0, 1),), linewidths=lw,
        antialiaseds=useAA, )
    barCollection = PolyCollection(barVerts, facecolors=colors, edgecolors=((0, 0, 0, 1),),
        antialiaseds=useAA, linewidths=lw, )
    minx, maxx = 0, len(rangeSegments)
    miny = min([low for low in lows if low != -1])
    maxy = max([high for high in highs if high != -1])
    corners = (minx, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()
    ax.add_collection(barCollection)
    ax.add_collection(rangeCollection)
    return rangeCollection, barCollection
