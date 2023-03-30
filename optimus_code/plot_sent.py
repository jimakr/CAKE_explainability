import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib import cm, transforms


def plot_sentence_heatmap(sentences, scores, title="", width=10, height=0.2, verbose=0, max_characters=143):
    """ This function was originally published in the Innvestigate library: https://innvestigate.readthedocs.io/en/latest/ but modified to work for sentences.
    Args:
		sentences: The sentences
		scores: The scores per sentence
		title: The title of the plot
		width: The width of the plot
		height: The height of the plot
		verbose: If information about the plots is going to be presented
		max_characters: The max characters per row
    """
    fig = plt.figure(figsize=(width, height), dpi=200)

    ax = plt.gca()
    ax.set_title(title, loc='left')

    cmap = plt.cm.ScalarMappable(cmap=cm.bwr)
    cmap.set_clim(0, 1)

    canvas = ax.figure.canvas
    t = ax.transData

    # normalize scores to the followings:
    # - negative scores in [0, 0.5]
    # - positive scores in (0.5, 1]
    normalized_scores = 0.5 * scores / np.max(np.abs(scores)) + 0.5
    characters_per_line = 0
    loc_y = -0.2
    for i, token in enumerate(sentences):
        if token == '.':
            score = 0.5
        else:
            score = normalized_scores[i]
        *rgb, _ = cmap.to_rgba(score, bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)

        bbox_options = {'facecolor': color, 'pad': -0.1, 'linewidth': 0,
                        'boxstyle': 'round,pad=0.44,rounding_size=0.2'}
        text = ax.text(0.0, loc_y, token, bbox=bbox_options, transform=t)

        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        if characters_per_line + len(token) + 1 >= max_characters + 1:
            loc_y = loc_y - 1.8
            t = ax.transData
            characters_per_line = len(token) + 1
        else:
            t = transforms.offset_copy(
                text._transform, x=ex.width+15, units='dots')
            characters_per_line = characters_per_line + len(token) + 1

    ax.axis('off')