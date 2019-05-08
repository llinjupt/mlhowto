"""
Broadcast Visualization
-----------------------
Figure A.1

A visualization of NumPy array broadcasting. Note that the extra memory
indicated by the dotted boxes is never allocated, but it can be convenient
to think about the operations as if it is.
"""
# Author: Jake VanderPlas
# License: BSD
#   The figure produced by this code is published in the textbook
#   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
#   For more information, see http://astroML.github.com
#   To report a bug or issue, use the following forum:
#    https://groups.google.com/forum/#!forum/astroml-general
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
def setup_text_plots(fontsize=8, usetex=True):
    """
    This function adjusts matplotlib settings so that all figures in the
    textbook have a uniform format and look.
    """
    from distutils.version import LooseVersion
    matplotlib.rc('legend', fontsize=fontsize, handlelength=3)
    matplotlib.rc('axes', titlesize=fontsize)
    matplotlib.rc('axes', labelsize=fontsize)
    matplotlib.rc('xtick', labelsize=fontsize)
    matplotlib.rc('ytick', labelsize=fontsize)
    matplotlib.rc('text', usetex=usetex)
    matplotlib.rc('font', size=fontsize, family='serif',
                  style='normal', variant='normal',
                  stretch='normal', weight='normal')
    matplotlib.rc('patch', force_edgecolor=True)
    if LooseVersion(matplotlib.__version__) < LooseVersion("3.1"):
        matplotlib.rc('_internal', classic_mode=True)
    else:
        # New in mpl 3.1
        matplotlib.rc('scatter.edgecolors', 'b')
    matplotlib.rc('grid', linestyle=':')
    matplotlib.rc('errorbar', capsize=3)
    matplotlib.rc('image', cmap='viridis')
    matplotlib.rc('axes', xmargin=0)
    matplotlib.rc('axes', ymargin=0)
    matplotlib.rc('xtick', direction='in')
    matplotlib.rc('ytick', direction='in')
    matplotlib.rc('xtick', top=True)
    matplotlib.rc('ytick', right=True)
    
setup_text_plots(fontsize=8, usetex=True)

#------------------------------------------------------------
# Draw a figure and axis with no boundary
fig = plt.figure(figsize=(5, 3), facecolor='w')
ax = plt.axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
plt.style.use('ggplot')

# FIXME. 8 is the font size, how to get the text font size?
def font_height(fontsize=8):
    return fontsize / fig.dpi

def font_width(fontsize=8):
    return font_height(fontsize) / 2

def draw_cube(ax, xy, size, depth=0.4,
              edges=None, label=None, label_kwargs=None, **kwargs):
    """draw and label a cube.  edges is a list of numbers between
    1 and 12, specifying which of the 12 cube edges to draw"""
    if edges is None:
        edges = range(1, 13)

    x, y = xy
    
    # first plot background edges
    if 9 in edges:
        ax.plot([x + depth, x + depth + size],
                [y + depth + size, y + depth + size], **kwargs)
    if 10 in edges:
        ax.plot([x + depth + size, x + depth + size],
                [y + depth, y + depth + size], **kwargs)
    if 11 in edges:
        ax.plot([x + depth, x + depth + size],
                [y + depth, y + depth], **kwargs)
    if 12 in edges:
        ax.plot([x + depth, x + depth],
                [y + depth, y + depth + size], **kwargs)
    
    # second plot middile edges
    if 5 in edges:
        ax.plot([x, x + depth],
                [y + size, y + depth + size], **kwargs)
    if 6 in edges:
        ax.plot([x + size, x + size + depth],
                [y + size, y + depth + size], **kwargs)
    if 7 in edges:
        ax.plot([x + size, x + size + depth],
                [y, y + depth], **kwargs)
    if 8 in edges:
        ax.plot([x, x + depth],
                [y, y + depth], **kwargs)
    
    # last plot foreground edges 
    if 1 in edges: # top edge
        ax.plot([x, x + size],
                [y + size, y + size], **kwargs)
    if 2 in edges: # right 
        ax.plot([x + size, x + size],
                [y, y + size], **kwargs)
    if 3 in edges: # bottom
        ax.plot([x, x + size],
                [y, y], **kwargs)
    if 4 in edges: # left
        ax.plot([x, x],
                [y, y + size], **kwargs)

    if label:
        if label_kwargs is None:
            label_kwargs = {}
        
        ax.text(x + 0.5 * size, y + 0.5 * size - font_height() / 2, 
                label, ha='center', va='center', **label_kwargs)

solid = dict(c='black', ls='-', lw=1,      # solid border style and color
             label_kwargs=dict(color='k')) # text color 
dotted = dict(c='black', ls=':', lw=0.5,        # virtual border style and color
              label_kwargs=dict(color='gray'))  # text color
depth = 0.3

# xy is the start point with style (x,y)
def draw_vector(vector, xy, title="1D Vector", with_axis=True):
    if vector.ndim != 1:
        print("{} is not a vector".format(vector))
        return
    
    x,y = xy
    size = len(vector)
    
    # draw title at the center
    if len(title): 
        ax.text(x + size / 2, y + 1.5, title,
                size=12, ha='center', va='bottom')

    # draw axes
    if with_axis:
        starx = x - 0.5
        endx = x + size + 0.5
        axisy = y + 1.1
        ax.annotate("", xy=(endx, axisy), xytext=(starx, axisy),
                    arrowprops=dict(arrowstyle="simple", color='black'))
        ax.text(endx - 1, axisy, r'axis 0',
                size=10, ha='center', va='bottom')

    if size == 1:
        draw_cube(ax, (x, y), 1, depth, [1, 2, 3, 4], str(vector[0]), **solid)
    else:
        for i in range(size - 1):
            draw_cube(ax, (x + i, y), 1, depth, [1, 3, 4], str(vector[i]), **solid)
        draw_cube(ax, (x + i + 1, y), 1, depth, [1, 2, 3, 4], str(vector[i+1]), **solid)

def draw_vector_without_bottom(vector, xy):
    if vector.ndim != 1:
        print("{} is not a vector".format(vector))
        return
    
    x,y = xy
    size = len(vector)
    if size == 1:
        draw_cube(ax, (x, y), 1, depth, [1, 2, 4], str(vector[0]), **solid)
    else:
        for i in range(size - 1):
            draw_cube(ax, (x + i, y), 1, depth, [1, 4], str(vector[i]), **solid)
        draw_cube(ax, (x + i + 1, y), 1, depth, [1, 2, 4], str(vector[i+1]), **solid)

def draw_column(cols, xy, size=1):
    x,y = xy
    
    x += 0.5 * size
    x += 1
    y += 0.5 * size - font_height() / 2
    # FIXME. 8 is the font size, how to get the text font size?
    for i in range(cols):
        ax.text(x + i, y, 'col' + str(i), ha='center', va='center', color='k')

def draw_row(rows, xy, size=1):
    x,y = xy
    
    x += 0.5 * size
    y += 0.5 * size - 8 / fig.dpi / 2
    y -= 1
    # FIXME. 8 is the font size, how to get the text font size?
    for i in range(rows):
        ax.text(x, y - i, 'row' + str(i), ha='center', va='center', color='k')

def draw_matrix(matrix, xy, title="2D Matrix", with_row_col=True, with_axis=True):
    if matrix.ndim != 2:
        print("{} is not a matrix".format(matrix))
        return
    
    x, y = xy
    
    width = matrix.shape[1]
    heigh = matrix.shape[0]
    if with_row_col:
        width += 1
        heigh += 1
    
    # draw title at the center
    if len(title): 
        ax.text(x + width / 2, y + 1.5, title,
                size=12, ha='center', va='bottom')
    
    # draw axes
    if with_axis:
        starx = x - 0.5
        endx = x + width + 0.5
        axisy = y + 1
        ax.annotate("", xy=(endx, axisy), xytext=(starx, axisy),
                    arrowprops=dict(arrowstyle="simple", color='black'))
        ax.text(endx - 1, axisy, r'axis 1',
                size=10, ha='center', va='bottom')
        
        axisx = x - 0.5
        starty = y + 1
        endy = y - heigh + 0.5
        ax.annotate("", xy=(axisx, endy), xytext=(axisx, starty),
                    arrowprops=dict(arrowstyle="simple", color='black'))
        ax.text(axisx, endy - 0.4, r'axis 0',
                size=10, ha='center', va='bottom')
        
    if with_row_col:
        draw_row(matrix.shape[0], xy)
        draw_column(matrix.shape[1], xy)

        x += 1
        y -= 1

    rows = matrix.shape[0]
    if rows == 1:
        draw_vector(matrix[0], xy, title='', with_axis=False)
    else:
        for i in range(rows - 1):
            draw_vector_without_bottom(matrix[i], (x, y - i))
        
        print(matrix[i+1])
        draw_vector(matrix[i+1], (x, y - i - 1), title='', with_axis=False)
        
#------------------------------------------------------------
# Draw top operation: vector plus scalar
'''
draw_cube(ax, (1, 10), 1, depth, [1, 2, 3, 4, 5, 6, 9], '0', **solid)
draw_cube(ax, (2, 10), 1, depth, [1, 2, 3, 6, 9], '1', **solid)
draw_cube(ax, (3, 10), 1, depth, [1, 2, 3, 6, 7, 9, 10], '2', **solid)

draw_cube(ax, (6, 10), 1, depth, [1, 2, 3, 4, 5, 6, 7, 9, 10], '5', **solid)
draw_cube(ax, (7, 10), 1, depth, [1, 2, 3, 6, 7, 9, 10, 11], '5', **dotted)
draw_cube(ax, (8, 10), 1, depth, [1, 2, 3, 6, 7, 9, 10, 11], '5', **dotted)

draw_cube(ax, (12, 10), 1, depth, [1, 2, 3, 4, 5, 6, 9], '5', **solid)
draw_cube(ax, (13, 10), 1, depth, [1, 2, 3, 6, 9], '6', **solid)
draw_cube(ax, (14, 10), 1, depth, [1, 2, 3, 6, 7, 9, 10], '7', **solid)

ax.text(5, 10.5, '+', size=12, ha='center', va='center')
ax.text(10.5, 10.5, '=', size=12, ha='center', va='center')
'''

def draw_sum_axis0(a=None, xy=(4,9)):
    if a is None:
        a = np.arange(16).reshape(4,4)
    
    x, y = xy
    draw_matrix(a, (x,y), title=r'$np.sum(a, axis=0)$')
    sum = np.sum(a, axis=0)
    
    draw_vector(sum, (x + 1, y - a.shape[0] - 1.5), title='', with_axis=False)
    
    columns = a.shape[1]

    # draw arrows
    for i in range(columns):        
        ax.annotate("", xy=(x + 1.5 + i, y - a.shape[0] - 0.5), xytext=(x + 1.5 + i, y), 
                        arrowprops=dict(arrowstyle="simple", alpha=0.3, color='red'))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(2.5, 12)
    plt.show()

def draw_sum_axis1(a=None, xy=(4,9)):
    if a is None:
        a = np.arange(16).reshape(4,4)
    
    x, y = xy
    draw_matrix(a, (x, y), title=r'$np.sum(a, axis=1)$')
    sum = np.sum(a, axis=1)
    
    draw_matrix(sum.reshape(a.shape[0],1), (x + a.shape[1] + 1.5, y - 1), title='', with_row_col=False, with_axis=False)

    rows = a.shape[0]

    # draw arrows
    for i in range(rows):        
        ax.annotate("", xy=(x + a.shape[1] + 1.5, y - 0.5 - i), xytext=(x + 1, y - 0.5 - i), 
                    arrowprops=dict(arrowstyle="simple", alpha=0.3, color='red'))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(2.5, 12)
    plt.show()

#draw_sum_axis1()
A = np.array([1, 2])
B = np.array([3, 4])
print(np.stack((A, B), axis=1))
