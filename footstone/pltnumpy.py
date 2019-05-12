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
# 
# Author: Red Liu (lli_njupt@163.com)
#   Add more useful functions to draw vector and matrix
# 

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
fig = plt.figure(figsize=(5.2, 3), facecolor='w')

ax = plt.axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
# dray first dimension
fig_xsize, fig_ysize = fig.get_size_inches()
ax.set_xlim(0, fig_xsize * 3)
ax.set_ylim(0, fig_ysize * 3)
plt.style.use('ggplot')

# FIXME. 8 is the font size, how to get the text font size?
def font_height(fontsize=8):
    return fontsize / fig.dpi

def font_width(fontsize=8):
    return font_height(fontsize) / 2

def draw_cube(ax, xy, size, depth=0.3,
              edges=None, label=None, label_kwargs=None, **kwargs):
    """draw and label a cube.  edges is a list of numbers between
    1 and 12, specifying which of the 12 cube edges to draw"""
    if edges is None:
        edges = range(1, 13)

    x, y = xy
    y -= size # set left/up corner as the first (0,0) for one cube
    
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

# xy is the start point with style (x,y)
def draw_vector(vector, xy, title="1D Vector", with_axis=True, color='gray'):
    if vector.ndim != 1:
        print("{} is not a vector".format(vector))
        return
    
    x,y = xy
    size = len(vector)
    
    # draw title at the center
    if len(title):
        axisy = y + 0.1
        if with_axis:
            axisy = y + 1
            
        ax.text(x + size / 2, axisy, title,
                size=12, ha='center', va='bottom')

    # draw axes
    if with_axis:
        starx = x - 0.5
        endx = x + size + 0.5
        axisy = y + 0.3
        ax.annotate("", xy=(endx, axisy), xytext=(starx, axisy),
                    arrowprops=dict(arrowstyle="simple", color='black'))
        ax.text(endx - 1, axisy, r'axis 0',
                size=10, ha='center', va='bottom')
    
    if size == 1:
        draw_cube(ax, (x, y), 1, depth, [1, 2, 3, 4], str(vector[0]), **solid)
        draw_square_mask(ax, (x, y), color=color)
    else:
        for i in range(size - 1):
            draw_cube(ax, (x + i, y), 1, depth, [1, 3, 4], str(vector[i]), **solid)
            draw_square_mask(ax, (x + i, y), color=color)
            
        draw_cube(ax, (x + i + 1, y), 1, depth, [1, 2, 3, 4], str(vector[i+1]), **solid)
        draw_square_mask(ax, (x + i + 1, y), color=color)

def draw_indices_vector(vector, xy, color='gray'):
    if vector.ndim != 1:
        print("{} is not a vector".format(vector))
        return

    dotted = dict(c='k', ls='--', lw=0.2,        # virtual border style and color
              label_kwargs=dict(color='k'))  # text color
    
    x,y = xy
    size = len(vector)

    if size == 1:
        draw_cube(ax, (x, y), 1, depth, [1, 2, 3, 4], str(vector[0]), **dotted)
        draw_square_mask(ax, (x, y), color=color)
    else:
        for i in range(size - 1):
            draw_cube(ax, (x + i, y), 1, depth, [1, 3, 4], str(vector[i]), **dotted)
            draw_square_mask(ax, (x + i, y), color=color)
            
    draw_cube(ax, (x + i + 1, y), 1, depth, [1, 2, 3, 4], str(vector[i+1]), **dotted)
    draw_square_mask(ax, (x + i + 1, y), color=color)

def draw_vector_head(xy, color='gray'):
    x,y = xy
    x -= 2
    draw_cube(ax, (x, y), 1, depth, [1, 2, 3], '', **solid)
    draw_cube(ax, (x + 1, y), 1, depth, [1, 3], '', **solid)

def draw_vector_tail(xy, color='gray'):
    x,y = xy
    draw_cube(ax, (x, y), 1, depth, [1, 2, 3], '', **solid)
    draw_cube(ax, (x + 1, y), 1, depth, [1, 3], '', **solid)

def draw_vertical_vector(vector, xy, title="1D Vector", color='gray'):
    if vector.ndim != 1:
        print("{} is not a vector".format(vector))
        return
    
    x,y = xy
    size = len(vector)
    
    # draw title at the center
    if len(title): 
        ax.text(x + 1 / 2, y + 1, title,
                size=12, ha='center', va='bottom')

    if size == 1:
        draw_cube(ax, (x, y), 1, depth, [1, 2, 3, 4], str(vector[0]), **solid)
        draw_square_mask(ax, (x, y), color=color)
    else:
        for i in range(size - 1):
            draw_cube(ax, (x, y - i), 1, depth, [1, 2, 4], str(vector[i]), **solid)
            draw_square_mask(ax, (x, y - i), color=color)
            
        draw_cube(ax, (x, y - i - 1), 1, depth, [1, 2, 3, 4], str(vector[i+1]), **solid)
        draw_square_mask(ax, (x, y - i - 1), color=color)

from matplotlib.patches import Polygon,Rectangle

def create_stype(color='gray', alpha=0.5):
    if color == None:
        return dict(edgecolor='k', lw=1, fill=False)

    return dict(edgecolor=None, lw=0, facecolor=color, alpha=alpha)

def draw_square_mask(ax, xy, size=1, color='gray', alpha=0.1):
    if color == 'gray' or color is None: #don't fill gray just let it be background
        return
    
    '''xy is the left-top corner'''
    style = create_stype(color, alpha=alpha)
    
    rect = Rectangle(xy, width=size, height=-size, **style)
    ax.add_patch(rect)

# top diamond and square
def draw_top_mask(ax, xy, depth=0.3, size=1, color='gray', alpha=0.2):
    # draw top diamond
    x, y = xy
    points = [(x,y), (x+depth, y+depth), (x+size+depth, y+depth), (x+size, y)]
    style = create_stype(color=color, alpha=alpha)
    top = Polygon(points, closed=True, **style)
    ax.add_patch(top)

def draw_top_nmask(ax, xy, columns=1, color='gray'):
    x, y = xy
    for i in range(columns):
        draw_top_mask(ax, (x + i, y), color=color)

# dir mean direction 'h' or 'v'
def draw_square_nmask(ax, xy, cubes=1, color='gray', dir='h'):
    x, y = xy
    
    if dir == 'h':
        for i in range(cubes):
            draw_square_mask(ax, (x + i, y), color=color)
    else:
        for i in range(cubes):
            draw_square_mask(ax, (x, y - i), color=color)

def draw_square_nmask_column(ax, xy, rows=1, color='gray'):
    x, y = xy
    
    draw_top_nmask(ax, xy, 1, color=color)
    draw_square_nmask(ax, xy, cubes=rows, color=color, dir='v')

def draw_right_nmask(ax, xy, rows=1, color='gray'):
    x, y = xy
    
    for i in range(rows):
        draw_right_mask(ax, (x, y - i), color=color)

def draw_square_nmask_with_right(ax, xy, columns=1, color='gray'):
    x, y = xy
    for i in range(columns):
        draw_square_mask(ax, (x + i, y), color=color)
    draw_right_mask(ax, (x+i, y), color=color)

def draw_square_nmask_with_top(ax, xy, columns=1, color='gray'):
    x, y = xy

    draw_top_nmask(ax, xy, columns=columns, color=color)
    for i in range(columns):
        draw_square_mask(ax, (x + i, y), color=color)
    draw_right_mask(ax, (x+i, y), color=color)

# right diamond and square
def draw_right_mask(ax, xy, depth=0.3, size=1, color='gray', alpha=0.3):
    # draw right diamond
    x, y = xy
    
    points = [(x+size,y), (x+size+depth, y+depth), (x+size+depth, y-size+depth), 
              (x+size, y-size)]
    style = create_stype(color=color, alpha=alpha)
    top = Polygon(points, closed=True, **style)
    ax.add_patch(top)

def draw_top(vector, xy, color='gray'):
    if vector.ndim != 1:
        print("{} is not a vector".format(vector))
        return
    
    x,y = xy
    size = len(vector)
    draw_cube(ax, (x, y), 1, depth, [5, 6, 9], str(vector[0]), **solid)
    
    if size > 1:
        for i in range(1,size):
            draw_cube(ax, (x + i, y), 1, depth, [6, 9], str(vector[i]), **solid)

    # fill top surface color
    if color is not None:
        for i in range(size):
            draw_top_mask(ax, (x + i, y), color=color, alpha=0.2)

def draw_right(xy, size, color='gray'):
    x,y = xy
    
    for i in range(size):
        draw_cube(ax, (x, y - i), 1, depth, [7, 10], **solid)
        
        if color is not None:
            draw_right_mask(ax, (x, y-i), color=color, alpha=0.4)

# draw a cloumn without bottom lines
def draw_vector_no_bottom(vector, xy, color='gray'):
    if vector.ndim != 1:
        print("{} is not a vector".format(vector))
        return
    
    x,y = xy
    size = len(vector)
    if size == 1:
        draw_cube(ax, (x, y), 1, depth, [1, 2, 4], str(vector[0]), **solid)
        draw_square_mask(ax, (x, y), color=color)
    else:
        for i in range(size - 1):
            draw_cube(ax, (x + i, y), 1, depth, [1, 4], str(vector[i]), **solid)
            draw_square_mask(ax, (x + i, y), color=color)
        draw_cube(ax, (x + i + 1, y), 1, depth, [1, 2, 4], str(vector[i+1]), **solid)
        draw_square_mask(ax, (x + i + 1, y), color=color)

# draw column comment : col0 col1 ...
def draw_column(cols, xy, size=1):
    x,y = xy
    
    x += 0.5 * size
    x += 1
    y += 0.5 * size - font_height() / 2
    
    for i in range(cols):
        ax.text(x + i, y, 'col' + str(i), ha='center', va='center', color='k')

# draw row comment : row0 row1 ...
def draw_row(rows, xy, size=1):
    x,y = xy
    
    x += 0.5 * size
    y += 0.5 * size - font_height() / 2
    y -= 1
    
    for i in range(rows):
        ax.text(x, y - i, 'row' + str(i), ha='center', va='center', color='k')

def draw_axes(array, xy, with_row_col=True):
    x, y = xy
    
    if array.ndim == 1: # vector just draw axis 0
        size = len(array)
        endx = x + size + 0.5
        ax.annotate("", xy=(endx, y), xytext=xy,
                    arrowprops=dict(arrowstyle="simple", color='black'))
        ax.text(endx - 1, y, r'axis 1',
                size=10, ha='center', va='bottom')
        return
    
    # handle matrix
    heigh = array.shape[0]
    width = array.shape[1]

    if with_row_col:
        width += 1
        heigh += 1
        x -= 1
        y += 1
    
    starx = x
    endx = x + width + 0.5
    axisy = y
    ax.annotate("", xy=(endx + depth, axisy), xytext=(starx, axisy),
                arrowprops=dict(arrowstyle="simple", color='black'))
    ax.text(endx - 1, axisy, r'axis 1',
            size=10, ha='center', va='bottom')

    axisx = x
    starty = y
    endy = y - heigh - 0.5
    ax.annotate("", xy=(axisx, endy), xytext=(axisx, starty),
                arrowprops=dict(arrowstyle="simple", color='black'))
    ax.text(axisx, endy - 0.4, r'axis 0',
            size=10, ha='center', va='bottom')

def draw_matrix(matrix, xy, title="2D Matrix", with_row_col=True, with_axis=True, color='gray'):
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
        if with_row_col:
            startx = x - 1 + (width  + 1)/ 2
        else:
            startx = x + width / 2

        axisy = y + depth + 0.1
        if with_axis:
            axisy = y + 1.5
    
        ax.text(startx, axisy, title,
                size=12, ha='center', va='bottom')
    
    # draw axes
    if with_axis:
        if with_row_col:
            draw_axes(matrix, (x, y), with_row_col=with_row_col)
        else:
            draw_axes(matrix, (x - 0.3, y + 0.3), with_row_col=with_row_col)

    if with_row_col:
        draw_row(matrix.shape[0], (x - 1, y))
        draw_column(matrix.shape[1], (x - 1 + 0.2, y))

    rows = matrix.shape[0]
    draw_top(matrix[0], (x, y), color=color)
    if rows == 1:
        draw_vector(matrix[0], xy, title='', with_axis=False, color=color)
    else:
        for i in range(rows - 1):
            draw_vector_no_bottom(matrix[i], (x, y - i), color=color)

        draw_vector(matrix[i+1], (x, y - i - 1), title='', with_axis=False, color=color)
    draw_right((x + matrix.shape[1] - 1, y), rows, color=color)

def draw_sum_axis0(a=None, xy=(5.5,7)):
    if a is None:
        a = np.arange(16).reshape(4,4)
    
    x, y = xy
    draw_matrix(a, (x,y), title=r'$np.sum(a, axis=0)$')
    sum = np.sum(a, axis=0)
    
    draw_vector(sum, (x, y - a.shape[0] - 0.5), title='', with_axis=False)
    columns = a.shape[1]

    # draw arrows
    for i in range(columns):        
        ax.annotate("", xy=(x + 0.5 + i, y - a.shape[0] - 0.5), xytext=(x + 0.5 + i, y), 
                        arrowprops=dict(arrowstyle="simple", alpha=0.3, color='red'))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    plt.show()

def draw_sum_axis1(a=None, xy=(5,7)):
    if a is None:
        a = np.arange(16).reshape(4,4)

    x, y = xy
    draw_matrix(a, (x, y), title=r'$np.sum(a, axis=1)$')
    sum = np.sum(a, axis=1)
    
    draw_vertical_vector(sum, (x + a.shape[1] + 0.5 + depth, y), title='')
    rows = a.shape[0]

    # draw arrows
    for i in range(rows):        
        ax.annotate("", xy=(x + a.shape[1] + depth + 0.5, y - 0.5 - i), xytext=(x, y - 0.5 - i), 
                    arrowprops=dict(arrowstyle="simple", alpha=0.3, color='red'))
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    plt.show()

def draw_top_right_mask(ax, xy, depth=0.3, size=1, color='gray'):
    # draw right diamond
    x, y = xy

    points = [(x+size,y), (x+size+depth, y+depth), (x+size+depth, y-size+depth), 
              (x+size, y-size)]
    style = create_stype(color)
    top = Polygon(points, closed=True, **style)
    ax.add_patch(top)

    # draw top diamond
    x, y = xy
    points = [(x,y), (x+depth, y+depth), (x+size+depth, y+depth), (x+size, y)]
    style = create_stype(color)
    top = Polygon(points, closed=True, **style)
    ax.add_patch(top)

    draw_square_mask(ax, xy, size, color)

# draw imgs/numpy/narraytypes.png
def draw_vector_matrix_sample():
    a = np.arange(16).reshape(4,4)
    v = np.arange(4)
    draw_vector(v, (1,7))
    draw_matrix(a, (8,6.6), with_row_col=True)
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    plt.show()

def draw_hstack_sample():
    v1 = np.array([0,1,2])
    v2 = np.array([3,4]) * 10
    
    startx = 1
    starty = 8
    draw_vector(v1, (startx, starty), title='v1', with_axis=False)
    
    startx += len(v1) + 1
    draw_vector(v2, (startx, starty), title='v2', with_axis=False)
    
    startx += len(v2) + 1
    draw_vector(np.hstack([v1,v2]), (startx, starty), title='np.hstack([v1,v2])', with_axis=False)

    m1 = np.arange(6).reshape(2,3)
    m2 = (np.arange(4).reshape(2,2) + 3) * 10
    
    startx = 1
    starty = 4
    draw_matrix(m1, (startx, starty), title='m1', with_axis=False, with_row_col=False)
    startx += m1.shape[1] + 1
    draw_matrix(m2, (startx, starty), title='m2', with_axis=False, with_row_col=False)
    startx += m2.shape[1] + 1
    draw_matrix(np.hstack([m1,m2]), (startx, starty), title='np.hstack([m1,m2])', 
                with_axis=False, with_row_col=False)
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    plt.show()

def draw_vstack_sample():
    v1 = np.array([0,1])
    v2 = np.array([3,4]) * 10
    
    startx = 1
    starty = 8
    draw_vector(v1, (startx, starty), title='v1', with_axis=False)
    
    starty -= 2
    draw_vector(v2, (startx, starty), title='v2', with_axis=False)
    
    starty -= 3
    draw_matrix(np.vstack([v1,v2]), (startx, starty), title='np.vstack([v1,v2])', 
                with_axis=False, with_row_col=False)

    m1 = np.arange(4).reshape(2,2)
    m2 = (np.arange(6).reshape(3,2) + 3) * 10

    startx = 7
    starty = 9
    draw_matrix(m1, (startx, starty), title='m1', with_axis=False, with_row_col=False)
    draw_matrix(m2, (startx + m1.shape[1] + 1.5, starty), title='m2', with_axis=False, with_row_col=False)
    
    starty -= m2.shape[0] + 0.5
    draw_matrix(np.vstack([m1,m2]), (startx, starty), title='np.vstack([m1,m2])', 
                with_axis=False, with_row_col=False)
    
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    plt.show()

def draw_3d_diamond(xy, rl=(2,2), color='gray'):
    startx, starty = xy
    step = 0.5 * np.sin(np.pi/4)
    rows = rl[0]
    columns = rl[1]

    ax.arrow(startx, starty, columns, 0, width=0.05,
                color=color, alpha=0.5, head_width=0, head_length=0)
    for i in range(1,rows,1):
        ax.arrow(startx - i * step, starty - i * step, columns, 0, width=0.01, ls='--',
                    color=color, alpha=0.5, head_width=0, head_length=0)

    ax.arrow(startx - rows * step, starty - rows * step, columns, 0, width=0.05,
                color=color, alpha=0.5, head_width=0, head_length=0)

    dx = - 0.5 * rows * np.sin(np.pi/4)
    dy = - 0.5 * rows * np.cos(np.pi/4)

    ax.arrow(startx, starty, dx, dy, width=0.05,
                color=color, alpha=0.5, head_width=0, head_length=0)
    for i in range(1,columns,1):
        ax.arrow(startx + i, starty, dx, dy, width=0.01, ls='--',
                    color=color, alpha=0.5, head_width=0, head_length=0)

    ax.arrow(startx + columns, starty, dx, dy, width=0.05,
                color=color, alpha=0.5, head_width=0, head_length=0)

def draw_3d_matrix(xy, array, color='gray'):
    shape = array.shape
    
    x, y = xy
    heigth = shape[0]
    rl = (shape[1], shape[2])
    
    for i in range(heigth):
        draw_3d_diamond((x, y + i), rl, color=color)
        
# xy means the orgin point names can be 'x', 'y', 'z' array muse be 3D
def draw_3d_axes(xy, array, lens=[2,2,2], names=['0', '1', '2'], color='k'):
    if array.ndim != 3:
        print("only support 3D ndarray")
        return
    
    startx, starty = xy

    '''
    fig = plt.figure(figsize=(5, 3), facecolor='w')
    ax = plt.axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
    plt.style.use('ggplot')
    '''
    
    height, rows, columns = array.shape
    lens[0] = rows * 0.5 + 0.5
    lens[1] = columns + 0.5
    lens[2] = height - 1 + 0.5

    dx = - lens[0] * np.sin(np.pi/4)
    dy = - lens[0] * np.cos(np.pi/4)

    # latex style
    axis_font_size = 12
    axis_color = color
    for i in range(len(names)):
        names[i] = '${' + names[i] + '}$'
        
    ax.arrow(startx, starty, dx, dy, width=0.05, color=axis_color, clip_on=False, 
             head_width=0.15, head_length=0.15)
    ax.text(startx + dx - 0.3, starty + dy - 0.3, names[0],
            size=axis_font_size, ha='center', va='center')
    
    # second dimension
    dx = lens[1]
    endx = startx + dx 
    endy = starty
    ax.arrow(startx, starty, dx, 0, width=0.05, color=axis_color, clip_on=False, 
             head_width=0.15, head_length=0.15)

    ax.text(endx + 0.5, endy, names[1],
            size=axis_font_size, ha='center', va='center')
    
    # third dimension
    dy = lens[2]
    endx = startx 
    endy = starty + dy
    ax.arrow(startx, starty, 0, dy, width=0.05, color=axis_color, clip_on=False, 
             head_width=0.15, head_length=0.15)
    ax.text(endx, endy + 0.5, names[2],
            size=axis_font_size, ha='center', va='center')
    
    draw_3d_matrix(xy, array, color=axis_color)

def draw_3axes_sample():
    a = np.arange(8).reshape(2,2,2)
    draw_3d_axes((2,4.5), a, names=['x','y','z'], color='gray')
    ax.text(2.5, 2, "Cartesian", size=12, ha='center', va='center')
    draw_3d_axes((7,4.5), a, names=['0','1','2'], color='gray')
    ax.text(7.5, 2, "Octave/Matlab", size=12, ha='center', va='center')
    draw_3d_axes((12,4.5), a, names=['1','2','0'], color='gray')
    ax.text(12.5, 2, "Numpy", size=12, ha='center', va='center')
    
    plt.show()

def draw_octave_3d_axis_sample():
    a = np.array([0,2,1,3,4,6,5,7]).reshape(2,2,2)
    draw_3d_axes((3.5,4.5), a, names=['0','1','2'], color='gray')
    
    height, rows, columns = a.shape
    startx, starty = 8.5, 4.5
    
    endx, endy = startx, starty
    for i in range(height):
        draw_matrix(a[i], (endx,endy), title="", with_row_col=False, 
                    with_axis=False, color='gray')
        endx += columns
        endy += rows

    endx -= columns
    endy -= rows

    if height > 1:
        ax.plot([startx + depth, endx], [starty + depth, endy], 
                ls='--', lw=1, color='gray')
        ax.plot([startx + depth + columns, endx + columns], [starty + depth, endy], 
                ls='--', lw=1, color='gray')
        ax.plot([startx + depth + columns, endx + columns], [starty + depth - rows, endy - rows], 
                ls='--', lw=1, color='gray')

    # draw height axes
    ax.annotate("", xy=(endx - 0.5, endy), xytext=(startx - 0.5, starty),
                arrowprops=dict(arrowstyle="simple", color='black'))
    ax.text(endx - 0.5, endy, r'2',
            size=11, ha='center', va='bottom')    
    
    # draw horizontal
    ax.annotate("", xy=(startx + columns, starty), xytext=(startx - 0.5, starty),
                arrowprops=dict(arrowstyle="simple", color='black'))
    ax.text(startx + columns - 0.4, starty, r'1',
            size=11, ha='center', va='bottom')    
    
    # draw vertical
    ax.annotate("", xy=(startx - 0.5, starty - rows), xytext=(startx - 0.5, starty),
                arrowprops=dict(arrowstyle="simple", color='black'))
    ax.text(startx - 0.5, starty - rows - 0.5, r'0',
            size=11, ha='center', va='bottom')   
    
    draw_square_mask(ax, (endx, endy), size=1, color='red', alpha=0.3)
    
    plt.show()    

def draw_numpy_3d_axis_sample():
    a = np.arange(8).reshape(2,2,2, order='F')
    names=['1','2','0']
    draw_3d_axes((3.5,4.5), a, names=names, color='gray')
    
    height, rows, columns = a.shape
    startx, starty = 8.5, 4.5
    
    endx, endy = startx, starty
    for i in range(height):
        draw_matrix(a[i], (endx,endy), title="", with_row_col=False, 
                    with_axis=False, color='gray')
        endx += columns
        endy += rows

    endx -= columns
    endy -= rows

    if height > 1:
        ax.plot([startx + depth, endx], [starty + depth, endy], 
                ls='--', lw=1, color='gray')
        ax.plot([startx + depth + columns, endx + columns], [starty + depth, endy], 
                ls='--', lw=1, color='gray')
        ax.plot([startx + depth + columns, endx + columns], [starty + depth - rows, endy - rows], 
                ls='--', lw=1, color='gray')

    # draw height axes
    ax.annotate("", xy=(endx - 0.5, endy), xytext=(startx - 0.5, starty),
                arrowprops=dict(arrowstyle="simple", color='black'))
    ax.text(endx - 0.5, endy, names[2],
            size=11, ha='center', va='bottom')    
    
    # draw horizontal
    ax.annotate("", xy=(startx + columns, starty), xytext=(startx - 0.5, starty),
                arrowprops=dict(arrowstyle="simple", color='black'))
    ax.text(startx + columns - 0.4, starty, names[1],
            size=11, ha='center', va='bottom')    
    
    # draw vertical
    ax.annotate("", xy=(startx - 0.5, starty - rows), xytext=(startx - 0.5, starty),
                arrowprops=dict(arrowstyle="simple", color='black'))
    ax.text(startx - 0.5, starty - rows - 0.5, names[0],
            size=11, ha='center', va='bottom')
    
    draw_square_mask(ax, (startx+1, starty), size=1, color='red', alpha=0.3)
    
    plt.show()     

def draw_tree_index_sample():
    names=['1','2','0']
    a = np.arange(8).reshape(2,2,2)
    
    height, rows, columns = a.shape
    startx,starty = 5, 7
    
    endy = starty
    half = rows * 1.0 / 2
    
    # draw horizontal
    ax.annotate("", xy=(startx + columns, starty), xytext=(startx - 0.5, starty),
                arrowprops=dict(arrowstyle="simple", color='black'))
    ax.text(startx + columns - 0.4, starty, names[1],
            size=11, ha='center', va='bottom')    
    
    # draw vertical
    ax.annotate("", xy=(startx - 0.5, starty - rows), xytext=(startx - 0.5, starty),
                arrowprops=dict(arrowstyle="simple", color='black'))
    ax.text(startx - 0.5, starty - rows - 0.5, names[0],
            size=11, ha='center', va='bottom')
    
    for i in range(height):
        draw_matrix(a[i], (startx, endy), title='', with_row_col=False, with_axis=False)
        ax.plot([startx, startx - 1],
                [endy - half, endy - half], c='black', ls='-', lw=2)
        endy -= rows + 1
    
    endy += rows + 1
    endy -= half

    ax.plot([startx - 1, startx - 1],
            [starty - half, endy], c='black', ls='-', lw=2)
    
    # draw vertical
    ax.annotate("", xy=(startx - 1, starty - half + 0.3), xytext=(startx - 1, starty),
                arrowprops=dict(arrowstyle="simple", color='black'))
    ax.text(startx - 1, starty, names[2],
            size=11, ha='center', va='bottom')      

def create_indices_array(rows, columns, order='C'):
    indices_list = []
    
    if order == 'C':
        for i in range(rows):
            for j in range(columns):
                indices_list.append(str(i) + ',' + str(j))
    else:
        for i in range(columns):
            for j in range(rows):
                indices_list.append(str(j) + ',' + str(i))
    
    return np.array(indices_list)

def create_3indices_array(height, rows, columns, order='C'):
    indices_list = []
    
    if order == 'C':
        for i in range(height):
            for j in range(rows):
                for k in range(columns):
                    indices_list.append(str(i) + ',' + str(j) + ',' + str(k))
    else:
        for i in range(columns):
            for j in range(rows):
                for k in range(height):
                    indices_list.append(str(k) + ',' + str(j) + ',' + str(i))
    
    return np.array(indices_list)

def draw_row_first_sample():
    a = np.arange(9).reshape(3,3)
    
    startx,starty = 6, 7
    rows, columns = a.shape
    draw_matrix(a, (startx, starty), title='', with_row_col=True, with_axis=False, color=None)
    draw_top_nmask(ax, (startx, starty), columns, color='red')
    
    colors = ['red', 'green', 'blue']
    for i in range(rows):
        draw_square_nmask_with_right(ax, (startx, starty - i), columns, color=colors[i])

    startx, starty = 3, 2.5
    
    draw_vector_head((startx, starty), color=None)
    draw_vector(a.ravel(), (startx, starty), title="", with_axis=False, color=None)
    draw_vector_tail((startx + len(a.ravel()), starty), color=None)
    for i in range(rows):
        draw_square_nmask(ax, (startx + columns * i, starty), columns, color=colors[i])    

    indices = create_indices_array(3, 3, order='C')
    draw_indices_vector(indices, (startx, starty - 1), color=None)

def draw_column_first_sample():
    a = np.arange(9).reshape(3,3,order='F')

    startx,starty = 6, 7
    rows, columns = a.shape
    draw_matrix(a, (startx, starty), title='', with_row_col=True, with_axis=False, color=None)
    #draw_top_nmask(ax, (startx, starty), columns, color='red')
    
    colors = ['red', 'green', 'blue']
    for i in range(rows):
        draw_square_nmask_column(ax, (startx + i, starty), rows, color=colors[i])
    draw_right_nmask(ax, (startx + columns - 1, starty), rows, color=colors[-1])
    
    startx, starty = 3, 2.5
    
    draw_vector_head((startx, starty), color=None)
    draw_vector(a.T.ravel(), (startx, starty), title="", with_axis=False, color=None)
    draw_vector_tail((startx + len(a.T.ravel()), starty), color=None)
    for i in range(rows):
        draw_square_nmask(ax, (startx + columns * i, starty), columns, color=colors[i])
        
    indices = create_indices_array(3, 3, order='F')
    draw_indices_vector(indices, (startx, starty - 1), color=None)

def draw_indices_sample():
    startx, starty = 2, 7.5

    indices = create_indices_array(3, 3, order='C')
    draw_indices_vector(indices, (startx, starty), color=None)
    starty -= 1.5
    indices = create_indices_array(3, 3, order='F')
    draw_indices_vector(indices, (startx, starty), color=None)

    starty -= 2
    indices = create_3indices_array(2, 2, 3, order='C')
    draw_indices_vector(indices, (startx, starty), color=None)
    starty -= 1.5
    indices = create_3indices_array(2, 2, 3, order='F')
    draw_indices_vector(indices, (startx, starty), color=None)

if __name__ == "__main__":
    draw_tree_index_sample() 
    plt.show()
