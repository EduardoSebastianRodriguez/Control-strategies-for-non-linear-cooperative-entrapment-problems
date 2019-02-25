import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


"""
This function draws an object in the location specified by loc,
with the colour specified by 'colour' and with the size specified
by 'size':
    
    * 'loc' must be a 3 dimension vector [x,y,th]
    
    * 'size' must be a scalar related to the dimension of the field (D)
    
"""

def drawbrobot(loc,colour,size):
    
    """ Setting the characteristic dimension of the object """
    side           = 0.5*size/50

    """ Setting the square which represents the object """
    bottom_right   = np.array([side,-side,1])
    bottom_left    = np.array([-side,-side,1])
    top_right      = np.array([side,side,1])
    top_left       = np.array([-side,side,1])
    
    """ Extract the x,y and theta variables from loc """
    x              = loc[0]
    y              = loc[1]
    th             = loc[2]
    
    """ Homogeneous transform matrix constructed from x,y aand theta """
    H              = np.array([[np.cos(th), -np.sin(th), x],
                              [np.sin(th), np.cos(th), y], 
                              [0,        0 ,        1]])
    
    """ The square is built up """
    square         = np.array([bottom_right, bottom_left, top_left, top_right, bottom_right])
    
    """ The square is transformed to represent the location of the object """
    figure         = np.dot(H,np.transpose(square))
    
    """ The square is plotted with the colour specified """
    plt.plot(figure[0,:], figure[1,:], colour)
  
  
def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None, legend = None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean, label = legend)
    
class LegendObject(object):
    def __init__(self, facecolor='red', edgecolor='white', dashed=False):
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dashed = dashed
 
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            [x0, y0], width, height, facecolor=self.facecolor,
            # and whose edges are the faded color
            edgecolor=self.edgecolor, lw=3)
        handlebox.add_artist(patch)
 
        # if we're creating the legend for a dashed line,
        # manually add the dash in to our rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2*width/5, y0], width/5, height, facecolor=self.edgecolor,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)
 
        return patch