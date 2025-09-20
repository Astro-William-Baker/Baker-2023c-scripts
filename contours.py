#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 12:21:26 2021

@author: will
"""

__author__='William M. Baker'
__date__='17/09/21'

import numpy as np
import scipy.stats as st
from scipy.interpolate import interp1d


"""Usage examples:
Plot contours onto a matplotlib scatter, hexbin plot etc.
import matplotlib.pyplot as plt
from contours import cont

fig, ax=plt.subplots()
f, xx, yy=cont(x, y)
ax.contour(xx,yy,f, levels=5, colors='grey')

can change levels
ax.contour(xx,yy,f, levels=[0.1, 0.7, 6.], colors='black')

can add filled contours
ax.contourf(xx, yy, f, cmap='binary', levels=[0.1, 0.7, 6.])
ax.contour(xx,yy,f, levels=[0.1, 0.7, 6.], colors='black')

Any other questions please feel free to contact me

"""



def cont(x,y, bw='scott'):
    delta_x=(max(x)-min(x))/10
    delta_y=(max(y)-min(y))/10
    
    xmin=min(x)- delta_x
    xmax=max(x)+ delta_x
    
    ymin=min(y)-delta_y
    ymax=max(y)+delta_y    
    
    xx, yy=np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    
    positions=np.vstack([xx.ravel(), yy.ravel()])
    values=np.vstack([x,y])

    kernel=st.gaussian_kde(values, bw_method=bw)
    f=np.reshape(kernel(positions), xx.shape)
    return f, xx,yy

def weighted_cont(x,y,weights, bw='scott'):
    delta_x=(max(x)-min(x))/10
    delta_y=(max(y)-min(y))/10
    
    xmin=min(x)- delta_x
    xmax=max(x)+ delta_x
    
    ymin=min(y)-delta_y
    ymax=max(y)+delta_y    
    
    xx, yy=np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    
    positions=np.vstack([xx.ravel(), yy.ravel()])
    values=np.vstack([x,y])
    #weights=np.vstack([weights,weights])
    #print(values)
    weights=np.array((weights))
    kernel=st.gaussian_kde(values, weights=weights, bw_method=bw)
    #print(kernel)
    f=np.reshape(kernel(positions), xx.shape)
    return f, xx,yy
                    

def percentiles_cont(x, y, percentiles=(0.9, 0.5, 0.3), weights=None, bw='scott'):
    """Create contours that contain (approximately) the percentiles `percentiles`
    of the scatter points"""

    delta_x=(max(x)-min(x))/10
    delta_y=(max(y)-min(y))/10
    
    xmin=min(x)- delta_x
    xmax=max(x)+ delta_x
    
    ymin=min(y)-delta_y
    ymax=max(y)+delta_y    
    
    xx, yy=np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    
    positions=np.vstack([xx.ravel(), yy.ravel()])
    values=np.vstack([x,y])
    weights = np.array((weights)) if weights is not None else weights

    kernel = st.gaussian_kde(values, weights=weights, bw_method=bw)

    f = np.reshape(kernel(positions), xx.shape)

    f /= f.sum()
    n = 1000
    t = np.linspace(0., f.max(), n)
    integral = ((f >= t[:, None, None]) * f).sum(axis=(1,2))
    fint = interp1d(integral, t)
    t_contours = fint(np.array(percentiles))
    t_contours = np.hstack([t_contours, np.inf])
    f[np.isclose(f, 0)]=0
    # Pad with zeros to close contours!
    f[0,  :] = 0
    f[-1, :] = 0
    f[:,  0] = 0
    f[:, -1] = 0

    return f, xx, yy, t_contours



def main():
    np.random.seed(100116)

    import matplotlib.pyplot as plt

    n_points = 10000
    fig, ax = plt.subplots()
    x0, y0 = np.random.normal(0, 1, size=n_points), np.random.normal(0.5, 1, size=n_points)
    x1, y1 = np.random.normal(0.2, 0.3, size=n_points), np.random.normal(-0.5, 0.5, size=n_points)
    x2, y2 = np.random.normal(1, 1, size=n_points), np.random.normal(-1, 1, size=n_points)

    x = np.hstack((x0, x1, x2))
    y = np.hstack((y0, y1, y2))

    f, xx, yy, t_contours = percentiles_cont(x, y, percentiles=(0.99, 0.9, 0.5, 0.1))

    ax.scatter(x, y, alpha=0.05, edgecolor='none', s=5, c='k')

    ax.contour(xx, yy, f, levels=t_contours, colors='forestgreen')
    ax.set_title('$\mathrm{Percentile\;contours}$', fontsize=20)

    plt.show(block=True)


if __name__=="__main__":
    main()
