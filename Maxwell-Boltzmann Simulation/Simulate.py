from Particles import Particles
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
import numpy as np

from time import time

################################################################################
## Simulation Parameters

NUM_PARTICLES = 100
SYSTEM_SIZE = 50

# frames per second
FPS = 30

# number of histogram bins
NUM_BINS = 60

# RGB of the ball color
# Use a color picker to choose the correct tuple
BALL_RGB = (0, 64, 3)

# Opacity of a still particle
MIN_OPACITY = 0.1

################################################################################

particles = Particles(N=NUM_PARTICLES, L=SYSTEM_SIZE, dt=1/FPS)

################################################################################
## Here the simulation can be set up

particles.set_speed(.1)
particles.freeze()
particles.randomize('RV')

################################################################################
################################################################################

colors = np.empty((particles.N, 4))
colors[:,:3] = np.ones((particles.N, 1)) * np.array(BALL_RGB)/255

def update_alphas():
    sp = particles.speeds() + particles.get_speed()*MIN_OPACITY
    sp = sp/sp.max()
    colors[:,3] = sp

# TODO figure out the scaling factor something to do with 72, 100, and pi
def get_sizes():
    bbox = pt_ax.get_window_extent().inverse_transformed(figure.dpi_scale_trans)
    return ((bbox.width*36)/(particles.L))**2*np.pi*1.24

# TODO Rearrange Subplots
# Left: particles in a box
# right top: speed histogram + theoretical distibution
# right bottom: temperature, energy or some other plot

figure = plt.figure(figsize=(18,9))
gs = gridspec.GridSpec(2,4)
pt_ax = plt.subplot(gs[:,:2])
hst_ax = plt.subplot(gs[0,-2:])
par_ax = figure.subplot(gs[:,:2])

pt_ax.set_xlim(0, particles.L)
pt_ax.set_ylim(0, particles.L)
pt_ax.set_aspect(1)

hst_ax.set_xlim(0, 3)
hst_ax.set_ylim(0, 3)
hst_ax.set_aspect(.5)

# update_alphas()
points = pt_ax.scatter(*particles.R,
                        s = get_sizes())#,
                        # c = colors, edgecolors = 'none')


# TODO FIX HISTOGRAMS
# TODO FIX ZORDER
v = np.linspace(0,3,100)
MB_dist, = hst_ax.plot(v, 4*np.pi**(-1/2)*v**2*np.exp(-v**2), 'tab:orange', zorder=10)

nspeeds = particles.normalized_speeds()
hst_bins = np.linspace(0, 3, NUM_BINS)
_, _, bar_container = hst_ax.hist(nspeeds, hst_bins, density=True)
# hst_ax.clear()

def update_histogram():
    nspeeds = particles.normalized_speeds()
    hst_n, _ = np.histogram(nspeeds, hst_bins, density=True)
    for h, bar in zip(hst_n, bar_container):
        bar.set_height(h)
        bar.set_zorder(10)
    MB_dist.set_zorder(1)

def update_particles():
    # update positions
    points.set_offsets(particles.R.T)

    # update particle alphas according to their speeds
    # update_alphas()
    # points.set_facecolor(colors)
    # points.set_edgecolor('none')

    # update particle sizes in case window gets resized
    # COMMENT OUT WHEN SAVING MOVIE
    points.set_sizes([get_sizes()]*particles.N)


def anim(frame):
    # simulate one step
    timer = time()
    particles.step()
    print((time() - timer)*1e3)
    # update Particles
    update_particles()

    # update histogram
    update_histogram()

    return bar_container + [points]


def main():
    movie = FuncAnimation(figure, anim, blit = True, interval=1/FPS)
    plt.show()

if __name__=='__main__':
    main()
