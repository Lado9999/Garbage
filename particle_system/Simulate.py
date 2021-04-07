from Particles import Particles
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from time import time
import numpy as np

fps = 60

particles = Particles(N = 400,
                      dt=1/fps,
                      system_size=100)

# particles.randomize('RV')
choice = np.arange(0,100,3)
particles.V[:,choice] = (2*np.random.rand(2,choice.size)-1)*particles.speed

figure, axes = plt.subplots(1, 2, figsize=(18,9))
pt_ax , hst_ax = axes

pt_ax.set_xlim(0, particles.W)
pt_ax.set_ylim(0, particles.H)
pt_ax.set_aspect(1)

# TODO fix circle sizes
bbox = pt_ax.get_window_extent().inverse_transformed(figure.dpi_scale_trans)
width, height = bbox.width, bbox.height
width *= figure.dpi
height *= figure.dpi

colors = np.zeros((particles.N, 4))
# colors[:,2] = np.ones(particles.N)
colors[:,3] = np.ones(particles.N)
s = height/particles.H*3

points = pt_ax.scatter(particles.R[0], particles.R[1], s = [s]*particles.N, c=colors)

hst_ax.set_xlim(0, 2*particles.speed)
hst_ax.set_ylim(0, particles.N/7)
hst_ax.set_aspect(1)

speeds = particles.speeds()
hst_bins = np.linspace(0, np.exp(1)*particles.speed, 50)
_, _, bar_container = hst_ax.hist(speeds, hst_bins)

def anim(frame):
    timer = time()
    particles.step()
    timer = time() - timer
    print(f'Elapsed Time: {timer*1e3} ms')
    points.set_offsets(particles.R.T)
    speeds = particles.speeds()
    hst_n, _ = np.histogram(speeds, hst_bins)
    for h, bar in zip(hst_n, bar_container):
        bar.set_height(h)
    sp = particles.speeds() + particles.speed/5
    sp = sp/sp.max()
    colors[:,3] = sp
    points.set_color(colors)
    return bar_container + [points]

def main():
    movie = FuncAnimation(figure, anim, blit = True, interval=0)
    plt.show()

if __name__=='__main__':
    main()
