import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from perlin_noise import PerlinNoise

def distance(p1,p2):
    dist = np.linalg.norm(p1 - p2)
    return dist


if __name__ == '__main__':
    seed = 185948245
    np.random.seed(seed)
    world_size = 38
    points = []
    idx = 0
    for i in range(0,world_size):
        for j in range(0,world_size):
            offset = np.random.rand() * 0.45
            if (np.random.rand() > 0.5):
                offset *= -1
            points.append([i+offset,j+offset])
            idx += 1
    points = np.array(points)
    vor = Voronoi(points)
    voronoi_plot_2d(vor, show_vertices=False)
    tri = Delaunay(points)

    noise1 = PerlinNoise(octaves=3,seed=seed)
    noise2 = PerlinNoise(octaves=6,seed=seed)


    print(len(vor.regions))
    print(len(vor.points))

    world = []
    for p in vor.points:
        world.append([0.0, p])
    #for r in vor.regions:
    idx = 0

    scale = 6
    pic = []

    ### for x in range(0,len(vor.points)):
     #   row = []
      #  for y in range(0,len(vor.points)):
       #     noise_val = 0.2 + noise1([x*scale/len(vor.points)*scale,y*scale/len(vor.points)*scale])
       #     world[idx] = [noise_val, world[idx][1]]
       #     row.append(noise_val)
      #  pic.append(row)

     #   if idx < len(vor.points):
       #     idx += 1

    for cell in world:
        row = []
        # creating noise
        x = world[idx][1][0]
        y = world[idx][1][1]
        noise_val = 0.2 + noise1([x * scale / len(vor.points) * scale, y * scale / len(vor.points) * scale])
        noise_val = noise_val + noise2([x * scale / len(vor.points) * scale, y * scale / len(vor.points) * scale])

        world[idx] = [noise_val, world[idx][1]]
        row.append(noise_val)
        pic.append(row)

        if idx < len(vor.points):
            idx += 1

    for cell in world:

        # coloring cells
        point_index = np.argmin(np.sum((points - cell[1]) ** 2, axis=1))
        ridges = np.where(vor.ridge_points == point_index)[0]
        vertex_set = set(np.array(vor.ridge_vertices)[ridges, :].ravel())
        region = [x for x in vor.regions if set(x) == vertex_set][0]
        if not -1 in region:
            polygon = vor.vertices[region]
            print(cell)
            if 0.15 < cell[0] < 0.25:
                plt.fill(*zip(*polygon), color='c')
            elif 0.25 < cell[0] < 0.4:
                plt.fill(*zip(*polygon), color='y')
            elif 0.4 < cell[0] < 0.65:
                plt.fill(*zip(*polygon), color='g')
            elif 0.65 < cell[0] < 0.85:
                plt.fill(*zip(*polygon), color='gray')
            elif cell[0] > 0.85:
                plt.fill(*zip(*polygon), color='white')
            else:
                plt.fill(*zip(*polygon), color='blue')
    #    for idx in r:
    #        print(vor.vertices[idx])
    plt.figure(dpi=500)
    #plt.imshow(pic, cmap='gray', alpha=1)
    plt.show()
