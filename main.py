import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from perlin_noise import PerlinNoise

def distance(p1,p2):
    dist = np.linalg.norm(p1 - p2)
    return dist


if __name__ == '__main__':
    seed = 176235471
    np.random.seed(seed)
    world_size = 20
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
    voronoi_plot_2d(vor)
    tri = Delaunay(points)

    noise1 = PerlinNoise(octaves=10,seed=seed)
    noise2 = PerlinNoise(octaves=20)
    noise3 = PerlinNoise(octaves=30)

    print(len(vor.regions))
    print(len(vor.points))

    land = []
    for p in vor.points:
        land.append([False, p])
    #for r in vor.regions:
    idx = 0

    scale = 10
    for x in range(0,len(vor.points)):
        for y in range(0,len(vor.points)):
            noise_val = noise1([x*scale/len(vor.points),y*scale/len(vor.points)*scale])
            noise_val += 0.45 + noise2([x/len(vor.points),y/len(vor.points)])
            noise_val += 0.2 + noise3([x/len(vor.points),y/len(vor.points)])
            if noise_val > 0.65:
                land[idx] = [True, land[idx][1]]
            elif noise_val <= 0.65:
                land[idx] = [False, land[idx][1]]
        #coloring cells
        point_index = np.argmin(np.sum((points - land[idx][1]) ** 2, axis=1))
        ridges = np.where(vor.ridge_points == point_index)[0]
        vertex_set = set(np.array(vor.ridge_vertices)[ridges, :].ravel())
        region = [x for x in vor.regions if set(x) == vertex_set][0]
        if not -1 in region:
            polygon = vor.vertices[region]

            if land[idx][0] == True:
                plt.fill(*zip(*polygon), color='green')
            else:
                plt.fill(*zip(*polygon), color='blue')
        if idx < len(vor.regions):
            idx +=1
    #    for idx in r:
    #        print(vor.vertices[idx])
    plt.figure(dpi=300)
    plt.show()
