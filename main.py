import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from perlin_noise import PerlinNoise

def distance(p1,p2):
    dist = np.linalg.norm(p1 - p2)
    return dist

def generate_voronoi_points(size):
    points = []
    idx = 0
    for i in range(0,size):
        for j in range(0,size):
            offset = np.random.rand() * 0.45
            if (np.random.rand() > 0.5):
                offset *= -1
            points.append([i+offset,j+offset])
            idx += 1
    points = np.array(points)
    return points

def generate_noise(world, scale):
    idx = 0
    for cell in world:
        row = []
        # creating noise
        x = world[idx][1][0]
        y = world[idx][1][1]
        noise_val = 0.2 + noise1([x * scale / len(vor.points) * scale, y * scale / len(vor.points) * scale])
        noise_val = noise_val + noise2([x * scale / len(vor.points) * scale, y * scale / len(vor.points) * scale])

        world[idx] = [noise_val, world[idx][1]]
        row.append(noise_val)

        if idx < len(vor.points):
            idx += 1

def add_edge_noise(vor, num_of_line_segments):
    last_visited = []
    idx = 0
    for simplex in vor.ridge_vertices:
        pts = []

        simplex = np.asarray(simplex)
        line_noise = noise2([idx/len(vor.ridge_vertices),idx/len(vor.ridge_vertices)])

        first_pt = vor.vertices[simplex,0]
        last_pt = vor.vertices[simplex,1]
        pts.append(first_pt)
        if first_pt[0] > last_pt[0] or first_pt[1] > last_pt[1]:
            diff = first_pt-last_pt
            for n in range(1,4):
                pt = last_pt+(n*diff/4)
                pts.append(pt)

        elif last_pt[0] > first_pt[0] or last_pt[1] > first_pt[1]:
            diff = last_pt - first_pt
            for n in range(1,4):
                pt = first_pt+(n*diff/4)
                pts.append(pt)


        pts.append(last_pt)

        for i in range(0,len(points)):
            if i > 0:
                if np.all(simplex >= 0):
                    plt.plot(points[i-1], points[i], 'k-')

def color_cells(world,vor):
    for cell in world:

        # coloring cells
        point_index = np.argmin(np.sum((points - cell[1]) ** 2, axis=1))
        ridges = np.where(vor.ridge_points == point_index)[0]
        vertex_set = set(np.array(vor.ridge_vertices)[ridges, :].ravel())
        region = [x for x in vor.regions if set(x) == vertex_set][0]
        if not -1 in region:
            polygon = vor.vertices[region]
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

if __name__ == '__main__':
    seed = 185948245
    np.random.seed(seed)
    world_size = 36

    points = generate_voronoi_points(world_size)

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

    scale = 4

    generate_noise(world,scale)

    #add_edge_noise(vor,5)

    color_cells(world,vor)

    plt.figure(dpi=500)
    plt.show()
