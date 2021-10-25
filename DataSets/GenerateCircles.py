import numpy as np
from sklearn.datasets import make_circles

def makeSingleCircleDataSet(nPoints, factor = 0.6, noise = 0.05):
    points, circle = make_circles(n_samples = nPoints, factor = factor, noise = noise)
    # plt.scatter(points[circle == 0, 0], points[circle == 0, 1], c = "blue")
    # plt.scatter(points[circle == 1, 0], points[circle == 1, 1], c = "red")
    # plt.axis("equal")
    # plt.show()
    circle = circle.reshape(nPoints, 1)
    return points, circle