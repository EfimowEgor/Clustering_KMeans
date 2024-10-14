import numpy as np
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d import Axes3D


class Matplotlib3DGifCreator:
    def __init__(self, filename, names, fps, elev=30, azim=45):
        self.filename = filename
        self.fps = fps
        self.frames = []
        self.elev = elev
        self.azim = azim
        self.names = names

    def add_frame(self, X, labels, centroids, iteration):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap="viridis")
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker="x", s=100, c="red")
        ax.set_title(f"Iteration: {iteration}")
        ax.set_xlabel(self.names[0])
        ax.set_ylabel(self.names[1])
        ax.set_zlabel(self.names[2])

        ax.view_init(elev=self.elev, azim=self.azim)  

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        self.frames.append(image)
        plt.close(fig)

    def save(self):
        imageio.mimsave(self.filename, self.frames, fps=self.fps)