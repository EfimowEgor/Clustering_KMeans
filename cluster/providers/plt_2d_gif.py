import numpy as np
import matplotlib.pyplot as plt
import imageio


class Matplotlib2DGifCreator:
    def __init__(self, names, filename, fps):
        self.filename = filename
        self.fps = fps
        self.frames = []
        self.names = names

    def add_frame(self, X, labels, centroids, iteration):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
        ax.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=100, c="red")
        ax.set_title(f"Iteration: {iteration}")
        ax.set_xlabel(self.names[0][0])
        ax.set_ylabel(self.names[1][0])

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        self.frames.append(image)
        plt.close(fig)

    def save(self):
        imageio.mimsave(self.filename, self.frames, fps=self.fps)
