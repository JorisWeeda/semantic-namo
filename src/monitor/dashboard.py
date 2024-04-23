
import numpy as np
import matplotlib.pyplot as plt


class Dashboard:
    def __init__(self):
        self.fig, (self.ax_1, self.ax_2, self.ax_3) = plt.subplots(1, 3)
        
        plt.ion()
        plt.show()

    def update_plot(self, rollouts, all_states, best_states, pose, goal):
        colors = plt.cm.viridis(np.linspace(0, 1, all_states.shape[0]))

        self.ax_1.clear()
        for i in range(all_states.shape[0]):
            x_values, y_values = all_states[i, :, 0], all_states[i, :, 2]
            self.ax_1.plot(x_values, y_values, color=colors[i], alpha=0.4)

        self.ax_1.plot(best_states[0, :, 0], best_states[0, :, 2], color='red', alpha=1.0)

        self.ax_1.set_title('Robot DOF Position')
        self.ax_1.autoscale(enable=True)
        self.ax_1.set_aspect('equal')

        self.ax_2.clear()
        for i in range(rollouts.shape[1]):
            x_values, y_values = rollouts[:, i, 0], rollouts[:, i, 1]
            self.ax_2.plot(x_values, y_values, color=colors[i])

        self.ax_2.set_title('Rollouts Base Link')
        self.ax_2.autoscale(enable=True)
        self.ax_2.set_aspect('equal')

        self.ax_3.clear()
        self.ax_3.set_xlim(-4, 4)
        self.ax_3.set_ylim(-4, 4)

        self.ax_3.scatter(goal[1], goal[0], color='red', label='Goal')
        self.ax_3.scatter(pose[2], pose[0], color='green', label='Robot')

        self.ax_3.set_title('Coordinates')
        self.ax_3.set_aspect('equal')

        self.ax_3.invert_xaxis()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
