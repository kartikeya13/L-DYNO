import numpy as np
import matplotlib.pyplot as plt

def read_6dof_data_from_txt(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            elements = line.strip().split()
            t_x, t_y, t_z, roll, pitch, yaw = map(float, elements)
            data.append([t_x, t_y, t_z, roll, pitch, yaw])
    return np.array(data)

def plot_trajectory(data):
    plt.figure()
    plt.plot(data[:, 0], data[:, 2], 'b-', label='Trajectory')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Trajectory Plot')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Usage
input_file_path = "output_trans.txt"
data = read_6dof_data_from_txt(input_file_path)
plot_trajectory(data)

