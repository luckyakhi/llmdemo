from matplotlib import pyplot as plt
import random


def simple_line_plot(x_axis_array, y_axis_array):
    plt.plot(x_axis_array, y_axis_array)
    plt.title("Simple plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

def scattered_plot(x_axis_array, y_axis_array):
    plt.scatter(x_axis_array, y_axis_array)
    plt.title("Scattered plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

def histogram(data_array, bins):
    plt.hist(data_array, bins)
    plt.title("Histogram plot")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()



if __name__ == '__main__':
    #x_axis_array = [1, 2, 3, 4, 5]
    #y_axis_array = [2, 3, 5, 7, 11]
    #Generate random data of numbers between 1 and 100

    data = [random.randint(1, 100) for _ in range(100)]
    histogram(data,5)