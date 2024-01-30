"""
Juha Närhi DATA:ML.100 Linear solver
Click the left mouse button as many times as you want and then click the right button.
Then linear fit is plotted
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
def my_linfit(x, y):

#Mathematical part (a and b)
    b = (sum(x * y) * sum(x) - sum(y) * sum(x * x)) / (sum(x * x) + sum(x) * sum(x))
    a = (-b * sum(x) + sum(x * y)) / sum(x * x)
    return a, b
#class for points
class points:
    def __init__(self):
        self.list_of_x = []
        self.list_of_y = []
    def getdata(self):
        return self.list_of_y,  self.list_of_x
def plot():
#Here the line is plotted


    list_of_x, list_of_y = point.getdata()

    xi = np.array(list_of_x)
    yi = np.array(list_of_y)

    a, b = my_linfit(xi, yi)
    plt.plot(xi, yi, "kx")
    xp = np.arange(0, 10, 0.1)
    plt.plot(xp, a * xp + b, "r-")
    print(f'My fit: a={b} and b={b}')


def onclick(event):
#Mouse clicking function
    if event.button is MouseButton.LEFT:
        point.list_of_y.append(event.xdata)
        point.list_of_x.append(event.ydata)
    if event.button is MouseButton.RIGHT:
        plot()
        fig.canvas.draw()

#figure
fig, ax = plt.subplots()
plt.grid()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_autoscale_on(False)
point = points()
press = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()


