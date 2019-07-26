from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


def main():
	result = open('value','r')
	curve = np.zeros((50,50),dtype='float64')

	i = 0
	x_axis = np.zeros(50)
	y_axis = np.zeros(50)

	for item in result:

	    curve[i]=np.array(item.strip().split())
	    x_axis[i]=(i * 1.7 / 50)-1.2
	    y_axis[i]=(i * 0.14 /50)-0.07
	    i += 1

	x_axis,y_axis = np.meshgrid(x_axis,y_axis)


	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	ax.plot_wireframe(x_axis,y_axis,curve)
	plt.show()
if __name__ == '__main__':
    main()
