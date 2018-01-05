import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_file(filename):
	vals = []
	try:
		f = open('../output/' + filename, 'r')
		time = f.readline()
		line = f.readline().rstrip()
		while line:
			vals.append(float(line))
			line = f.readline().rstrip()
	finally:
		f.close()
	return time, vals

def create_graph(filename, time, vals):
	plt.plot(vals)
	plt.title('Training Values For Column ' + filename[:-4] + '\nMax value: %.6f' % max(vals) + ' Min value: %.6f' % min(vals))
	plt.ylabel('Training Cost')
	plt.xlabel('Number Of Training Iterations')
	plt.savefig('../output/graphs/'+ filename[:-4] + '.png', bbox_inches='tight')
	plt.clf()

if __name__ == '__main__':
	for filename in os.listdir('../output'):
		if filename.endswith('.txt'):
			time, vals = load_file(filename)
			create_graph(filename, time, vals)