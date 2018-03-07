import sqlite3
from pprint import pprint
import numpy as np
import tensorflow as tf
import time
import random

db = sqlite3.connect("../dataset/database.sqlite")

def write_to_file(target_attr,time, values, accuracy):
	f = open('../output/' + target_attr + '_vals.txt','w')
	f.write("%6.6f" % time + "\n")
	f.write("%.10f" % accuracy + "\n")
	for x in values:
		f.write("%.10f" % x + "\n")
	f.close()

def write_to_file_top_down(target, accuracies):
	f = open('../output/' + target + '_top_down_vals.txt', 'w')
	for key in accuracies:
		f.write(key + ":%3.8f" % accuracies[key] + '\n')
	f.close()

def write_to_file_bottom_up(target, accuracies):
	f = open('../output/' + target + '_bottom_up_vals.txt', 'w')
	for key in accuracies:
		f.write(key + ":%3.8f" % accuracies[key] + '\n')
	f.close()


def contains_none(r):
    for v in r:
        if v is None:
            return True
    return False

def select(db, table, stats):

    c = db.cursor()
    c.execute("select " + list_to_string(list(stats.keys())) + " from %s" % table + " order by random()")
    attr_names = [x for x in list(stats.keys())]
    while True:
        row = c.fetchone()
        if row == None:
            break
        elif contains_none(row):
        	continue
        else:
            yield list(zip(attr_names, row))
    c.close()

def peek_row(db, table):
    c = db.cursor()
    c.execute("select * from %s limit 1" % table)
    row = c.fetchone()
    c.close()
    return list(zip([x[0] for x in c.description], row))

def guess_datatype(attr_name, value):
    if attr_name.endswith("id"):
        return "ID"
    elif isinstance(value, int) or isinstance(value, float):
        return "NUM"
    elif len(value) < 10:
        return "CAT"
    else:
        return "UNKNOWN"

def get_num_stats(db, table, attr):
    c = db.cursor()
    c.execute('''
        select min(%s), max(%s) from %s
        ''' % (attr, attr, table))
    row = c.fetchone()
    c.close()
    return dict(datatype='NUM', min=row[0], max=row[1])

def get_cat_stats(db, t, a):
    c = db.cursor()
    c.execute('''
        select distinct %s as x from %s where x is not null order by x
        ''' % (a, t))
    rows = c.fetchall() 
    c.close()
    return dict(datatype='CAT', 
            size=len(rows), 
            domain=dict((x[0], i) for (i, x) in enumerate(rows)))

def get_table_stat(db, table):
    c = db.cursor()
    stats = dict()
    for attr_name, value in peek_row(db, table):
        datatype = guess_datatype(attr_name, value)
        if datatype == 'NUM':
            stats[attr_name] = get_num_stats(db, table, attr_name)
        elif datatype == 'CAT':
            stats[attr_name] = get_cat_stats(db, table, attr_name)
    return stats

def encode(stat, datavalue):
    "Returns a numpy vector"
    if stat['datatype'] == 'NUM':
        min, max = stat['min'], stat['max']
        vec = [(datavalue - min) / (max-min)]
    elif stat['datatype'] == 'CAT':
        vec = [0] * stat['size']
        i = stat['domain'].get(datavalue)
        if i is not None: vec[i] = 1
    return vec

def table_to_numpy(db, table):
	"Return a numpy 2D matrix, and an encoding scheme"
	stats = get_table_stat(db, table)
	matrix = []
	scheme = dict()
	for r in select(db, table, stats):
		row = []
		for (attr_name, value) in r:
			if attr_name in stats:
				i = len(row)
				row.extend(encode(stats[attr_name], value))
				j = len(row)
				if attr_name not in scheme:
					scheme[attr_name] = (i, j)
		matrix.append(row)
	return np.asarray(matrix), scheme

class Model():
    pass

def build_nn(table_stat, scheme, target_attr):
    "Building a MLP that accepts inputs which are not target_attr"

    def get_attr_dim(attr):
        stat = table_stat.get(attr)
        if stat['datatype'] == 'NUM':
            return 1
        else:
            return stat['size']

    def get_input_dim():
        dim = 0
        for attr in table_stat.keys():
            if attr != target_attr:
            	dim += get_attr_dim(attr)
        return dim

    def get_output_dim():
        return get_attr_dim(target_attr)

    def get_cost(ref, output):
        if table_stat[target_attr]['datatype'] == 'NUM':
            return tf.reduce_mean(tf.pow(ref - output, 2))
        else:
            return tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=ref, logits=output))

    N1 = 100
    N2 = 20
    N_output = get_output_dim()
    mlp = Model()
    mlp.input = tf.placeholder(tf.float32, (None, get_input_dim()))
    mlp.ref   = tf.placeholder(tf.float32, (None, N_output))
    mlp.L1    = tf.layers.dense(inputs=mlp.input, units=N1, activation=tf.nn.relu)
    mlp.L2    = tf.layers.dense(inputs=mlp.L1, units=N2, activation=tf.nn.relu)
    mlp.output = tf.layers.dense(inputs=mlp.L2, units=N_output)
    mlp.cost   = get_cost(mlp.ref, mlp.output)
    return mlp

def get_training_data(table, scheme, target_attr, n, train_size=1000):
    i, j = scheme[target_attr]
    x = np.hstack([table[n*train_size:n*train_size+train_size, :i], table[n*train_size:n*train_size+train_size, j:]])
    y = table[n*train_size:n*train_size+train_size, i:j]
    return x, y

def get_testing_data(table, i, train_size=1000, num_data=100):
    y = table[i*train_size:i*train_size+num_data, :]
    return y

def get_training_data_set(table, train_size=1000):
    return table[:train_size, :]

def get_testing_data_set(table, train_size=1000, num_data=100):
    return table[train_size:train_size+num_data, :]


def train(model, x, y):
    learning_rate = 0.0001
    epochs = 1000

    optimizer = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(model.cost)
    feed = {model.input: x, model.ref: y}
    values = []
    s = tf.Session()
    s.run(tf.global_variables_initializer())
    values.append(s.run(model.cost, feed))
    #print("Initial cost: %.7f" % values[0])
    for i in range(len(y)):
    	s.run(optimizer, feed)
    	values.append(s.run(model.cost, feed))
    	#print("Cost: %.6f" % values[i])
    #print("Final cost: %.7f" % values[-1])
    return values, model



def test(target_attr, test_data, model, table_stat, scheme):
	#splitting test_data into input values(x) and expected output(y)
	i, j = scheme[target_attr]
	x_vals = np.hstack([test_data[:,:i], test_data[:,j:]])
	y_vals = test_data[:,i:j]
	datatype = table_stat[target_attr]['datatype']
	s = tf.Session()
	s.run(tf.global_variables_initializer())
	accuracy = 0.0
	#if target attribute is numerical, calculate % correct of guess compared to actual value
	if(datatype == 'NUM'):
		guess_vals = s.run(model.output,feed_dict={model.input:x_vals}) 
		e = []
		#for each guess check how close to actual value it was
		for i in range(len(test_data)):
			if(y_vals[i] != 0):
				e.append(np.absolute(np.absolute(guess_vals[i])-y_vals[i])/y_vals[i])
			else:
				e.append(1-np.absolute(guess_vals[i]))
		#calculate accuracy
		accuracy = sum(e)/float(len(e))*100
	elif(datatype == 'CAT' and table_stat[target_attr]['size'] > 2):
		guess_vals = s.run(model.output,feed_dict={model.input:x_vals})
		num_correct = 0
		for i in range(len(test_data)):
			#get top two guesses
			index_1 = np.where(guess_vals[i]==max(guess_vals[i]))
			temp = list(filter(lambda x: x != max(guess_vals[i]), guess_vals[i]))
			index_2 = np.where(temp==max(temp))
			#check if one of top two guesses is correct
			if(y_vals[i][index_1]==1. or y_vals[i][index_2]==1.):
				num_correct+=1
		#calculate accuracy
		accuracy = num_correct/len(test_data)*100
	elif(datatype == 'CAT' and table_stat[target_attr]['size'] == 2):
		guess_vals = s.run(model.output,feed_dict={model.input:x_vals})
		num_correct = 0
		for i in range(len(test_data)):
			#get top guess
			index = np.where(guess_vals[i]==max(guess_vals[i]))
			#check if top guess is correct
			if(y_vals[i][index]==1.):
				num_correct+=1
		accuracy = num_correct/len(test_data)*100

	#print('Accuracy: %.4f' % accuracy)
	return accuracy

def list_to_string(list):
	s = ''
	for l in list:
		s += l + ', '
	return s[:-2]


def get_table_cols(db):
	c = db.execute('select * from Player_Attributes')
	return  [description[0] for description in c.description]

def regular_training(m, s, cols, training_size):
	i = 0
	for target in cols:
		if(target not in ['id', 'player_fifa_api_id', 'player_api_id', 'date']):
			print("Training Column: " + target)
			start_time = time.time()
			#target = 'preferred_foot'
			table_stat = get_table_stat(db, 'Player_Attributes')
			model = build_nn(table_stat,s,target)
			x, y = get_training_data(m, s, target, i, training_size)
			y_test= get_testing_data(m, i, training_size)
			values, model = train(model, x,y)
			accuracy = test(target, y_test, model, table_stat, s)
			end_time  = time.time()
			#print("Total time: %.3f" % end_time-start_time)
			write_to_file(target, end_time-start_time, values, accuracy)
			i+=1

def bottom_up(cols, training_data, test_data, scheme):
	cols = list(filter(lambda x: x not in ['id', 'player_fifa_api_id', 'player_api_id', 'date'], cols))
	for target in cols:
		print('Calculating bottom up accuracies for ' + target)
		table_stat = get_table_stat(db, 'Player_Attributes')
		i, j = scheme[target]
		target_training_data = training_data[:, i:j]
		target_test_data = test_data[:,i:j]
		target_scheme = dict()
		target_scheme[target] = (0, j-i)
		target_table_stat = dict()
		target_table_stat[target] = table_stat[target]
		col_list = list(filter(lambda x: x != target,cols))
		used_cols = []
		accuracies = dict()
		while(len(col_list) > 1):
			temp_acc = dict()
			for col in col_list:
				temp_scheme = dict((key, value) for key, value in target_scheme.items())
				temp_training_data = target_training_data
				temp_testing_data = target_test_data
				temp_table_stat = dict((key, value) for key, value in target_table_stat.items())
				temp_table_stat[col] = table_stat[col]
				i, j = scheme[col]
				temp_training_data = np.hstack([temp_training_data,training_data[:,i:j]])
				temp_testing_data = np.hstack([temp_testing_data,test_data[:,i:j]])
				max_val = max(temp_scheme.values())[1]
				temp_scheme[col] = (max_val, max_val+j-i)
				i, j = temp_scheme[target]
				x = np.hstack([temp_training_data[:, :i], temp_training_data[:, j:]])
				y = training_data[:, i:j]
				model = build_nn(temp_table_stat,temp_scheme,target)
				values, model = train(model, x,y)
				accuracy = test(target, temp_testing_data, model, temp_table_stat, temp_scheme)
				temp_acc[col] = accuracy
			highest_acc = next(iter(temp_acc))
			for key in temp_acc:
				if temp_acc[key] > temp_acc[highest_acc]:
					highest_acc = key
			print('Adding column ' + highest_acc)
			i, j = scheme[highest_acc]
			target_table_stat[highest_acc] = table_stat[col]
			target_training_data = np.hstack([target_training_data,training_data[:,i:j]])
			target_test_data = np.hstack([target_test_data, test_data[:,i:j]])
			max_val = max(target_scheme.values())[1]
			target_scheme[highest_acc] = (max_val, max_val+j-i)
			col_list = list(filter(lambda x: x != highest_acc, col_list))
			used_cols.append(highest_acc)
			print('Number of columns: %d' % len(used_cols))
			accuracy = temp_acc[highest_acc]
			accuracies[list_to_string(used_cols)] = accuracy
			print('Accuracy using columns ' + list_to_string(used_cols) + ' is: %.5f' % accuracy)

		write_to_file_bottom_up(target, accuracies)

def top_down(cols, training_data, test_data, scheme):
	cols = list(filter(lambda x: x not in ['id', 'player_fifa_api_id', 'player_api_id', 'date'], cols))
	for target in cols:
		table_stat = get_table_stat(db, 'Player_Attributes')
		target_training_data = training_data
		target_test_data = test_data
		target_scheme = scheme
		print('Calculating top down accuracies for ' + target)
		col_list = list(filter(lambda x: x != target,cols))
		accuracies = dict()
		i, j = target_scheme[target]
		x = np.hstack([target_training_data[:, :i], target_training_data[:, j:]])
		y = target_training_data[:, i:j]
		model = build_nn(table_stat,target_scheme,target)
		values, model = train(model, x,y)
		accuracy = test(target, target_test_data, model, table_stat, target_scheme)
		accuracies['all'] = accuracy
		print('Accuracy using all columns: %.5f' % accuracy )
		while(len(col_list) > 1):
			temp_acc = dict()
			for col in col_list:
				temp_table_stat = dict((key, value) for key, value in table_stat.items() if key != col)
				i, j = target_scheme[target]
				x = np.hstack([target_training_data[:, :i], target_training_data[:, j:]])
				y = target_training_data[:, i:j]
				model = build_nn(table_stat,target_scheme,target)
				values, model = train(model, x,y)
				accuracy = test(target, target_test_data, model, table_stat, target_scheme)
				temp_acc[col] = accuracy
			lowest_acc = next(iter(temp_acc))
			for key in temp_acc:
				if temp_acc[key] < temp_acc[lowest_acc]:
					lowest_acc = key
			print('Removing column ' + lowest_acc)
			i, j = target_scheme[lowest_acc]
			print('%d, %d' % (i, j))
			target_training_data = np.hstack([target_training_data[:, :i], target_training_data[:, j:]])
			target_test_data = np.hstack([target_test_data[:, :i], target_test_data[:, j:]])
			table_stat = dict((key, value) for key, value in table_stat.items() if key != lowest_acc)
			for key in target_scheme.keys():
				if target_scheme[key][0] > i:
					target_scheme[key] = (target_scheme[key][0]-(j-i), target_scheme[key][1]-(j-i))
			col_list = list(filter(lambda x: x != lowest_acc, col_list))
			print('Number of columns: %d' % len(col_list))
			accuracy = temp_acc[lowest_acc]
			accuracies[list_to_string(col_list)] = accuracy
			print('Accuracy using columns ' + list_to_string(col_list) + ' is: %.5f' % accuracy)

		write_to_file_top_down(target, accuracies)


if __name__ == '__main__':
	m, s = table_to_numpy(db, 'Player_Attributes')
	print(m.shape)
	cols = get_table_cols(db)
	training_size = 1000
	train_vals = get_training_data_set(m, training_size)
	test_vals= get_testing_data_set(m, training_size, 100)
	top_down(cols, train_vals, test_vals, s)
	bottom_up(cols, train_vals, test_vals, s)