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


def contains_none(r):
    for v in r:
        if v is None:
            return True
    return False

def select(db, table):
    c = db.cursor()
    c.execute("select * from %s" % table + " order by random()")
    attr_names = [x[0] for x in c.description]
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
    for r in select(db, table):

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

def get_training_data(table, scheme, target_attr, i, train_size=1000):
    i, j = scheme[target_attr]
    x = np.hstack([table[i*train_size:i*train_size+train_size, :i], table[i*train_size:i*train_size+train_size, j:]])
    y = table[i*train_size:i*train_size+train_size, i:j]
    print(x.shape)
    print(y.shape)
    return x, y

def get_testing_data(table, i, train_size=1000, num_data=100):
    y = table[i*train_size:i*train_size+num_data, :]
    return y

def train(model, x, y):
    learning_rate = 0.001
    epochs = 10000

    optimizer = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(model.cost)

    feed = {model.input: x, model.ref: y}
    values = []
    s = tf.Session()
    s.run(tf.global_variables_initializer())
    values.append(s.run(model.cost, feed))
    print("Initial cost: %.7f" % values[0])
    for i in range(len(y)):
    	s.run(optimizer, feed)
    	values.append(s.run(model.cost, feed))
    	#print("Cost: %.6f" % values[i])
    print("Final cost: %.7f" % values[-1])
    return values, model



def test(target_attr, test_data, model, table_stat, scheme):
	#splitting test_data into input values(x) and expected output(y)
	i, j = scheme[target_attr]
	x_vals = np.hstack([test_data[:,:i], test_data[:,j:]])
	y_vals = test_data[:,i:j]
	datatype = table_stat[target_attr]['datatype']
	s = tf.Session()
	s.run(tf.global_variables_initializer())
	#if target attribute is numerical, calculate % correct of guess compared to actual value
	if(datatype == 'NUM'):
		guess_vals = s.run(model.output,feed_dict={model.input:x_vals}) 
		e = []
		for i in range(len(test_data)):
			e.append(np.absolute(np.absolute(guess_vals[i])-y_vals[i])/y_vals[i])
		accuracy = sum(e)/float(len(e))*100
		print('Accuracy: %.4f' % accuracy)
		return accuracy
	elif(datatype == 'CAT' and table_stat[target_attr]['size'] > 2):
		guess_vals = s.run(model.output,feed_dict={model.input:x_vals})
		num_correct = 0
		for i in range(len(test_data)):
			index_1 = np.where(guess_vals[i]==max(guess_vals[i]))
			temp = list(filter(lambda x: x != max(guess_vals[i]), guess_vals[i]))
			index_2 = np.where(temp==max(temp))
			if(y_vals[i][index_1]==1. or y_vals[i][index_2]==1.):
				num_correct+=1
		accuracy = num_correct/len(test_data)
		print('Accuracy: %.4f' % accuracy*100)
		return accuracy
	elif(datatype == 'CAT' and table_stat[target_attr]['size'] == 2):
		guess_vals = s.run(model.output,feed_dict={model.input:x_vals})
		num_correct = 0
		for i in range(len(test_data)):
			index = np.where(guess_vals[i]==max(guess_vals[i]))
			if(y_vals[i][index]==1.):
				num_correct+=1
		accuracy = num_correct/len(test_data)

		print('Accuracy: %.4f' % accuracy*100)
		return accuracy



def get_table_cols(db):
	c = db.execute('select * from Player_Attributes')
	return  [description[0] for description in c.description]

if __name__ == '__main__':
    m, s = table_to_numpy(db, 'Player_Attributes')
    print(m.shape)
    cols = get_table_cols(db)
    i = 0
    for target in cols:
    	if(target not in ['id', 'player_fifa_api_id', 'player_api_id', 'date']):
	    	print("Training Column: " + target)
	    	start_time = time.time()
	    	#target = 'preferred_foot'
	    	table_stat = get_table_stat(db, 'Player_Attributes')
	    	model = build_nn(table_stat,s,target)
	    	x, y = get_training_data(m, s, target, i, 1000)
	    	y_test= get_testing_data(m, i, 1000)
	    	values, model = train(model, x,y)
	    	accuracy = test(target, y_test, model, table_stat, s)
	    	end_time  = time.time()
	    	#print("Total time: %.3f" % end_time-start_time)
	    	write_to_file(target, end_time-start_time, values, accuracy)
	    	i+=1
