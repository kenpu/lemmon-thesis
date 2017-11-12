import sqlite3
from pprint import pprint
import numpy as np
import tensorflow as tf

db = sqlite3.connect("../dataset/database.sqlite")

def select(db, table):
    c = db.cursor()
    c.execute("select * from %s" % table)
    attr_names = [x[0] for x in c.description]
    while True:
        row = c.fetchone()
        if row == None:
            break
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
        select distinct %s as x from %s order by x
        ''' % (a, t))
    rows = c.fetchall()
    c.close()
    return dict(datatype='CAT', 
            size=len(rows), 
            domain=dict((i, x[0]) for (i, x) in enumerate(rows)))

def get_stats(db, table):
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

def contains_none(r):
    for (a, v) in r:
        if v is None:
            return True
    return False

def table_to_numpy(db, table):
    "Return a numpy 2D matrix, and an encoding scheme"
    stats = get_stats(db, table)
    matrix = []
    scheme = dict()
    for r in select(db, table):
        if contains_none(r):
            continue

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

def build_nn(stats, scheme, target_attr):
    "Building a MLP that accepts inputs which are not target_attr"

    def get_attr_dim(attr):
        stat = stats.get(attr)
        if stat['datatype'] == 'NUM':
            return 1
        else:
            return stat['size']

    def get_input_dim():
        dim = 0
        for attr in stats.keys():
            if attr != target_attr:
            	dim += get_attr_dim(attr)
        return dim

    def get_output_dim():
        return get_attr_dim(target_attr)

    def get_cost(ref, output):
        if stats[target_attr]['datatype'] == 'NUM':
            return tf.reduce_mean(tf.sqrt(tf.pow(ref - output, 2)))
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
    mlp.L1    = tf.layers.dense(
                    inputs=mlp.input, units=N1, activation=tf.nn.relu)
    mlp.L2    = tf.layers.dense(
                    inputs=mlp.L1, units=N2, activation=tf.nn.relu)
    mlp.output = tf.layers.dense(
                    inputs=mlp.L2, units=N_output)
    mlp.cost   = get_cost(mlp.ref, mlp.output)
    return mlp

def get_training_data(table, scheme, target_attr):
    i, j = scheme[target_attr]
    x = np.hstack([table[:, :i], table[:, j:]])
    y = table[:, i:j]
    return x, y

def train(model, x, y):
    learning_rate = 0.01
    epochs = 1000

    optimizer = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(model.cost)

    feed = {model.input: x, model.ref: y}

    s = tf.Session()
    s.run(tf.global_variables_initializer())
    print("Initial cost: %.2f" % s.run(model.cost, feed))

if __name__ == '__main__':
    m, s = table_to_numpy(db, 'Player_Attributes')
    print(m.shape)
    model = build_nn(get_stats(db, 'Player_Attributes'),s,'overall_rating')
    train(model, *get_training_data(m, s, 'overall_rating'))
