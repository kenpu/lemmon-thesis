import sqlite3
from pprint import pprint
import numpy as np

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

def build_nn(matrix, stats, scheme, target_attr):
    x = tf.placeholder(tf.float32,[None,matrix.shape[0]])
    W = tf.Variable(tf.zeros(matrix.shape))
	b = tf.Variable(tf.zeros([matrix.shape[1]]))

	y = tf.nn.softmax(tf.matmul(x, W) + b)

	y_ = tf.placeholder(tf.float32, [None, matrix.shape[1]])

def main():
    m, s = table_to_numpy(db, 'Player_Attributes')
    print(m.shape)
    build_nn(m,get_stats(db, 'Player_Attributes'),s,'overall_rating')

if __name__ == '__main__':
    main()
