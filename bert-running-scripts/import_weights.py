import tensorflow as tf
import pickle
import numpy as np
import os

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "import_from_checkpoint",
    None,
    "Path for initial checkpint to read weights - eg. Path for Google's pre-trained (multilingual/base/large) BERT")

flags.DEFINE_string(
    "import_to_checkpoint",
    None,
    "Path for empty extended BERT checkpint")

flags.DEFINE_string(
    "output_dir",
    None,
    "Path to store the extended model")

flags.DEFINE_string(
    "run",
    None,
    "")

### we separate into 2 runs so that the tf graphs don't interfere with each other
def run1():

    '''
    create model from the meta file of inital checkpoint
    '''
    init_checkpoint_meta = FLAGS.import_from_checkpoint + '.meta'
    saver = tf.train.import_meta_graph(init_checkpoint_meta)

    '''
    Get names of all variables 
    '''

    graph = tf.get_default_graph()
    all_weight_names = [item.name for item in graph.get_collection('variables')]

    '''
    create a dictionary of weight names - value 
    '''

    weight_name_value = {}

    with tf.Session() as sess:
        saver.restore(sess, FLAGS.import_from_checkpoint)
        for weight_name in all_weight_names:
            weight = sess.run(weight_name)
            weight_name_value[weight_name] = weight

    pickle.dump(weight_name_value, open("tmp.pk", 'wb+'))

def run2():
    weight_name_value = pickle.load(open("tmp.pk", 'rb'))
    extended_checkpoint_meta = FLAGS.import_to_checkpoint + '.meta'

    '''
    Read the empty extended bert along with their weights
    '''

    saver = tf.train.import_meta_graph(extended_checkpoint_meta)
    sess = tf.Session()
    saver.restore(sess, FLAGS.import_to_checkpoint)

    '''
    Get names of all the variables in extended BERT and check if they are in init weights
    '''

    graph = tf.get_default_graph()
    extended_bert_names = []

    for item in graph.get_collection('variables'):
        extended_bert_names.append(item.name)


    is_in_init = []

    for item in extended_bert_names:
        try:
            temp = weight_name_value[item]
            is_in_init.append(True)
        except:
            is_in_init.append(False)


    print('******* number of variables = ' , len(extended_bert_names)  , ' *************')

    for (i, weight_name) in enumerate(extended_bert_names):
        is_name_in_init = is_in_init[i]
        if is_name_in_init:
            try:
                v = sess.graph.get_tensor_by_name(weight_name)
                init_weight = weight_name_value[weight_name]
                assignment = tf.assign(v, init_weight)
                sess.run(assignment)
                print(i, weight_name)
            except:
                print("Not assigned : ", i, weight_name )



    sp_weight_1 = 'bert/embeddings/word_embeddings:0'
    sp_weight_2 = 'cls/predictions/output_bias:0'



    init_weight = weight_name_value[sp_weight_1]
    extended_rn_vec = sess.graph.get_tensor_by_name(sp_weight_1)
    l = init_weight.shape[0]
    rn_vec = sess.run(extended_rn_vec[l:])

    extended_init_vec = np.concatenate([init_weight, rn_vec])
    assignment = tf.assign(extended_rn_vec, extended_init_vec)
    sess.run(assignment)

    print(sp_weight_1, "  Written ")

    init_weight = weight_name_value[sp_weight_2]
    extended_rn_vec = sess.graph.get_tensor_by_name(sp_weight_2)
    l = init_weight.shape[0]
    rn_vec = sess.run(extended_rn_vec[l:])

    extended_init_vec = np.concatenate([init_weight, rn_vec])
    assignment = tf.assign(extended_rn_vec, extended_init_vec)
    sess.run(assignment)

    print(sp_weight_2, "  Written ")

    output = os.path.join(FLAGS.output_dir , 'model.ckpt')
    new_saver = tf.train.Saver()
    save_path = new_saver.save(sess, output)
    print("************* Completed  Successfully *************")

if FLAGS.run == "1":
    run1()
elif FLAGS.run == "2":
    run2()
else:
    assert False
