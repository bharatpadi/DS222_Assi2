import tensorflow as tf
import numpy as np
import random
import datetime
import time
import threading
import multiprocessing as mp

train_start = time.time()

train_data = np.load('/home/bharatp/MLLD/Ass2/Ass1/DBPedia.full/train_mat.npy')
train_labels = np.load('/home/bharatp/MLLD/Ass2/Ass1/DBPedia.full/train_lab_onehot_row_averaged.npy')

val_data = np.load('/home/bharatp/MLLD/Ass2/Ass1/DBPedia.full/dev_mat.npy')
val_labels = np.load('/home/bharatp/MLLD/Ass2/Ass1/DBPedia.full/dev_lab_onehot_row_averaged.npy')

test_data = np.load('/home/bharatp/MLLD/Ass2/Ass1/DBPedia.full/test_mat.npy')
test_labels = np.load('/home/bharatp/MLLD/Ass2/Ass1/DBPedia.full/test_lab_onehot_row_averaged.npy')

X_train = train_data
y_train = train_labels

def get_accuracy(pred_labels, true_labels):
    acc = 0.0
    for i in range(len(pred_labels)):
        if true_labels[i,pred_labels[i]] > 0:
            acc += 1
    return 1.0 * acc / len(pred_labels)

def batch_iter(data, labels, batch_size, num_epochs, seed):
    random.seed(seed)
    data_size = len(data)
    for e in range(num_epochs):
        for b in range(data_size/batch_size):
            l = random.sample(range(data_size), batch_size)
            batch = data[l]
            batch_labels = labels[l]
            yield batch, batch_labels

max_words = 10000
beta = 0.006

input_x = tf.placeholder(tf.float32, [None, max_words])
input_y = tf.placeholder(tf.float32,[None, 50])

w = tf.Variable(tf.truncated_normal([max_words, 50], stddev = 0.1))
b = tf.Variable(tf.random_normal([50]))

global_step = tf.Variable(0, name="global_step", trainable=False)
logits = tf.add(tf.matmul(input_x, w), b)
predictions = tf.nn.softmax(logits)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = input_y) + beta*tf.nn.l2_loss(w) + beta*tf.nn.l2_loss(b))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
grads_and_vars = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
pred_label = tf.argmax(predictions,1)
correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(input_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

batch_size = 50
no_epochs = 10
data_size = len(X_train)

train_start = time.time()
NUM_CONCURRENT_STEPS = 4
stale_count = [0]*NUM_CONCURRENT_STEPS
stale_limit = 10

def is_stale(proc_id):
   for i in range(NUM_CONCURRENT_STEPS):
       if stale_count[proc_id] > (stale_count[i] + stale_limit):
            return True
   return False

with tf.Session() as sess:
    sess.run(init)
    def train_function(proc_id):
        print proc_id
        t_loss = []
        max_val_acc  = 0.0
        seed = (proc_id+1)*1729
        batches = batch_iter(X_train, y_train, batch_size, no_epochs, seed)
        i = 0
        for batch in batches:
            stale_flag = True
            while not is_stale(proc_id):
              if stale_flag:
               stale_flag = False
               i += 1
               _, step, train_loss, train_acc = sess.run([train_op, global_step, loss, accuracy], feed_dict={input_x: batch[0], input_y: batch[1]})
               current_step = tf.train.global_step(sess, global_step)
               stale_count[proc_id] += 1
               time_str = datetime.datetime.now().isoformat()
               if i % (data_size/(batch_size*8)) == 0:
                print ("{}: {}: step {}, mini batch loss {:g}, mini batch acc {:g}".format(proc_id,time_str, step, train_loss, train_acc))
               if i % (data_size/batch_size) == 0:
                t_loss.append(train_loss) 
                pred_labels = sess.run([pred_label], feed_dict={input_x: val_data, input_y: val_labels})
                val_acc = get_accuracy(pred_labels[0], val_labels)
                print ('Accuracy on val data:', val_acc)
                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    saver.save(sess, '/home/bharatp/MLLD/Ass2/models/log_reg_const_lr_0.01_4proc_10epo_stale_10_' + str(proc_id) + '_' +str(val_acc))
                    best_model_path = '/home/bharatp/MLLD/Ass2/models/log_reg_const_lr_0.01_4proc_10epo_stale_10_' + str(proc_id) + '_' + str(val_acc)
              else:
               break   
        np.save('training_loss_const_lr_0.01_4proc_10epochs_ssp10_asgd_'+str(proc_id) +'.npy',np.array(t_loss))    

    
    train_threads = []
    for _ in range(NUM_CONCURRENT_STEPS):
        train_threads.append(threading.Thread(target=train_function, args=(_,)))

    for t in train_threads:
       t.start()
    for t in train_threads:
       t.join()
    
    #pool = mp.Pool(processes=NUM_CONCURRENT_STEPS)
    #for x in range(NUM_CONCURRENT_STEPS):
    #     pool.apply_async(train_function, args=(x,))
    #output = [p.get() for p in results]
    #print(output)


train_end = time.time()
print ('Train time:', train_end - train_start)

'''
test_start = time.time()
best_model_path = '/home/bharatp/MLLD/Ass2/models/log_reg_const_lr_0.01_6epo_3_0.775192285803'
print ('Testing starts.')
print(best_model_path)
init = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
pred_labels_train = []
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, best_model_path)
    pred_labels = sess.run([pred_label], feed_dict={input_x: test_data, input_y: test_labels})
    test_acc = get_accuracy(pred_labels[0], test_labels)
    test_end = time.time()
    for i in range(len(train_data)):
        pred_labels = sess.run([pred_label], feed_dict={input_x: np.reshape(train_data[i],(1,10000)), input_y: np.reshape(train_labels[i], (1,50))})
        pred_labels_train.append(pred_labels[0][0])

    train_acc = get_accuracy(pred_labels_train, train_labels)

#test_end = time.time()
#print ('Train accuracy:', train_acc)
print ('Test accuracy:',test_acc)
#np.save('training_loss_const_lr_0.01_40epochs.npy',np.array(t_loss))

#print ('Train time:', train_end - train_start)
print ('Testing time:', test_end - test_start)
'''

