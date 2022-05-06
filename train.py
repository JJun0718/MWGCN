from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from sklearn import metrics
from loads import *
from models import GCN, MLP
import random
import os
import sys
from data.metrics import f1_np, get_metrics

from config import CONFIG
cfg = CONFIG()
dataset = cfg.dataset
multi_label = 1

# Set random seed
# seed = 19
seed = 1
# seed = random.randint(1, 200)
print("seed:", seed)
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

flags = tf.app.flags
FLAGS = flags.FLAGS
# 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
# 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 500, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 500, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.00000001,
                   'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    FLAGS.dataset)
print(adj)
# print(adj[0], adj[1])
features = sp.identity(features.shape[0])  # featureless

print("adj.shape:", adj.shape)
print("features.shape:", features.shape)

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    # 'support': [tf.sparse_placeholder(tf.int64) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32),
    'result': tf.placeholder(tf.int32)
}
# print("------------placeholders")

# Create model
print("features[2][1]:", features[2][1])
model = model_func(placeholders, input_dim=features[2][1], multi_label=multi_label, logging=True)

# Initialize session
session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=session_conf)

# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(
        features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy,
                     model.layers[0].embedding], feed_dict=feed_dict)

    # Validation
    cost, acc, pred, labels, duration = evaluate(
        features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(
              outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    micro_f1, macro_f1 = f1_np(labels, pred)
    print('Test micro_f1, macro_f1', micro_f1, macro_f1)

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, pred, labels, test_duration = evaluate(
    features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

test_pred = []
test_labels = []
print(len(test_mask))
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(labels[i])

test_labels = np.array(test_labels)
test_pred = np.array(test_pred)

import pandas as pd
t_true = pd.DataFrame(test_labels)
t_true.to_excel("results/测试集标签.xlsx", index=False)
t_pred = pd.DataFrame(test_pred)
t_pred.to_excel("results/测试集预测值.xlsx", index=False)
print("标签文件已生成！")

micro_f1, macro_f1 = f1_np(test_labels, test_pred)
print('Test micro_f1, macro_f1', micro_f1, macro_f1)

A, P, R, F1, h, z, c, r, a= get_metrics(test_labels, test_pred)
# print("Accuracy：", A)
print("Precision：", P)
# print("Recall：", R)
print("F1：", F1)
print("汉明损失：",h)
# print("0-1 损失：",z)
# print("覆盖误差：",c)
# print("排名损失：",r)
# print("平均精度损失：",a)


# print("Test Precision, Recall and F1-Score...")
# print(metrics.classification_report(test_labels, test_pred, digits=4))
# print("Macro average Test Precision, Recall and F1-Score...")
# print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
# print("Micro average Test Precision, Recall and F1-Score...")
# print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))

# # doc and word embeddings
# print('embeddings:')
# word_embeddings = outs[3][train_size: adj.shape[0] - test_size]
# train_doc_embeddings = outs[3][:train_size]  # include val docs
# test_doc_embeddings = outs[3][adj.shape[0] - test_size:]
#
# print(len(word_embeddings), len(train_doc_embeddings),
#       len(test_doc_embeddings))
# print(word_embeddings)
#
# f = open('data/corpus/' + dataset + '_vocab.txt', 'r')
# words = f.readlines()
# f.close()
#
# vocab_size = len(words)
# word_vectors = []
# for i in range(vocab_size):
#     word = words[i].strip()
#     word_vector = word_embeddings[i]
#     word_vector_str = ' '.join([str(x) for x in word_vector])
#     word_vectors.append(word + ' ' + word_vector_str)
#
# word_embeddings_str = '\n'.join(word_vectors)
# f = open('data/' + dataset + '_word_vectors.txt', 'w')
# f.write(word_embeddings_str)
# f.close()
#
# doc_vectors = []
# doc_id = 0
# for i in range(train_size):
#     doc_vector = train_doc_embeddings[i]
#     doc_vector_str = ' '.join([str(x) for x in doc_vector])
#     doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
#     doc_id += 1
#
# for i in range(test_size):
#     doc_vector = test_doc_embeddings[i]
#     doc_vector_str = ' '.join([str(x) for x in doc_vector])
#     doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
#     doc_id += 1
#
# doc_embeddings_str = '\n'.join(doc_vectors)
# f = open('data/' + dataset + '_doc_vectors.txt', 'w')
# f.write(doc_embeddings_str)
# f.close()
