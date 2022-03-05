"""
Implementation of a spatial attack.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from itertools import product, repeat
import random

import tensorflow as tf
import numpy as np

from pgd_attack import LinfPGDAttack

class SpatialAttack:
  def __init__(self, model, config):
    self.model = model
    self.grid_store = []

    if config.use_linf:
        self.linf_attack = LinfPGDAttack(model, config)
    else:
        self.linf_attack = None

    self.use_spatial = config.use_spatial
    if config.use_spatial:
        self.method = config.spatial_method
        self.limits = config.spatial_limits

        if self.method == 'grid':
            self.granularity = config.grid_granularity
        elif self.method == 'random':
            self.random_tries = config.random_tries

  def perturb(self, x_nat, y, sess):
      if not self.use_spatial:
          t = np.zeros([len(x_nat), 3])
          if self.linf_attack:
              x = self.linf_attack.perturb(x_nat, y, sess, trans=t)
          else:
              x = x_nat
          return x, t
      if self.method == 'grid':
          return self.perturb_grid(x_nat, y, sess, -1)
      else: # random
          return self.perturb_grid(x_nat, y, sess, self.random_tries)

  def perturb_grid(self, x_nat, y, sess, random_tries=-1):
    n = len(x_nat)
    if random_tries > 0:
        # subsampling this list from the grid is a bad idea, instead we
        # will randomize each example from the full continuous range
        grid = [(42, 42, 42) for _ in range(random_tries)] # dummy list
    else: # exhaustive grid
        grid = product(*list(np.linspace(-l, l, num=g)
                             for l, g in zip(self.limits, self.granularity)))

    worst_x = np.copy(x_nat)
    worst_t = np.zeros([n, 3])
    max_xent = np.zeros(n)
    all_correct = np.ones(n).astype(bool)

    for tx, ty, r in grid:
        if random_tries > 0:
            # randomize each example separately
            t = np.stack((np.random.uniform(-l, l, n) for l in self.limits),
                         axis=1)
        else:
            t = np.stack(repeat([tx, ty, r], n))

        if self.linf_attack:
            x = self.linf_attack.perturb(x_nat, y, sess, trans=t)
        else:
            x = x_nat

        curr_dict = {self.model.x_input: x,
                     self.model.y_input: y,
                     self.model.is_training: False,
                     self.model.transform: t}

        cur_xent, cur_correct = sess.run([self.model.y_xent,
                                          self.model.correct_prediction], 
                                         feed_dict = curr_dict) # shape (bsize,)
        cur_xent = np.asarray(cur_xent)
        cur_correct = np.asarray(cur_correct)

        # Select indices to update: we choose the misclassified transformation 
        # of maximum xent (or just highest xent if everything else if correct).
        idx = (cur_xent > max_xent) & (cur_correct == all_correct)
        idx = idx | (cur_correct < all_correct)
        max_xent = np.maximum(cur_xent, max_xent)
        all_correct = cur_correct & all_correct

        idx = np.expand_dims(idx, axis=-1) # shape (bsize, 1)
        worst_t = np.where(idx, t, worst_t) # shape (bsize, 3)

        idx = np.expand_dims(idx, axis=-1) 
        idx = np.expand_dims(idx, axis=-1) # shape (bsize, 1, 1, 1)
        worst_x = np.where(idx, x, worst_x,) # shape (bsize, 32, 32, 3)


    return worst_x, worst_t

if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model()
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.train.Saver()

  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))
