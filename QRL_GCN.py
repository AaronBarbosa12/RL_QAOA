import os 
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq
import tensorflow_probability as tfp
import networkx as nx 

import cirq
from cirq.circuits import InsertStrategy

import pandas as pd
from tqdm import tqdm

import picos as pic
from picos.tools import diag_vect

import cvxopt as cvx
import cvxopt.lapack

import numpy as np
from itertools import combinations
from QRL import *
from GCN_Layer import *
#tf.keras.backend.set_floatx('float64')

class QRL_GCN(QRL):
    def __init__(self, qaoa_depth, num_neurons, gamma, learning_rate, data_path):
        self.create_data_folders(data_path)

        # Log model progress with Tensorboard
        self.create_tf_loggers()

        self.qaoa_depth = qaoa_depth
        self.state_size = 2*self.qaoa_depth + 2 + 30

        # Model Parameters
        self.learning_rate = learning_rate
        self.num_neurons = num_neurons
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.expectation_layer = tfq.layers.Expectation() 

        self.reset_memory()
        self.build_net()

    def build_net(self):        
        state_input = tf.keras.layers.Input(shape = (self.state_size))
        #concatenate_layer = tf.keras.layers.concatenate([state_input, gcn_layer])
        dense1 = tf.keras.layers.Dense(self.num_neurons,activation = tf.keras.activations.elu)(state_input)
        do1 = tf.keras.layers.Dropout(rate=0.2)(dense1)
        dense2 = tf.keras.layers.Dense(self.num_neurons,activation = tf.keras.activations.elu)(do1)
        do2 = tf.keras.layers.Dropout(rate=0.2)(dense2)
        dense3 = tf.keras.layers.Dense(self.num_neurons,activation = tf.keras.activations.elu)(do2)
        do3 = tf.keras.layers.Dropout(rate=0.2)(dense3)
        mu = tf.keras.layers.Dense(2*self.qaoa_depth)(do3)
        sigma = tf.keras.layers.Dense(2*self.qaoa_depth,activation = tf.keras.activations.elu)(do3)

        self.model = tf.keras.models.Model(inputs=[state_input], outputs=[mu,sigma])
        self.gcn_layer = GCN_Layer(num_kernels = 5, num_neurons = 30, qaoa_depth = self.qaoa_depth)

    def generate_model_data(self,graphs,path):
        num_samples = len(graphs)
        self.save_graphs(graphs,path)

        qaoa_circuits, qaoa_parameters, cost_hams = self.generate_QAOA_Circuit_Batch(graphs)

        circuits = tfq.convert_to_tensor(qaoa_circuits)
        cost_hams = tfq.convert_to_tensor([cost_hams])
        cost_hams = tf.transpose(cost_hams)
        symbols = tf.convert_to_tensor([str(element) for element in qaoa_parameters])

        # Generate Random Initial Parameters
        initial_mu = tf.convert_to_tensor(np.zeros(shape=(num_samples,self.qaoa_depth*2)).astype(np.float32))
        intitial_std = tf.convert_to_tensor(np.ones(shape=(num_samples,self.qaoa_depth*2)).astype(np.float32))
        graph_laplacians = [np.asarray(nx.normalized_laplacian_matrix(graph).todense()) for graph in graphs]

        data = [circuits, cost_hams, symbols, initial_mu, intitial_std, graphs, graph_laplacians]

        return data   

    def get_GCN_OUT(self,laplacians):
        gcn_out = []
        loop = tqdm(total = len(laplacians), position = 0)
        for i,M in enumerate(laplacians):
            # Update progress bar
            loop.set_description("Processing Graph Laplacians: ".format(i))
            loop.update(1)

            laplacians_tensor = tf.expand_dims(tf.convert_to_tensor(M), -1)
            laplacians_tensor = tf.expand_dims(tf.convert_to_tensor(laplacians_tensor), 0)
            y = tf.squeeze(self.gcn_layer(laplacians_tensor), axis= 0 )
            gcn_out.append(y)
            
        loop.close()

        gcn_out = tf.convert_to_tensor(gcn_out)

        return gcn_out

    def call(self, x, num_guesses):
        circuits = x[0]
        cost_hams = x[1]
        symbols = x[2]
        init_mu = x[3]
        init_sigma = x[4]
        laplacians = x[6]

        num_samples = len(circuits)

        actions, probs, qaoa_vals = self.sample_action_space(circuits, cost_hams, symbols, init_mu,init_sigma)
        # state --> [qaoa_params, qaoa_val , improvement]
   
        state = tf.concat([actions,qaoa_vals], axis = 1)
        state = tf.concat([state,tf.zeros((num_samples, 1))], axis = 1) 

        gcn_outs = self.get_GCN_OUT(laplacians)
        state = tf.concat([state,gcn_outs], axis = 1) 

        best_qaoa_vals = qaoa_vals
        
        for guess in range(num_guesses):
            #Sample action according to current policy
            mu, sigma = self.model(state)

            sigma = tf.clip_by_value(sigma,clip_value_min = 0.1, clip_value_max = 5)
            actions,probs,qaoa_vals = self.sample_action_space(circuits, cost_hams, symbols, mu,sigma)
            rewards = tf.math.maximum(tf.zeros((num_samples,1)), qaoa_vals-best_qaoa_vals)
            best_qaoa_vals = tf.math.maximum(best_qaoa_vals,qaoa_vals)

            self.store_transition(state,actions,rewards,probs,mu,sigma)

            next_state = tf.concat([actions,qaoa_vals], axis = 1) 
            next_state = tf.concat([next_state,rewards], axis = 1) 
            next_state = tf.concat([next_state,gcn_outs], axis = 1) 
            state = next_state

        return state
