from QRL_GCN import *

import cirq
import numpy as np
import os 
from datetime import datetime
from tqdm import tqdm
from collections import deque# Ordered collection with ends

import networkx as nx 
import random

import uuid 
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#tf.keras.backend.set_floatx('float64')

def calculate_avg_approx_rat(data, maxcuts, qaoa_params_predicted):
    expectation_layer = tfq.layers.Expectation() 

    circuits = data[0]
    hams = data[1]
    symbols = data[2]
    graphs = data[5]

    approx_ratios = []
    num_fcalls = []

    num_params = len(symbols)
    num_iters = len(qaoa_params_predicted)

    loop = tqdm(total = num_iters, position = 0)
    #Run optimizer using guesses from the model
    for ind,guess in enumerate(qaoa_params_predicted):
        loop.set_description("Optimizing graphs ".format(ind))
        loop.update(1)

        circuit_in =  tf.slice(circuits, [ind], [1])
        ham_in =  tf.slice(hams, [ind, 0], [1, -1])
        guess_in = tf.slice(qaoa_params_predicted, [ind, 0], [1, -1])
        graph_id_in = graphs[ind].id

        maxcut_current_graph = maxcuts.loc[graph_id_in]
        maxcut_current_graph = tf.convert_to_tensor(maxcut_current_graph, dtype=tf.float32)
    
        #cost function
        def get_QAOA_cut(symbol_vals):
            symbol_vals = tf.convert_to_tensor(symbol_vals)
            symbol_vals = tf.reshape(symbol_vals, shape = (1,num_params))
            exps = expectation_layer(circuit_in, symbol_names=symbols, symbol_values=symbol_vals, operators = ham_in)
            cut_ratio = tf.math.divide(exps,maxcut_current_graph)
            cut_ratio = tf.reshape(cut_ratio,[])
            cut_ratio = np.asarray(cut_ratio)
            return -cut_ratio
        
        #run optimizer
        res = minimize(get_QAOA_cut, x0 = np.asarray(guess), method="Nelder-Mead",options= {'maxfev':1e6, 'fatol':1e-4})
        optim_val = res.fun
        optim_fcalls= res.nfev
    
        approx_ratios.append(optim_val)
        num_fcalls.append(optim_fcalls)

    loop.close()

    approx_ratios = np.asarray(approx_ratios)
    num_fcalls = np.asarray(num_fcalls)

    return -1*approx_ratios, num_fcalls

def generate_ER_graphs(num_samples, p_edge, size_range):
    '''
    Randomly generates Erdos-Renyi graphs with 
    size_range[0] --> size_range[1] nodes
    '''
    min_num_nodes = size_range[0]
    max_num_nodes = size_range[1]

    all_graphs = []
    
    for i in range(num_samples):

        # Randomly select number of nodes
        n = random.randint(min_num_nodes,max_num_nodes)

        # Generate Erdos-Renyi Graph
        current_graph = nx.erdos_renyi_graph(n, p_edge, seed=None, directed=False)

        # make sure graph is connected 
        while nx.is_connected(current_graph) == False:
            current_graph = nx.erdos_renyi_graph(n, p_edge, seed=None, directed=False)

        current_graph.family = "Erdos-Renyi_P_edge={}".format(p_edge)
        current_graph.id = str(uuid.uuid1())

        all_graphs.append(current_graph)
    return all_graphs

def generate_NReg_graphs(degree,num_samples, size_range):
    '''
    Randomly generates Erdos-Renyi graphs with 
    size_range[0] --> size_range[1] nodes
    '''
    min_num_nodes = size_range[0]
    max_num_nodes = size_range[1]

    all_graphs = []
    
    for i in range(num_samples):

        # Randomly select number of nodes
        n = random.randint(min_num_nodes,max_num_nodes)

        # Generate Erdos-Renyi Graph
        current_graph = nx.random_regular_graph(degree,n, seed=None)

        # make sure graph is connected 
        while nx.is_connected(current_graph) == False:
            current_graph = nx.random_regular_graph(degree,n, seed=None)

        current_graph.family = "{}Regular".format(degree) 
        current_graph.id = str(uuid.uuid1())       

        all_graphs.append(current_graph)
    
    return all_graphs

def build_test_set():
    all_graphs = []

    #barbell graphs
    for i in range(2,5):
        for j in range(2,6):
            current_graph = nx.generators.classic.barbell_graph(i,j)
            current_graph.family = "Barbell" 
            current_graph.id = str(uuid.uuid1()) 
            all_graphs.append(current_graph)

    #complete tree graphs
    for i in range(3,14):
        current_graph = nx.generators.classic.complete_graph(i)
        current_graph.family = "Complete" 
        current_graph.id = str(uuid.uuid1()) 
        all_graphs.append(current_graph)

    #circular ladder graphs
    for i in range(2,7):
        current_graph = nx.generators.classic.circular_ladder_graph(i)
        current_graph.family = "CirculantLadder" 
        current_graph.id = str(uuid.uuid1()) 
        all_graphs.append(current_graph)

    #cycle graphs
    for i in range(3,14):
        current_graph = nx.generators.classic.cycle_graph(i)
        current_graph.family = "Cycle" 
        current_graph.id = str(uuid.uuid1()) 
        all_graphs.append(current_graph)

    # ladder graphs
    for i in range(2,9):
        current_graph = nx.generators.classic.ladder_graph(i)
        current_graph.family = "Ladder" 
        current_graph.id = str(uuid.uuid1()) 
        all_graphs.append(current_graph)

    # Lolipop graphs
    for m in range(3,8):
        for n in range(3,7):
            current_graph = nx.generators.classic.lollipop_graph(m,n)
            current_graph.family = "Lolipop" 
            current_graph.id = str(uuid.uuid1()) 
            all_graphs.append(current_graph)

    # star graphs
    for m in range(3,14):
        current_graph = nx.generators.classic.star_graph(m)
        current_graph.family = "Star" 
        current_graph.id = str(uuid.uuid1()) 
        all_graphs.append(current_graph)

    # wheel graphs
    for m in range(3,14):
        current_graph = nx.generators.classic.wheel_graph(m)
        current_graph.family = "Wheel" 
        current_graph.id = str(uuid.uuid1()) 
        all_graphs.append(current_graph)

    #3Regular
    RegSizes = [4,6,8,10,12,14]
    for n,size in enumerate(RegSizes):
        Reg3Graphs = generate_NReg_graphs(3,10,size_range = [size,size])
        all_graphs.extend(Reg3Graphs)


    return all_graphs

#####################################
# Set Data Folders 
#####################################
data_path = "C://Users/barbo/Desktop/QAOA/"
time  = datetime.now() 

RL_data_path = data_path + time.strftime("GCN_RL_MODEL_%d_%m_%Y___%H_%M_%S/")
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

################################
# DEFINE MODEL PARAMETERS
################################
qaoa_depth = 3

training_size = 250
validation_size = 100
testing_size = 100 
num_neurons = 200
lr = 0.00001
gamma = 0.9

##############################
# CREATE QRL MODEL
#############################
print("Creating QRL Model...")
qrl = QRL_GCN(num_neurons = num_neurons, 
        qaoa_depth = qaoa_depth, 
        gamma = gamma, 
        learning_rate = lr, 
        data_path = RL_data_path)

###################################
# GENERATE TRAINING DATA
###################################

print("Generating training data...")
training_graphs = generate_ER_graphs(num_samples = training_size, size_range = [8,10], p_edge = 0.4)
training_graphs.extend(generate_ER_graphs(num_samples = training_size, size_range = [8,10], p_edge = 0.5))
training_graphs.extend(generate_ER_graphs(num_samples = training_size, size_range = [8,10], p_edge = 0.7))
training_graphs.extend(generate_NReg_graphs(degree = 4,num_samples = training_size, size_range = [6,12]))

print("Generating validation data...")
validation_graphs = generate_ER_graphs(num_samples = validation_size, size_range = [6,12], p_edge = 0.4)
validation_graphs.extend(generate_ER_graphs(num_samples = validation_size, size_range = [6,12], p_edge = 0.5))
validation_graphs.extend(generate_ER_graphs(num_samples = validation_size, size_range = [6,12], p_edge = 0.7))
validation_graphs.extend(generate_NReg_graphs(degree = 4,num_samples = validation_size, size_range = [6,12]))

testing_graphs = build_test_set()

training_data = qrl.generate_model_data(training_graphs, qrl.training_data_path)
validation_data = qrl.generate_model_data(validation_graphs, qrl.validation_data_path)
testing_data = qrl.generate_model_data(testing_graphs, qrl.testing_data_path)

df = qrl.load_graph_database(qrl.validation_data_path)
validation_maxcuts = df["MAXCUT"]
df = qrl.load_graph_database(qrl.testing_data_path)
testing_maxcuts = df["MAXCUT"]

###### TRAIN MODEL ###########
n_episodes = 2000
qaoa_guesses = 8

loop = tqdm(total = n_episodes, position = 0)
# Train over n_episodes.
for episode in range(n_episodes):
    # Update progress bar
    loop.set_description("Training QRL: ".format(episode))
    loop.update(1)

    #Training Set
    with tf.GradientTape(persistent=False) as tape:     
        training_pred = qrl.call(training_data,qaoa_guesses)
        training_loss = qrl.learn(tape,True)

    with qrl.train_summary_writer.as_default():
        tf.summary.scalar('actor_loss', training_loss, step = episode)

    # Validation Set
    with tf.GradientTape(persistent=False) as tape:     
        validation_pred = qrl.call(validation_data,qaoa_guesses)
        if episode % 10 == 0:
            qrl.save_memory(qrl.validation_data_path, validation_graphs, episode)
        validation_loss = qrl.learn(tape,False)

    validation_approx_ratios, validation_num_fcalls = calculate_avg_approx_rat(validation_data, validation_maxcuts ,validation_pred[:,0:2*qaoa_depth])
    if episode % 10 == 0:
        df = qrl.load_graph_database(qrl.validation_data_path)
        df["APPROX_RATIO_EPISODE_{}".format(episode)] = validation_approx_ratios
        df["NUM_CALLS_EPISODE_{}".format(episode)] = validation_num_fcalls
        qrl.save_graph_desc_to_csv(qrl.validation_data_path, df)
    

    with qrl.validation_summary_writer.as_default():
        tf.summary.scalar('actor_loss', validation_loss, step = episode)
        tf.summary.scalar('approx_ratio', tf.reduce_mean(validation_approx_ratios), step = episode)
        tf.summary.scalar('num_fcalls', tf.reduce_mean(validation_num_fcalls), step = episode)

    # Testing Set
    with tf.GradientTape(persistent=False) as tape:     
        testing_pred = qrl.call(testing_data,qaoa_guesses)
        if episode % 10 == 0:
            qrl.save_memory(qrl.testing_data_path, testing_graphs, episode)
        testing_loss = qrl.learn(tape,False)

    testing_approx_ratios, testing_num_fcalls = calculate_avg_approx_rat(testing_data, testing_maxcuts ,testing_pred[:,0:2*qaoa_depth])
    if episode % 10 == 0:
        df = qrl.load_graph_database(qrl.testing_data_path)
        df["APPROX_RATIO_EPISODE_{}".format(episode)] = testing_approx_ratios
        df["NUM_CALLS_EPISODE_{}".format(episode)] = testing_num_fcalls
        qrl.save_graph_desc_to_csv(qrl.testing_data_path, df)

    with qrl.testing_summary_writer.as_default():
        tf.summary.scalar('actor_loss', testing_loss, step = episode)
        tf.summary.scalar('approx_ratio', tf.reduce_mean(testing_approx_ratios), step = episode)
        tf.summary.scalar('num_fcalls', tf.reduce_mean(testing_num_fcalls), step = episode)

    if episode % 10 == 0:
        model_file= "QRL_{}.h5".format(episode)
        qrl.save_checkpoint(model_file)    
loop.close()
