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
#tf.keras.backend.set_floatx('float64')

class QRL(object):
    def __init__(self, qaoa_depth, num_neurons, gamma, learning_rate, data_path):
        self.create_data_folders(data_path)

        # Log model progress with Tensorboard
        self.create_tf_loggers()

        self.qaoa_depth = qaoa_depth
        self.state_size = 2*self.qaoa_depth + 2

        # Model Parameters
        self.learning_rate = learning_rate
        self.num_neurons = num_neurons
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
        self.expectation_layer = tfq.layers.Expectation() 

        self.reset_memory()
        self.build_net()
    
    def create_data_folders(self,data_path):
        self.data_path = data_path
        self.model_path = self.data_path + 'models/' #path where machine learning models are stored
        self.training_data_path = self.data_path + 'training/' #path to where all graphs/graph descriptions are stored
        self.validation_data_path = self.data_path + 'validation/' 
        self.testing_data_path = self.data_path + 'testing/' 

        folders = [self.data_path, 
                self.model_path, 
                self.training_data_path, 
                self.validation_data_path, 
                self.testing_data_path]
        
        for ind,folder in enumerate(folders):
            if not os.path.exists(folder):
                os.mkdir(folder)

        print("Saving models to: " + self.model_path)
        print("Saving training data to: " + self.training_data_path)
        print("Saving validation data to: " + self.validation_data_path)
        print("Saving testing data to: " + self.testing_data_path)

    def create_tf_loggers(self):
        self.log_dir = self.data_path + 'logs/'
        self.train_log_dir = self.log_dir + 'train/'
        self.validation_log_dir = self.log_dir + 'validation/'
        self.testing_log_dir = self.log_dir + 'testing/'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.validation_summary_writer = tf.summary.create_file_writer(self.validation_log_dir)
        self.testing_summary_writer = tf.summary.create_file_writer(self.testing_log_dir)
        print("Saving logs to: " + self.log_dir)

    def reset_memory(self):
        self.mu_memory = []
        self.sigma_memory = []
        self.state_memory = []
        self.action_memory = []
        self.probs_memory =  []
        self.reward_memory = []

    def build_net(self):
        inputs = tf.keras.layers.Input(shape = (self.state_size))
        dense1 = tf.keras.layers.Dense(self.num_neurons,activation = tf.keras.activations.elu)(inputs)
        do1 = tf.keras.layers.Dropout(rate=0.2)(dense1)
        dense2 = tf.keras.layers.Dense(self.num_neurons,activation = tf.keras.activations.elu)(do1)
        do2 = tf.keras.layers.Dropout(rate=0.2)(dense2)
        dense3 = tf.keras.layers.Dense(self.num_neurons,activation = tf.keras.activations.elu)(do2)
        do3 = tf.keras.layers.Dropout(rate=0.2)(dense3)
        mu = tf.keras.layers.Dense(2*self.qaoa_depth)(do3)
        sigma = tf.keras.layers.Dense(2*self.qaoa_depth,activation = tf.keras.activations.elu)(do3)

        self.model = tf.keras.models.Model(inputs=inputs, outputs=[mu,sigma])
        print(self.model.summary())
    def call(self, x, num_guesses):
        circuits = x[0]
        cost_hams = x[1]
        symbols = x[2]
        init_mu = x[3]
        init_sigma = x[4]

        num_samples = len(circuits)

        actions, probs, qaoa_vals = self.sample_action_space(circuits, cost_hams, symbols, init_mu,init_sigma)
        # state --> [qaoa_params, qaoa_val , improvement]
        state = tf.concat([actions,qaoa_vals], axis = 1)
        state = tf.concat([state,tf.zeros((num_samples, 1))], axis = 1) 
        best_qaoa_vals = qaoa_vals
        
        for guess in range(num_guesses):
            #Sample action according to current policy
            mu, sigma = self.model(state)

            sigma = tf.clip_by_value(sigma,clip_value_min = 0.1, clip_value_max = 5)
            actions,probs,qaoa_vals = self.sample_action_space(circuits, cost_hams, symbols, mu,sigma)
            rewards = tf.math.maximum(tf.zeros((num_samples,1)), qaoa_vals-best_qaoa_vals)
            best_qaoa_vals = tf.math.maximum(best_qaoa_vals,qaoa_vals)

            self.store_transition(state,actions,rewards,probs,mu, sigma)

            next_state = tf.concat([actions,qaoa_vals], axis = 1) 
            next_state = tf.concat([next_state,rewards], axis = 1) 

            state = next_state

        return state

    def sample_action_space(self, circuits, cost_hams, symbols, mu, sigma):
        norm_dists = tfp.distributions.Normal(loc = mu, scale = sigma, validate_args=True)
        actions = tf.squeeze(norm_dists.sample(1), axis=0)
        qaoa_vals = tf.cast(self.expectation_layer(circuits, 
                                                    symbol_names=symbols, 
                                                    symbol_values=actions, 
                                                    operators = cost_hams), tf.float32)

        probs = norm_dists.prob(actions) #probability of last action
        probs = tf.math.reduce_prod(probs, axis=-1, keepdims=True)

        return actions,probs,qaoa_vals

    def store_transition(self,state, action, reward, prob, mu, sigma):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.probs_memory.append(prob)
        self.reward_memory.append(reward)
        self.mu_memory.append(mu)
        self.sigma_memory.append(sigma)

    def save_memory(self, path, graphs, episode):
        state_memory_folder = path + "state_memory/"
        if not os.path.exists(state_memory_folder):
            os.mkdir(state_memory_folder)
        state_memory_folder += str(episode) + "/"
        if not os.path.exists(state_memory_folder):
            os.mkdir(state_memory_folder)

        mu_memory_folder = path + "mu_memory/"
        if not os.path.exists(mu_memory_folder):
            os.mkdir(mu_memory_folder)
        mu_memory_folder += str(episode) + "/"
        if not os.path.exists(mu_memory_folder):
            os.mkdir(mu_memory_folder)

        sigma_memory_folder = path + "sigma_memory/"
        if not os.path.exists(sigma_memory_folder):
            os.mkdir(sigma_memory_folder)
        sigma_memory_folder += str(episode) + "/"
        if not os.path.exists(sigma_memory_folder):
            os.mkdir(sigma_memory_folder)

        for ind, graph in enumerate(graphs): 
            state_memory_slice = tf.slice(self.state_memory, [0,ind, 0], [-1, 1, -1])
            mu_memory_slice = tf.slice(self.mu_memory, [0,ind, 0], [-1, 1, -1])
            sigma_memory_slice = tf.slice(self.sigma_memory, [0,ind, 0], [-1, 1, -1])

            state_memory_filename = state_memory_folder + graph.id 
            mu_memory_filename = mu_memory_folder + graph.id 
            sigma_memory_filename = sigma_memory_folder + graph.id 

            np.save(state_memory_filename, state_memory_slice)
            np.save(mu_memory_filename, mu_memory_slice)
            np.save(sigma_memory_filename, sigma_memory_slice)

    def learn(self, tape, update):
        self.state_memory = tf.stack(self.state_memory, axis =2)
        self.action_memory = tf.stack(self.action_memory, axis =2)
        self.probs_memory = tf.stack(self.probs_memory, axis =1)
        self.reward_memory = tf.stack(self.reward_memory, axis =1)
        
        num_guesses = np.shape(self.state_memory)[2]
        num_samples = np.shape(self.state_memory)[0]

        episode_rewards = tf.zeros((num_samples, 1))
        discount = 1
        for guess in range(num_guesses):
            episode_rewards += discount*self.reward_memory[:,guess]
            discount = discount*self.gamma

        #episode_rewards = (episode_rewards - tf.reduce_mean(episode_rewards,keepdims=True))/tf.math.reduce_std(episode_rewards,keepdims=True) #Normalize Rewards
        neg_log_probs = -tf.math.log(tf.math.reduce_prod(self.probs_memory, axis=1, keepdims=False)) #neg log prob of each trajectory
        loss = tf.reduce_mean(neg_log_probs*episode_rewards)

        if(update == True):
            # Update Model Parameters
            actor_grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(actor_grads, self.model.trainable_variables))

        self.reset_memory()

        return loss

    def load_checkpoint(self, filename):
        self.model.load_weights(self.model_path+ filename)

    def save_checkpoint(self, filename):
        self.model.save_weights(self.model_path+ filename)
        print("Model saved to: " + self.model_path+ filename)

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
        
        data = [circuits, cost_hams, symbols, initial_mu, intitial_std, graphs]

        return data   

    def generate_QAOA_Circuit_Batch(self, graphs):
        '''
        generate graph QAOA circuits
        '''
        # Generate circuits from graphs
        num_graphs = len(graphs)

        # Convert graphs to qaoa circuits
        qaoa_circuits = []
        cost_hams = []

        # Create progress bar
        loop = tqdm(total = num_graphs, position = 0)

        for graph_index, graph in enumerate(graphs):
            # Update progress bar
            loop.set_description("Generating QAOA circuits from graphs ".format(graph_index))
            loop.update(1)

            qaoa_circuit, qaoa_parameters, cost_ham = self.qaoa_circuit_from_graph(graph)
            
            qaoa_circuits.append(qaoa_circuit)
            cost_hams.append(cost_ham)

        loop.close()

        return qaoa_circuits, qaoa_parameters, cost_hams

    def qaoa_circuit_from_graph(self,graph):
        cirq_qubits = cirq.GridQubit.rect(1,graph.number_of_nodes())
        qaoa_circuit = cirq.Circuit()

        # Create Mixer ground state
        for qubit_index,qubit in enumerate(cirq_qubits):
            qaoa_circuit.append([cirq.H(qubit)], strategy=InsertStrategy.EARLIEST)

        qaoa_parameters = []

        for step in range(1,self.qaoa_depth+1):
            gamma_i = sympy.Symbol("gamma{}_p={}".format(step,self.qaoa_depth))
            beta_i = sympy.Symbol("beta{}_p={}".format(step,self.qaoa_depth))
            qaoa_parameters.append(gamma_i)
            qaoa_parameters.append(beta_i)

            #Apply ising hamiltonian
            for current_edge in graph.edges():
                qubit1 = cirq_qubits[current_edge[0]]
                qubit2 = cirq_qubits[current_edge[1]]

                qaoa_circuit.append([cirq.CNOT(qubit1,qubit2)], strategy=InsertStrategy.EARLIEST)
                qaoa_circuit.append([cirq.Rz(-1*gamma_i)(qubit2)], strategy=InsertStrategy.EARLIEST)
                qaoa_circuit.append([cirq.CNOT(qubit1,qubit2)], strategy=InsertStrategy.EARLIEST)
                
            #Apply Driver Hamiltonian 
            for current_node in graph.nodes():
                qubit = cirq_qubits[current_node]
                qaoa_circuit.append([cirq.Rx(beta_i)(qubit)], strategy=InsertStrategy.EARLIEST)

        #Generate Cost Hamiltionian
        cost_ham = None

        for current_edge in graph.edges():
            qubit1 = cirq_qubits[current_edge[0]]
            qubit2 = cirq_qubits[current_edge[1]]

            if cost_ham is None:
                cost_ham = -1/2*cirq.Z(qubit1)*cirq.Z(qubit2) + 1/2
            else:
                cost_ham += -1/2*cirq.Z(qubit1)*cirq.Z(qubit2) + 1/2
                
        return qaoa_circuit, qaoa_parameters, cost_ham

    def save_graphs(self,graphs,path):
        ########################################################################
        '''

        RANDOMLY GENERATES BATCH OF ERDOS-RENYI GRAPHS WITH 10-20 NODES

        INPUTS:
            batch_size - number of graphs to generate

        OUTPUTS:
            graph_ids - list of uniquely generated IDs for each graph
                        ** graphs are stored as dataframes in: .../*SELF.DATA_PATH*'/graphs/*GRAPH ID*.csv **
                        ** graph descriptions are stored as a dataframe in: .../*SELF.DATA_PATH*'/graphs/graph_desc.csv **
                        "GRAPH_ID": Uniquely generated ID for graph
                        "GW_CUT": Maximum cut predicted by the Geomanns-Williamson algorithm 
                        "GW_PROJECTIONS": Number of Geomanns-Williamson algorithm projections to obtain a cut value > 0.878*Maximum Cut
                        "MAXCUT": Maximum cut computed by brute force
                        "GW_APPROX_RAT": Geomanns-Williamson cut over Maxcut
                        "NUM_NODES": Number of nodes 
                        "NUM_EDGES": Number of edges 
                        "RAT_EDGETONODES": ratio of number of edges to the number of nodes 
                        "DENSITY": Density of graph
                        "RAT_GWTOEDGES": ratio of Geomanns-Williamson cut to the number of edges
                        "RAT_GWTONODES": ratio of Geomanns-Williamson cut to the number of nodes
                        "SPECTRAL_GAP": difference between first and second largest eigenvalue
                        "LARGESTEIGVAL": ratio of the largest eigenvalue
                        "SECONDLARGESTEIGVAL": ratio of the second largest eigenvalue
                        "THIRDLARGESTEIGVAL": ratio of the third largest eigenvalue
                        "FOURTHLARGESTEIGVAL": ratio of the fourth largest eigenvalue
                        "FIFTHLARGESTEIGVAL": ratio of the fifth largest eigenvalue
                        "SIXTHLARGESTEIGVAL": ratio of the sixth largest eigenvalue
                        "SEVTHLARGESTEIGVAL":ratio of the seventh largest eigenvalue
                        "SMALLESTEIGVAL": ratio of the smallest eigenvalue
                        "MAX_ECC": largest eccentricity of graph (diameter) 
        
        '''
        ########################################################################
        graphs_folder_path = path + 'graphs/' #path to where graph themselves are stored

        if not os.path.exists(graphs_folder_path):
            os.mkdir(graphs_folder_path)

        # create database for storing all graph properties
        graph_description = pd.DataFrame(columns = ["GRAPH_ID",
                                                    "GW_CUT",
                                                    "GW_PROJECTIONS", 
                                                    "MAXCUT",
                                                    "GW_APPROX_RAT",
                                                    "NUM_NODES",
                                                    "NUM_EDGES",
                                                    "RAT_EDGETONODES",
                                                    "DENSITY",
                                                    "RAT_GWTOEDGES",
                                                    "RAT_GWTONODES",
                                                    "SPECTRAL_GAP",
                                                    "LARGESTEIGVAL",
                                                    "SECONDLARGESTEIGVAL",
                                                    "THIRDLARGESTEIGVAL",
                                                    "FOURTHLARGESTEIGVAL",
                                                    "FIFTHLARGESTEIGVAL",
                                                    "SIXTHLARGESTEIGVAL",
                                                    "SEVTHLARGESTEIGVAL",
                                                    "SMALLESTEIGVAL",
                                                    "MAX_ECC",
                                                    "FAMILY"])
        # Calculate properties of each graph
        # Create progress bar
        loop = tqdm(total = len(graphs), position = 0)

        for i,current_graph in enumerate(graphs):
            # Update progress bar
            loop.set_description("Calculating properties of graph dataset ".format(i))
            loop.update(1)
            
            current_graph_GWcut, current_graph_numGWprojections = self.get_GW_cut(current_graph)
            current_graph_MaxCut = self.get_MaxCut(current_graph)
            current_graph_GW_ratio = current_graph_GWcut/current_graph_MaxCut

            current_graph_num_nodes = current_graph.number_of_nodes()
            current_graph_num_edges = current_graph.number_of_edges()
            current_graph_density = nx.density(current_graph) 

            current_graph_ratio_edgestonodes = current_graph_num_edges/current_graph_num_nodes
            current_graph_ratio_GWtoedges = current_graph_GWcut/current_graph_num_edges
            current_graph_ratio_GWtonodes = current_graph_GWcut/current_graph_num_nodes

            e = nx.laplacian_spectrum(current_graph)
            e.sort()

            try:
                current_graph_largesteigenval = e[len(e)-1]
            except:
                current_graph_largesteigenval = None

            try:
                current_graph_secondlargesteigenval = e[len(e)-2]
            except:
                current_graph_secondlargesteigenval = None

            try:
                current_graph_thirdlargesteigenval = e[len(e)-3]
            except:
                current_graph_thirdlargesteigenval = None

            try:
                current_graph_fourthlargesteigenval = e[len(e)-4]
            except:
                current_graph_fourthlargesteigenval = None

            try:
                current_graph_fifthlargesteigenval = e[len(e)-5]
            except:
                current_graph_fifthlargesteigenval = None

            try:
                current_graph_sixthlargesteigenval = e[len(e)-6]
            except:
                current_graph_sixthlargesteigenval = None

            try:
                current_graph_seventhlargesteigenval = e[len(e)-7]
            except:
                current_graph_seventhlargesteigenval = None

            try:
                current_graph_smallesteigenval = e[1] #Smallest NONTRIVIAL eigenvalue
            except:
                current_graph_smallesteigenval = None

            current_graph_spectral_gap = abs(current_graph_largesteigenval - current_graph_secondlargesteigenval)

            eccs = list(nx.eccentricity(current_graph).values())
            eccs.sort()
            current_graph_max_eccentricity = eccs[len(e)-1]
            current_graph_family = current_graph.family

            # generate random ID for the current graph
            current_graph_id = current_graph.id
            current_graph_file_path = graphs_folder_path + str(current_graph_id) + '.csv'
            
            # convert current graph into a pandas dataframe 
            graph_df = nx.to_pandas_edgelist(current_graph, dtype=int)

            # save graph to a csv file
            graph_df.to_csv(current_graph_file_path, index=True)  

            # append graph PROPERTIES to graph_description database
            graph_description = graph_description.append({"GRAPH_ID":current_graph_id,
                                                            "GW_CUT":current_graph_GWcut,
                                                            "GW_PROJECTIONS": current_graph_numGWprojections, 
                                                            "MAXCUT":current_graph_MaxCut,
                                                            "GW_APPROX_RAT":current_graph_GW_ratio,
                                                            "NUM_NODES":current_graph_num_nodes,
                                                            "NUM_EDGES":current_graph_num_edges,
                                                            "RAT_EDGETONODES":current_graph_ratio_edgestonodes,
                                                            "DENSITY":current_graph_density,
                                                            "RAT_GWTOEDGES":current_graph_ratio_GWtoedges,
                                                            "RAT_GWTONODES":current_graph_ratio_GWtonodes,
                                                            "SPECTRAL_GAP":current_graph_spectral_gap,
                                                            "LARGESTEIGVAL":current_graph_largesteigenval,
                                                            "SECONDLARGESTEIGVAL":current_graph_secondlargesteigenval,
                                                            "THIRDLARGESTEIGVAL":current_graph_thirdlargesteigenval,
                                                            "FOURTHLARGESTEIGVAL":current_graph_fourthlargesteigenval,
                                                            "FIFTHLARGESTEIGVAL":current_graph_fifthlargesteigenval,
                                                            "SIXTHLARGESTEIGVAL":current_graph_sixthlargesteigenval,
                                                            "SEVTHLARGESTEIGVAL":current_graph_seventhlargesteigenval,
                                                            "SMALLESTEIGVAL":current_graph_smallesteigenval,
                                                            "MAX_ECC":current_graph_max_eccentricity,
                                                            "FAMILY":current_graph_family}, ignore_index = True) 
        loop.close()

        graph_description.set_index("GRAPH_ID",inplace=True)

        self.save_graph_desc_to_csv(path,graph_description)

    def save_graph_desc_to_csv(self, path, graph_description):
        #save graph PROPERTIES database to a csv file
        #with open(self.desc_path, 'a') as f:
        desc_path = path + "desc.csv"
        graph_description.to_csv(desc_path, index=True)#, mode='w', index=False, header=f.tell()==0)

    def get_GW_cut(self,graph):
        ########################################################################
        # RETURNS AVERAGE GEOMANNS WILLIAMSON CUT FOR A GIVEN GRAPH
        ########################################################################

        G = graph
        N = len(G.nodes())

        # Allocate weights to the edges.
        for (i,j) in G.edges():
            G[i][j]['weight']=1.0

        maxcut = pic.Problem()

        # Add the symmetric matrix variable.
        X=maxcut.add_variable('X',(N,N),'symmetric')

        # Retrieve the Laplacian of the graph.
        LL = 1/4.*nx.laplacian_matrix(G).todense()
        L=pic.new_param('L',LL)

        # Constrain X to have ones on the diagonal.
        maxcut.add_constraint(pic.tools.diag_vect(X)==1)

        # Constrain X to be positive semidefinite.
        maxcut.add_constraint(X>>0)

        # Set the objective.
        maxcut.set_objective('max',L|X)

        # Solve the problem.
        maxcut.solve(verbose = 0,solver='cvxopt')

        # Use a fixed RNG seed so the result is reproducable.
        cvx.setseed(1)

        # Perform a Cholesky factorization.
        V=X.value
        cvxopt.lapack.potrf(V)
        for i in range(N):
            for j in range(i+1,N):
                V[i,j]=0

        # Do up to 100 projections. Stop if we are within a factor 0.878 of the SDP
        # optimal value.
        count=0
        obj_sdp=maxcut.obj_value()
        obj=0
        while (obj < 0.878*obj_sdp):
            r=cvx.normal(N,1)
            x=cvx.matrix(np.sign(V*r))
            o=(x.T*L*x).value
            if o > obj:
                x_cut=x
                obj=o
            count+=1
        x=x_cut

        return obj,count

    def get_MaxCut(self,graph):
        sub_lists = []
        for i in range(0, len(graph.nodes())+1):
            temp = [list(x) for x in combinations(graph.nodes(), i)]
            sub_lists.extend(temp)

        # Calculate the cut_size for all possible cuts
        cut_size = []
        for sub_list in sub_lists:
            cut_size.append(nx.algorithms.cuts.cut_size(graph,sub_list))

        maxcut = np.max(cut_size)

        return maxcut

    def fetch_graphs_from_folder(self,path):
        ########################################################################
        # Returns graphs given IDs
        '''
        INPUTS:
            ids - list containing IDs of graphs to be fetched
        OUTPUTS:
            all_graphs - list of graphs specified by ids
        '''
        ########################################################################
        all_graphs = []

        graph_directory = path + "graphs/"
        graph_desc_path = path + "desc.csv"
        all_graph_files = os.listdir(graph_directory)

        #shutil.copyfile(target, self.desc_path)
        graph_desc = self.load_graph_database(path)

        for index,current_file in enumerate(all_graph_files):
            current_graph_df = pd.read_csv(filepath_or_buffer = graph_directory + current_file)  
            current_graph = nx.from_pandas_edgelist(current_graph_df, 'source', 'target', ['weight'])
            current_graph.id = current_file[:-4]
            current_graph.family = graph_desc.loc[current_graph.id]["FAMILY"]
            all_graphs.append(current_graph)

        return all_graphs 
        
    def fetch_graphs_from_ID(self,ids):
        ########################################################################
        # Returns graphs given IDs
        '''
        INPUTS:
            ids - list containing IDs of graphs to be fetched
        OUTPUTS:
            all_graphs - list of graphs specified by ids
        '''
        ########################################################################
        all_graphs = []
        gw_maxcuts = [0]*len(all_graphs)

        for index,current_id in enumerate(ids):
           current_graph_file_path = self.graphs_path + str(current_id) + '.csv'
           current_graph_df = pd.read_csv(filepath_or_buffer = current_graph_file_path)           
           current_graph = nx.from_pandas_edgelist(current_graph_df, 'source', 'target', ['weight'])
           all_graphs.append(current_graph)

        return all_graphs

    def load_graph_database(self, path):
        desc_path = path + "desc.csv"
        df = pd.read_csv(filepath_or_buffer = desc_path)
        df.set_index("GRAPH_ID",inplace=True)

        return df    