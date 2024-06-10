import nest
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from helper import calculate_RBN_weights, rheobase_current, raster_plot

class ClusteredNetwork:
    def __init__(self, sim_dict, net_dict, stim_dict):

        # merge dictionaries of simulation, network and stimulus parameters
        self._params = {**sim_dict, **net_dict, **stim_dict}
        # list of neuron population groups [E_pops, I_pops]
        self._populations = []
        self._recording_devices = []
        self._currentsources = []
        self._model_build_pipeline = [
            self.setup_nest,
            self.create_populations,
            self.create_stimulation,
            self.create_recording_devices,
            self.connect,
        ]

        if self._params["clustering"] == "weight":
            jep = self._params["rep"]
            jip = 1.0 + (jep - 1) * self._params["rj"]
            self._params["jplus"] = np.array([[jep, jip], [jip, jip]])
        elif self._params["clustering"] == "probabilities":
            pep = self._params["rep"]
            pip = 1.0 + (pep - 1) ** self._params["rj"]
            self._params["pplus"] = np.array([[pep, pip], [pip, pip]])
        else:
            raise ValueError("Clustering type not recognized")

    def setup_nest(self):
        """Initializes the NEST kernel.

        Reset the NEST kernel and pass parameters to it.
        Updates randseed of parameters to the actual
        used one if none is supplied.
        """

        nest.ResetKernel()
        nest.set_verbosity("M_WARNING")
        nest.local_num_threads = self._params.get("n_vp", 4)
        nest.resolution = self._params.get("dt")
        self._params["randseed"] = self._params.get("randseed")
        nest.rng_seed = self._params.get("randseed")

    def create_populations(self):
        """Create all neuron populations.

        n_clusters excitatory and inhibitory neuron populations
        with the parameters of the network are created.
        """

        # make sure number of clusters and units are compatible
        if self._params["N_E"] % self._params["n_clusters"] != 0:
            raise ValueError("N_E must be a multiple of Q")
        if self._params["N_I"] % self._params["n_clusters"] != 0:
            raise ValueError("N_E must be a multiple of Q")
        if self._params["neuron_type"] != "iaf_psc_exp":
            raise ValueError("Model only implemented for iaf_psc_exp neuron model")

        if self._params["I_th_E"] is None:
            I_xE = self._params["I_xE"]  # I_xE is the feed forward excitatory input in pA
        else:
            I_xE = self._params["I_th_E"] * helper.rheobase_current(
                self._params["tau_E"], self._params["E_L"], self._params["V_th_E"], self._params["C_m"]
            )

        if self._params["I_th_I"] is None:
            I_xI = self._params["I_xI"]
        else:
            I_xI = self._params["I_th_I"] * helper.rheobase_current(
                self._params["tau_I"], self._params["E_L"], self._params["V_th_I"], self._params["C_m"]
            )

        E_neuron_params = {
            "E_L": self._params["E_L"],
            "C_m": self._params["C_m"],
            "tau_m": self._params["tau_E"],
            "t_ref": self._params["t_ref"],
            "V_th": self._params["V_th_E"],
            "V_reset": self._params["V_r"],
            "I_e": I_xE
            if self._params["delta_I_xE"] == 0
            else I_xE * nest.random.uniform(1 - self._params["delta_I_xE"] / 2, 1 + self._params["delta_I_xE"] / 2),
            "tau_syn_ex": self._params["tau_syn_ex"],
            "tau_syn_in": self._params["tau_syn_in"],
            "V_m": self._params["V_m"]
            if not self._params["V_m"] == "rand"
            else self._params["V_th_E"] - 20 * nest.random.lognormal(0, 1),
        }
        I_neuron_params = {
            "E_L": self._params["E_L"],
            "C_m": self._params["C_m"],
            "tau_m": self._params["tau_I"],
            "t_ref": self._params["t_ref"],
            "V_th": self._params["V_th_I"],
            "V_reset": self._params["V_r"],
            "I_e": I_xI
            if self._params["delta_I_xE"] == 0
            else I_xI * nest.random.uniform(1 - self._params["delta_I_xE"] / 2, 1 + self._params["delta_I_xE"] / 2),
            "tau_syn_ex": self._params["tau_syn_ex"],
            "tau_syn_in": self._params["tau_syn_in"],
        }


        def plot_cluster_network(self):
            """Plot the network clusters."""
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))

            # Extract parameters
            n_clusters = self._params["n_clusters"]
            baseline_conn_prob = self._params["baseline_conn_prob"]

            # Compute cluster sizes
            cluster_sizes = np.zeros(n_clusters)
            for i in range(n_clusters):
                cluster_sizes[i] = int(self._params["N_E"] / n_clusters)

                # Plot connections between clusters
                for i in range(n_clusters):
                    for j in range(n_clusters):
                        conn_prob = baseline_conn_prob[0, 0] if i == j else baseline_conn_prob[0, 1]
                        if conn_prob > 0:
                            ax.plot([i, j], [cluster_sizes[i], cluster_sizes[j]], 'k-', linewidth=conn_prob * 10)

                # Set axis labels and title
                ax.set_xlabel('Cluster ID')
                ax.set_ylabel('Number of Neurons')
                ax.set_title('Network Clusters')

                # Show plot
                plt.show()

        class EIClusteredNetworkSimulation:
            def __init__(self, params):
                self._params = params
                self._populations = [[], []]
                self._currentsources = None
                self._recording_devices = None
                self._model_build_pipeline = []

            def create_populations(self):
                """Create the populations of excitatory and inhibitory neurons."""
                N_E = self._params["N_E"]
                N_I = self._params["N_I"]
                n_clusters = self._params["n_clusters"]
                neuron_params = {
                    "C_m": self._params["C_m"],
                    "tau_m": [self._params["tau_E"], self._params["tau_I"]],
                    "t_ref": self._params["t_ref"],
                    "V_th": [self._params["V_th_E"], self._params["V_th_I"]],
                    "V_reset": self._params["V_r"],
                    "tau_syn_ex": self._params["tau_syn_ex"],
                    "tau_syn_in": self._params["tau_syn_in"],
                    "E_L": self._params["E_L"],
                }
                for i, population in enumerate(self._populations):
                    for _ in range(n_clusters):
                        population.append(nest.Create(self._params["neuron_type"], N_E / n_clusters, neuron_params))

            def connect_synapses(self):
                """Connect synapses between neuron populations."""
                # Calculate synaptic weights
                js = calculate_RBN_weights(self._params)

                # Inhibitory to inhibitory neuron connections
                j_ii = js[1, 1] / np.sqrt(N)
                nest.CopyModel("static_synapse", "II", {"weight": j_ii, "delay": self._params["delay"]})

                if self._params["fixed_indegree"]:
                    K_II_plus = int(p_plus[1, 1] * self._params["N_I"] / self._params["n_clusters"])
                    K_II_minus = int(p_minus[1, 1] * self._params["N_I"] / self._params["n_clusters"])
                    conn_params_II_plus = {
                        "rule": "fixed_indegree",
                        "indegree": K_II_plus,
                        "allow_autapses": False,
                        "allow_multapses": True,
                    }
                    conn_params_II_minus = {
                        "rule": "fixed_indegree",
                        "indegree": K_II_minus,
                        "allow_autapses": False,
                        "allow_multapses": True,
                    }

                else:
                    conn_params_II_plus = {
                        "rule": "pairwise_bernoulli",
                        "p": p_plus[1, 1],
                        "allow_autapses": False,
                        "allow_multapses": True,
                    }
                    conn_params_II_minus = {
                        "rule": "pairwise_bernoulli",
                        "p": p_minus[1, 1],
                        "allow_autapses": False,
                        "allow_multapses": True,
                    }
                for i, pre in enumerate(self._populations[1]):
                    for j, post in enumerate(self._populations[1]):
                        if i == j:
                            # same cluster
                            for n in range(iterations[1, 1]):
                                nest.Connect(pre, post, conn_params_II_plus, "II")
                        else:
                            nest.Connect(pre, post, conn_params_II_minus, "II")

            # Other methods remain unchanged

            def connect_prob(self):
                """Connect the clusters with a probability EI-cluster scheme."""
                N = self._params["N_E"] + self._params["N_I"]

                if self._params["n_clusters"] > 1:
                    p = np.ones([N, N]) * self._params["pminus"]
                    size_E = self._params["N_E"] // self._params["n_clusters"]
                    size_I = self._params["N_I"] // self._params["n_clusters"]
                    for i in range(self._params["n_clusters"]):
                        for j in range(self._params["n_clusters"]):
                            if i == j:
                                p[size_E * i: size_E * (i + 1), size_E * j: size_E * (j + 1)] = self._params["pplus"][
                                    0, 0]
                                p[size_E * i: size_E * (i + 1),
                                self._params["N_E"] + size_I * j: self._params["N_E"] + size_I * (j + 1)] = \
                                self._params["pplus"][0, 1]
                                p[self._params["N_E"] + size_I * i: self._params["N_E"] + size_I * (i + 1),
                                size_E * j: size_E * (j + 1)] = self._params["pplus"][1, 0]
                                p[self._params["N_E"] + size_I * i: self._params["N_E"] + size_I * (i + 1),
                                self._params["N_E"] + size_I * j: self._params["N_E"] + size_I * (j + 1)] = \
                                self._params["pplus"][1, 1]
                else:
                    p = np.ones([N, N]) * self._params["pplus"]

                p *= self._params["p_scale"]

                E_neurons = [neuron for population in self._populations[0] for neuron in population]
                I_neurons = [neuron for population in self._populations[1] for neuron in population]
                neurons = E_neurons + I_neurons

                for i, neuron in enumerate(neurons):
                    conn_probs = p[i, :]
                    conn_probs[conn_probs > 1] = 1  # Ensure probabilities are between 0 and 1

                    post_neurons = np.random.choice(neurons, size=(conn_probs > 0).sum(), replace=False, p=conn_probs)
                    weights = self.calculate_RBN_weights()[
                        i, np.array([neurons.index(post_neuron) for post_neuron in post_neurons])]
                    delays = nest.random.uniform(min=self._params["delay_range"][0], max=self._params["delay_range"][1])
                    nest.Connect([neuron], list(post_neurons), syn_spec={"weight": weights, "delay": delays})

            def connect_weight(self):
                """Connect the clusters with a weight EI-cluster scheme."""
                N = self._params["N_E"] + self._params["N_I"]

                js = self.calculate_RBN_weights()

                if self._params["n_clusters"] > 1:
                    j = np.ones([N, N]) * self._params["jminus"]
                    size_E = self._params["N_E"] // self._params["n_clusters"]
                    size_I = self._params["N_I"] // self._params["n_clusters"]
                    for i in range(self._params["n_clusters"]):
                        for j in range(self._params["n_clusters"]):
                            if i == j:
                                j[size_E * i: size_E * (i + 1), size_E * j: size_E * (j + 1)] = self._params["jplus"][
                                    0, 0]
                                j[size_E * i: size_E * (i + 1),
                                self._params["N_E"] + size_I * j: self._params["N_E"] + size_I * (j + 1)] = \
                                self._params["jplus"][0, 1]
                                j[self._params["N_E"] + size_I * i: self._params["N_E"] + size_I * (i + 1),
                                size_E * j: size_E * (j + 1)] = self._params["jplus"][1, 0]
                                j[self._params["N_E"] + size_I * i: self._params["N_E"] + size_I * (i + 1),
                                self._params["N_E"] + size_I * j: self._params["N_E"] + size_I * (j + 1)] = \
                                self._params["jplus"][1, 1]
                else:
                    j = np.ones([N, N]) * self._params["jplus"]

                j *= self._params["j_scale"]

                E_neurons = [neuron for population in self._populations[0] for neuron in population]
                I_neurons = [neuron for population in self._populations[1] for neuron in population]
                neurons = E_neurons + I_neurons

                for i, neuron in enumerate(neurons):
                    conn_probs = self.calculate_RBN_weights()[i, :]
                    conn_probs[conn_probs > 1] = 1  # Ensure probabilities are between 0 and 1

                    post_neurons = np.random.choice(neurons, size=(conn_probs > 0).sum(), replace=False, p=conn_probs)
                    weights = j[i, np.array([neurons.index(post_neuron) for post_neuron in post_neurons])]
                    delays = nest.random.uniform(min=self._params["delay_range"][0], max=self._params["delay_range"][1])
                    nest.Connect([neuron], list(post_neurons), syn_spec={"weight": weights, "delay": delays})

            def rheobase_current(self, tau, E_L, V_th, C_m):
                """Compute the rheobase current based on LIF parameters."""
                return (V_th - E_L) / tau * C_m

            def calculate_RBN_weights(self):
                """Calculate random balanced network weights."""
                N = self._params["N_E"] + self._params["N_I"]
                K = self._params["K"]
                tau = np.array([self._params["tau_E"], self._params["tau_I"]])
                C_m = self._params["C_m"]
                V_th = np.array([self._params["V_th_E"], self._params["V_th_I"]])
                E_L = self._params["E_L"]

                js = np.zeros((N, N))
                for i in range(N):
                    if i < self._params["N_E"]:
                        for j in range(N):
                            if j < self._params["N_E"]:
                                js[i, j] = self.rheobase_current(tau[0], E_L, V_th[0], C_m) / np.sqrt(K)
                            else:
                                js[i, j] = self.rheobase_current(tau[0], E_L, V_th[0], C_m) / np.sqrt(K) * self._params[
                                    "rj"]
                    else:
                        for j in range(N):
                            if j < self._params["N_E"]:
                                js[i, j] = self.rheobase_current(tau[1], E_L, V_th[1], C_m) / np.sqrt(K) * self._params[
                                    "rj"]
                            else:
                                js[i, j] = self.rheobase_current(tau[1], E_L, V_th[1], C_m) / np.sqrt(K) * self._params[
                                    "rj"] ** 2

                return js

            def create_stimulation(self):
                """Create a current source and connect it to clusters."""
                if self._params["stim_clusters"] is not None:
                    stim_amp = self._params["stim_amp"]  # amplitude of the stimulation current in pA
                    stim_starts = self._params["stim_starts"]  # list of stimulation start times
                    stim_ends = self._params["stim_ends"]  # list of stimulation end times
                    amplitude_values = []
                    amplitude_times = []
                    for start, end in zip(stim_starts, stim_ends):
                        amplitude_times.append(start + self._params["warmup"])
                        amplitude_values.append(stim_amp)
                        amplitude_times.append(end + self._params["warmup"])
                        amplitude_values.append(0.0)
                    self._currentsources = [nest.Create("step_current_generator")]
                    for stim_cluster in self._params["stim_clusters"]:
                        nest.Connect(self._currentsources[0], self._populations[0][stim_cluster])
                    nest.SetStatus(
                        self._currentsources[0],
                        {
                            "amplitude_times": amplitude_times,
                            "amplitude_values": amplitude_values,
                        },
                    )

            def create_recording_devices(self):
                """Create recording devices based on simulation parameters."""
                self._recording_devices = [nest.Create("spike_recorder")]
                self._recording_devices[0].record_to = "memory"

                all_units = self._populations[0][0]
                for E_pop in self._populations[0][1:]:
                    all_units += E_pop
                for I_pop in self._populations[1]:
                    all_units += I_pop
                nest.Connect(all_units, self._recording_devices[0], "all_to_all")  # Spikerecorder

            def set_model_build_pipeline(self, pipeline):
                """Set _model_build_pipeline

                Parameters
                ----------
                pipeline: list
                    ordered list of functions executed to build the network model
                """
                self._model_build_pipeline = pipeline

            def setup_network(self):
                """Setup network in NEST

                Initializes NEST and creates
                the network in NEST, ready to be simulated.
                Functions saved in _model_build_pipeline are executed.
                """
                for func in self._model_build_pipeline:
                    func()

            def simulate(self):
                """Simulates network for a period of warmup+simtime"""
                nest.Simulate(self._params["warmup"] + self._params["simtime"])

            def get_recordings(self):
                """Extract spikes from Spikerecorder

                Extract spikes form the Spikerecorder connected
                to all populations created in create_populations.
                Cuts the warmup period away and sets time relative to end of warmup.
                Ids 1:N_E correspond to excitatory neurons,
                N_E+1:N_E+N_I correspond to inhibitory neurons.

                Returns
                -------
                spiketimes: ndarray
                    2D array [2xN_Spikes]
                    of spiketimes with spiketimes in row 0 and neuron IDs in row 1.
                """
                events = nest.GetStatus(self._recording_devices[0], "events")[0]
                # convert them to the format accepted by spiketools
                spiketimes = np.append(events["times"][None, :], events["senders"][None, :], axis=0)
                spiketimes[1] -= 1
                # remove the pre warmup spikes
                spiketimes = spiketimes[:, spiketimes[0] >= self._params["warmup"]]
                spiketimes[0] -= self._params["warmup"]
                return spiketimes

            def get_parameter(self):
                """Get all parameters used to create the network.
                Returns
                -------
                dict
                    Dictionary with all parameters of the network and the simulation.
                """
                return self._params

            def create_and_simulate(self):
                """Create and simulate the EI-clustered network.

                Returns
                -------
                spiketimes: ndarray
                    2D array [2xN_Spikes]
                    of spiketimes with spiketimes in row 0 and neuron IDs in row 1.
                """
                self.setup_network()
                self.simulate()
                return self.get_recordings()

            def get_firing_rates(self, spiketimes=None):
                """Calculates the average firing rates of
                all excitatory and inhibitory neurons.

                Calculates the firing rates of all excitatory neurons
                and the firing rates of all inhibitory neurons
                created by self.create_populations.
                If spiketimes are not supplied, they get extracted.

                Parameters
                ----------
                spiketimes: ndarray
                    2D array [2xN_Spikes] of spiketimes
                    with spiketimes in row 0 and neuron IDs in row 1.

                Returns
                -------
                tuple[float, float]
                    average firing rates of excitatory (0)
                    and inhibitory (1) neurons (spikes/s)
                """
                if spiketimes is None:
                    spiketimes = self.get_recordings()
                e_count = spiketimes[:, spiketimes[1] < self._params["N_E"]].shape[1]
                i_count = spiketimes[:, spiketimes[1] >= self._params["N_E"]].shape[1]
                e_rate = e_count / float(self._params["N_E"]) / float(self._params["simtime"]) * 1000.0
                i_rate = i_count / float(self._params["N_I"]) / float(self._params["simtime"]) * 1000.0
                return e_rate, i_rate

            def set_I_x(self, I_XE, I_XI):
                """Set DC currents for excitatory and inhibitory neurons
                Adds DC currents for the excitatory and inhibitory neurons.
                The DC currents are added to the currents already
                present in the populations.

                Parameters
                ----------
                I_XE: float
                    extra DC current for excitatory neurons [pA]
                I_XI: float
                    extra DC current for inhibitory neurons [pA]
                """
                for E_pop in self._populations[0]:
                    I_e_loc = nest.GetStatus(E_pop, "I_e")
                    nest.SetStatus(E_pop, {"I_e": I_e_loc + I_XE})
                for I_pop in self._populations[1]:
                    I_e_loc = nest.GetStatus(I_pop, "I_e")
                    nest.SetStatus(I_pop, {"I_e": I_e_loc + I_XI})

            def get_simulation(self, PathSpikes=None):
                """Create network, simulate and return results

                Creates the EI-clustered network and simulates it with
                the parameters supplied in the object creation.
                Returns a dictionary with firing rates,
                timing information (dict) and parameters (dict).
                If PathSpikes is supplied the spikes get saved to a pickle file.

                Parameters
                ----------
                PathSpikes: str (optional)
                    Path of file for spiketimes, if None, no file is saved

                Returns
                -------
                dict
                 Dictionary with firing rates,
                 spiketimes (ndarray) and parameters (dict)
                """

                self.setup_network()
                self.simulate()
                spiketimes = self.get_recordings()
                e_rate, i_rate = self.get_firing_rates(spiketimes)

                if PathSpikes is not None:
                    with open(PathSpikes, "wb") as outfile:
                        pickle.dump(spiketimes, outfile)
                return {
                    "e_rate": e_rate,
                    "i_rate": i_rate,
                    "_params": self.get_parameter(),
                    "spiketimes": spiketimes,
                }

            def run_simulation(self):
                """Run the NEST simulation."""
                nest.Simulate(self._sim_params["warmup"])
                nest.Simulate(self._sim_params["simtime"])

            def get_results(self):
                """Get the results from the simulation.

                Returns
                -------
                dict
                    Dictionary containing the results of the simulation.
                """
                spike_times = nest.GetStatus(self._recorders[0], "events")[0]["times"]
                spike_senders = nest.GetStatus(self._recorders[0], "events")[0]["senders"]

                return {"spike_times": spike_times, "spike_senders": spike_senders}


