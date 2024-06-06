import nest
import numpy as np
import pickle
import matplotlib.pyplot as plt

class EIClusteredNetwork:
    def __init__(self, network_params, synapse_params, thalamo_params):
        self._network_params = network_params
        self._synapse_params = synapse_params
        self._thalamo_params = thalamo_params
        self._populations = []
        self._recording_devices = []
        self.clusters = []

    def setup_network(self):
        print("Network Configuration:")
        print(f"Excitatory neurons: {self._network_params['N_ex']}")
        print(f"Inhibitory neurons: {self._network_params['N_in']}")
        print(f"Thalamic neurons: {self._thalamo_params['N_thalamo']}")
        print(f"Simtime: {self._network_params['simtime']} ms")
        print(f"Excitatory synaptic weight: {self._synapse_params['exc_weight']} mV")
        print(f"Inhibitory synaptic weight: {self._synapse_params['inh_weight']} mV")
        print(f"Thalamic synaptic weight: {self._synapse_params['thalamo_weight']} mV")
        print("Connecting Network:")
        self.create_populations()
        self.create_clusters()
        self.connect_clusters()
        self.create_recording_devices()

        # Adjust input currents
        for population in self._populations:
            nest.SetStatus(population, {"I_e": 500.0})

    def create_populations(self):
        N_E = self._network_params["N_ex"]
        N_I = self._network_params["N_in"]
        N_thalamo = self._thalamo_params["N_thalamo"]
        self._populations.append(nest.Create("iaf_psc_alpha", N_E))
        self._populations.append(nest.Create("iaf_psc_alpha", N_I))
        self._populations.append(nest.Create("iaf_psc_alpha", N_thalamo))

    def create_clusters(self):
        N_E = self._network_params["N_ex"]
        N_I = self._network_params["N_in"]
        N_thalamo = self._thalamo_params["N_thalamo"]

        # Proportionate number of clusters
        total_neurons = N_E + N_I + N_thalamo
        proportionate_clusters_exc = max(1, int((N_E / total_neurons) * self._network_params["total_clusters"]))
        proportionate_clusters_inh = max(1, int((N_I / total_neurons) * self._network_params["total_clusters"]))
        proportionate_clusters_thalamo = max(1, int((N_thalamo / total_neurons) * self._network_params["total_clusters"]))

        neurons_per_cluster = self._network_params["neurons_per_cluster"]

        # Ensure we don't exceed the number of neurons in each population
        actual_num_clusters_exc = min(proportionate_clusters_exc, N_E // neurons_per_cluster)
        actual_num_clusters_inh = min(proportionate_clusters_inh, N_I // neurons_per_cluster)
        actual_num_clusters_thalamo = min(proportionate_clusters_thalamo, N_thalamo // neurons_per_cluster)

        # Create clusters for excitatory population
        exc_neurons = self._populations[0].tolist()
        self.clusters.append([exc_neurons[i * neurons_per_cluster:(i + 1) * neurons_per_cluster] for i in range(actual_num_clusters_exc)])

        # Create clusters for inhibitory population
        inh_neurons = self._populations[1].tolist()
        self.clusters.append([inh_neurons[i * neurons_per_cluster:(i + 1) * neurons_per_cluster] for i in range(actual_num_clusters_inh)])

        # Create clusters for thalamic population
        thalamo_neurons = self._populations[2].tolist()
        self.clusters.append([thalamo_neurons[i * neurons_per_cluster:(i + 1) * neurons_per_cluster] for i in range(actual_num_clusters_thalamo)])

        # Detailed printout of clusters
        print("Created clusters:")
        print(f"Excitatory clusters: {len(self.clusters[0])} with sizes {[len(cluster) for cluster in self.clusters[0]]}")
        print(f"Inhibitory clusters: {len(self.clusters[1])} with sizes {[len(cluster) for cluster in self.clusters[1]]}")
        print(f"Thalamic clusters: {len(self.clusters[2])} with sizes {[len(cluster) for cluster in self.clusters[2]]}")

    def connect_clusters(self):
        syn_params_exc = {"weight": self._synapse_params["exc_weight"], "delay": 1.5}
        syn_params_inh = {"weight": self._synapse_params["inh_weight"], "delay": 1.0}
        syn_params_thalamo = {"weight": self._synapse_params["thalamo_weight"], "delay": 2.0}

        for population, syn_params in zip(self.clusters, [syn_params_exc, syn_params_inh, syn_params_thalamo]):
            for cluster in population:
                cluster_neuron_ids = [neuron for neuron in cluster]
                for neuron in cluster_neuron_ids:
                    nest.Connect([neuron], cluster_neuron_ids, syn_spec=syn_params)

    def create_recording_devices(self):
        for population in self._populations:
            recorder = nest.Create("spike_recorder", params={"record_to": "memory"})
            nest.Connect(population, recorder)
            self._recording_devices.append(recorder)

    def simulate(self):
        nest.Simulate(self._network_params["simtime"])

    def get_recordings(self):
        recordings = []
        for recorder in self._recording_devices:
            events = nest.GetStatus(recorder, "events")[0]
            spiketimes = np.array([events["times"], events["senders"]])
            recordings.append(spiketimes)
        return recordings

    def get_simulation(self, PathSpikes=None):
        self.setup_network()
        self.simulate()
        spiketimes = self.get_recordings()

        if PathSpikes is not None:
            with open(PathSpikes, "wb") as outfile:
                pickle.dump(spiketimes, outfile)

        return {
            "_params": self._network_params,
            "spiketimes": spiketimes,
        }

# Define network parameters
network_params = {
    "N_ex": 200,
    "N_in": 100,
    "N_thalamo": 50,
    "simtime": 100.0,
    "total_clusters": 15,
    "neurons_per_cluster": 10
}

# Define synaptic parameters
synapse_params = {
    "exc_weight": 35.0,
    "inh_weight": -15.0,
    "thalamo_weight": 5.0
}

# Define thalamo parameters
thalamo_params = {
    "N_thalamo": 50
}

# Create network instance
ei_clustered_network = EIClusteredNetwork(network_params, synapse_params, thalamo_params)

# Run simulation and save spike times
simulation_results = ei_clustered_network.get_simulation(PathSpikes="spike_times.pkl")
print(simulation_results)

# Extract spike times and neuron indices for each population
spike_times = simulation_results['spiketimes']

# Plot spike times
plt.figure(figsize=(12, 8))

# Plot excitatory neurons
for spike in spike_times[0].T:
    plt.scatter(spike[0], spike[1], color='blue', s=10)
plt.scatter([], [], color='blue', label='Excitatory')

# Plot inhibitory neurons
for spike in spike_times[1].T:
    plt.scatter(spike[0], spike[1], color='red', s=10)
plt.scatter([], [], color='red', label='Inhibitory')

# Plot thalamic neurons
for spike in spike_times[2].T:
    plt.scatter(spike[0], spike[1], color='green', s=10)
plt.scatter([], [], color='green', label='Thalamic')

plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.title('Cluster Plot of Spike Times')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()



























































































































































