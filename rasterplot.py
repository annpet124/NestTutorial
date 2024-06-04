import nest
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pickle

# Define variables
exc_weights = [40.0, 60.0, 80.0]
inh_weights = [-20.0, -25.0, -30.0]
cluster_strengths = [1, 5, 10]  # Define the range of cluster strengths

class EIClusteredNetwork:
    def __init__(self, network_params, synapse_params, thalamo_params):
        self._network_params = network_params
        self._synapse_params = synapse_params
        self._thalamo_params = thalamo_params
        self._populations = []
        self._recording_devices = []

    def setup_network(self):
        self.create_populations()
        self.create_recording_devices()
        self.connect_network()
        self.adjust_input_currents()
        self.adjust_simulation_time()

    def create_populations(self):
        N_E = self._network_params["N_ex"]
        N_I = self._network_params["N_in"]
        N_thalamo = self._thalamo_params["N_thalamo"]

        # Ensure that the subset size is proportional to the total population size
        subset_size_E = int(0.5 * N_E)
        subset_size_I = int(0.5 * N_I)

        # Select a random subset of excitatory and inhibitory neurons
        subset_E = random.sample(range(N_E), min(subset_size_E, N_E))
        subset_I = random.sample(range(N_I), min(subset_size_I, N_I))

        self._populations.append(nest.Create("iaf_psc_alpha", len(subset_E)))
        self._populations.append(nest.Create("iaf_psc_alpha", len(subset_I)))
        self._populations.append(nest.Create("iaf_psc_alpha", N_thalamo))

    def connect_network(self):
        cluster_strength = self._network_params.get("cluster_strength", 1.0)
        syn_params_exc = {"weight": self._synapse_params["exc_weight"] * cluster_strength * 0.01}  # Adjust excitatory weight scaling
        syn_params_inh = {"weight": self._synapse_params["inh_weight"] * cluster_strength}
        syn_params_thalamo = {"weight": self._synapse_params["thalamo_weight"] * cluster_strength}

        nest.Connect(self._populations[0], self._populations[1], "all_to_all", syn_spec=syn_params_exc)
        nest.Connect(self._populations[1], self._populations[0], "all_to_all", syn_spec=syn_params_inh)
        nest.Connect(self._populations[2], self._populations[0], "all_to_all", syn_spec=syn_params_thalamo)

    def adjust_input_currents(self):
        nest.SetStatus(self._populations[0], {"I_e": 2500.0})  # Increase excitatory input current even further
        nest.SetStatus(self._populations[1], {"I_e": 1700.0})
        nest.SetStatus(self._populations[2], {"I_e": 1000.0})

    def adjust_simulation_time(self):
        self._network_params['simtime'] = 2000.0

    def simulate(self):
        nest.Simulate(self._network_params["simtime"])
        time.sleep(1)

    def create_recording_devices(self):
        for population in self._populations:
            recorder = nest.Create("spike_recorder")
            nest.Connect(population, recorder)
            self._recording_devices.append(recorder)

    def get_recordings(self):
        recordings = []
        for recorder in self._recording_devices:
            try:
                events = nest.GetStatus(recorder, "events")[0]
                spiketimes = np.array([events["times"], events["senders"]])
                recordings.append(spiketimes)
            except Exception as e:
                if "DictError" in str(e):
                    recordings.append(None)
                else:
                    recordings.append(None)
        return recordings

    def get_simulation(self, cluster_strength, PathSpikes=None):
        self._network_params['cluster_strength'] = cluster_strength
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


def plot_spike_raster(spike_times_list, sim_duration, cluster_strength):
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r']  # Define colors for different populations
    labels = ["Excitatory", "Inhibitory", "Thalamic"]

    for i, spike_times in enumerate(spike_times_list):
        if spike_times is not None:
            population_index = i % 3  # 0: Excitatory, 1: Inhibitory, 2: Thalamic
            color = colors[population_index]
            label = labels[population_index]

            # Extract spikes for each neuron
            neuron_ids = np.unique(spike_times[1])
            for neuron_id in neuron_ids:
                neuron_spikes = spike_times[0][spike_times[1] == neuron_id]
                plt.plot(neuron_spikes, np.ones_like(neuron_spikes) * neuron_id, '.', color=color)

    # Adding legend for population types
    patches = [plt.Line2D([0], [0], color=color, label=label, markersize=10, marker='o') for color, label in zip(colors, labels)]
    plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.title(f'Spike Raster Plot for Cluster Strength = {cluster_strength}')
    plt.xlim(0, sim_duration)
    plt.ylim(0, 1500)  # Adjust y-axis limit for better visualization
    plt.tight_layout()
    plt.show()




# Define network parameters
network_params = {
    "N_ex": 500,
    "N_in": 250,
    "N_thalamo": 100,
    "simtime": 2000.0
}

# Define synaptic parameters
synapse_params = {
    "exc_weight": 35.0,
    "inh_weight": -15.0,
    "thalamo_weight": 2.0
}

# Define thalamo parameters
thalamo_params = {
    "N_thalamo": 100
}

# Create network instance
ei_clustered_network = EIClusteredNetwork(network_params, synapse_params, thalamo_params)

# Define cluster strengths
max_cluster_strength = 20
cluster_strengths = list(range(1, max_cluster_strength + 1))

# Run simulation and save spike times for different cluster strengths
firing_rates = []
for cluster_strength in cluster_strengths:
    simulation_results = ei_clustered_network.get_simulation(cluster_strength, PathSpikes=f"spike_times_cluster_strength_{cluster_strength}.pkl")
    print(f"Simulation for cluster strength {cluster_strength} completed.")

    # Extract spike times and neuron indices for each population
    spike_times = simulation_results['spiketimes']

    # Calculate firing rate
    firing_rate = np.sum([len(spikes) for spikes in spike_times[0]]) / (network_params["N_ex"] * network_params["simtime"] / 1000)
    firing_rates.append(firing_rate)

    # Plot spike raster for the current cluster strength
    plot_spike_raster(spike_times, network_params["simtime"], cluster_strength)

# Plot the cluster plot
def plot_cluster_plot(cluster_strengths, firing_rates):
    plt.figure(figsize=(8, 6))
    plt.plot(cluster_strengths, firing_rates, marker='o', linestyle='-')
    plt.xlabel('Cluster Strength')
    plt.ylabel('Firing Rate (Hz)')
    plt.title('Cluster Plot')
    plt.grid(True)
    plt.show()

plot_cluster_plot(cluster_strengths, firing_rates)

print("All simulations completed.")










