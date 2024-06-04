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

    def setup_network(self):
        print("Network Configuration:")
        print(f"Excitatory neurons: {self._network_params['N_ex']}")
        print(f"Inhibitory neurons: {self._network_params['N_in']}")
        print(f"Simtime: {self._network_params['simtime']} ms")
        print(f"Excitatory synaptic weight: {self._synapse_params['exc_weight']} mV")
        print(f"Inhibitory synaptic weight: {self._synapse_params['inh_weight']} mV")
        print("Connecting Network:")
        self.create_populations()
        self.connect_network()
        self.create_recording_devices()

        # Adjust input currents
        nest.SetStatus(self._populations[0], {"I_e": 500.0})
        nest.SetStatus(self._populations[1], {"I_e": 500.0})
        nest.SetStatus(self._populations[2], {"I_e": 500.0})

    def create_populations(self):
        N_E = self._network_params["N_ex"]
        N_I = self._network_params["N_in"]
        N_thalamo = self._thalamo_params["N_thalamo"]
        self._populations.append(nest.Create("iaf_psc_alpha", N_E))
        self._populations.append(nest.Create("iaf_psc_alpha", N_I))
        self._populations.append(nest.Create("iaf_psc_alpha", N_thalamo))

    def connect_network(self):
        syn_params_exc = {"weight": self._synapse_params["exc_weight"]}
        syn_params_inh = {"weight": self._synapse_params["inh_weight"]}
        syn_params_thalamo = {"weight": self._synapse_params["thalamo_weight"]}

        nest.Connect(self._populations[0], self._populations[1], "all_to_all", syn_spec=syn_params_exc)
        nest.Connect(self._populations[1], self._populations[0], "all_to_all", syn_spec=syn_params_inh)
        nest.Connect(self._populations[2], self._populations[0], "all_to_all", syn_spec=syn_params_thalamo)

    def create_recording_devices(self):
        for population in self._populations:
            recorder = nest.Create("spike_recorder", params={"record_to": "memory"})
            nest.Connect(population, recorder)
            self._recording_devices.append(recorder)

    def simulate(self):
        self.get_membrane_potentials()

    def get_membrane_potentials(self):
        print("Membrane Potential Before Simulation:")
        for i, population in enumerate(self._populations):
            mean_potential = nest.GetStatus(population, "V_m")[0]
            print(f"Population {i + 1} mean membrane potential: {mean_potential} mV")

        nest.Simulate(self._network_params["simtime"])

        print("Membrane Potential After Simulation:")
        for i, population in enumerate(self._populations):
            mean_potential = nest.GetStatus(population, "V_m")[0]
            print(f"Population {i + 1} mean membrane potential: {mean_potential} mV")

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
    "N_ex": 500,
    "N_in": 100,
    "N_thalamo": 1000,
    "simtime": 20000.0
}

# Define synaptic parameters
synapse_params = {
    "exc_weight": 35.0,
    "inh_weight": -15.0,
    "thalamo_weight": 2.0
}

# Define thalamo parameters
thalamo_params = {
    "N_thalamo": 1000
}

# Create network instance
ei_clustered_network = EIClusteredNetwork(network_params, synapse_params, thalamo_params)

# Run simulation and save spike times
simulation_results = ei_clustered_network.get_simulation(PathSpikes="spike_times.pkl")
print(simulation_results)

# Extract spike times and neuron indices for each population
spike_times = simulation_results['spiketimes']

# Define the percentage of each population to plot
exc_percentage = 0.5  # 50% of excitatory neurons
inh_percentage = 0.5  # 50% of inhibitory neurons
thalamic_percentage = 0.5  # 50% of thalamic neurons

# Get the total number of neurons in each population
total_exc_neurons = 2000
total_inh_neurons = 500
total_thalamic_neurons = 200

# Calculate the number of neurons to plot for each population
num_exc_neurons_to_plot = int(total_exc_neurons * exc_percentage)
num_inh_neurons_to_plot = int(total_inh_neurons * inh_percentage)
num_thalamic_neurons_to_plot = int(total_thalamic_neurons * thalamic_percentage)

# Randomly select neurons to plot for each population
exc_neurons_indices = np.random.choice(total_exc_neurons, num_exc_neurons_to_plot, replace=False)
inh_neurons_indices = np.random.choice(total_inh_neurons, num_inh_neurons_to_plot, replace=False)
thalamic_neurons_indices = np.random.choice(total_thalamic_neurons, num_thalamic_neurons_to_plot, replace=False)

print("Dimensions of spike_times[0][0]:", spike_times[0][0].shape)
print("Dimensions of spike_times[1][0]:", spike_times[1][0].shape)
print("Value of total_exc_neurons:", total_exc_neurons)

# Plot spike times
plt.figure(figsize=(12, 8))  # Increase plot size

# Plot excitatory neurons
for idx in range(len(exc_neurons_indices)):
    # Filter spike times for excitatory neurons
    exc_spike_times = spike_times[0][0][spike_times[1][0][:len(spike_times[0][0])] == exc_neurons_indices[idx]]  # Corrected indexing
    exc_spike_senders = spike_times[0][1][spike_times[1][0][:len(spike_times[0][0])] == exc_neurons_indices[idx]]  # Corrected indexing
    plt.scatter(exc_spike_times, exc_spike_senders, color='blue', s=10)

# Set label for excitatory neurons
plt.scatter([], [], color='blue', label='Excitatory')  # Set label

# Plot inhibitory neurons
for idx in range(len(inh_neurons_indices)):
    # Filter spike times for inhibitory neurons
    inh_spike_times = spike_times[0][0][spike_times[1][0][:len(spike_times[0][0])] == inh_neurons_indices[idx]]  # Corrected indexing
    inh_spike_senders = spike_times[0][1][spike_times[1][0][:len(spike_times[0][0])] == inh_neurons_indices[idx]]  # Corrected indexing
    plt.scatter(inh_spike_times, inh_spike_senders, color='red', s=10)

# Set label for inhibitory neurons
plt.scatter([], [], color='red', label='Inhibitory')  # Set label

# Plot thalamic neurons
for idx in range(len(thalamic_neurons_indices)):
    # Filter spike times for thalamic neurons
    thalamic_spike_times = spike_times[0][0][spike_times[1][0][:len(spike_times[0][0])] == thalamic_neurons_indices[idx]]  # Corrected indexing
    thalamic_spike_senders = spike_times[0][1][spike_times[1][0][:len(spike_times[0][0])] == thalamic_neurons_indices[idx]]  # Corrected indexing
    plt.scatter(thalamic_spike_times, thalamic_spike_senders, color='green', s=10)

# Set label for thalamic neurons
plt.scatter([], [], color='green', label='Thalamic')  # Set label

plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.title('Cluster Plot of Spike Times (50% of Each Population)')  # Add title
# Add custom legend
plt.legend(loc='upper right')  # Adjust legend position to right bottom
plt.grid(True)  # Add gridlines
plt.show()



























































































































































