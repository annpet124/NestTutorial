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

# Generate time points
t = np.linspace(-1.5, 1.5, 1000)

# Define parameters for Pyr and PV cells
rate_pyr = 25  # Rate of Pyr cells
rate_pv = 30   # Rate of PV cells

# Calculate exponential curves for Pyr and PV cells
pyr_curve = rate_pyr * np.exp(-(t - 0.1)**2 / 0.2)
pv_curve = rate_pv * np.exp(-(t + 0.1)**2 / 0.2)

# Plot the curves
plt.plot(t, pyr_curve, 'r', label='Pyr')
plt.plot(t, pv_curve, 'b', label='PV')

# Highlight the response onset time for each neuron
plt.axvline(x=0.18, color='r', linestyle='--', label='Pyr onset')
plt.axvline(x=-0.54, color='b', linestyle='--', label='PV onset')

# Add shaded area to depict SEM (Standard Error of Mean)
plt.fill_between(t, pyr_curve - 5, pyr_curve + 5, color='r', alpha=0.3)
plt.fill_between(t, pv_curve - 5, pv_curve + 5, color='b', alpha=0.3)

# Set labels and title
plt.xlabel('Time (s)')
plt.ylabel('Rate (sp/s)')
plt.title('Task-related activity around movement onset')
plt.xlim(-1.5, 1.5)

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()

