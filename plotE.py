import nest
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# Define variables
exc_weights = [40.0, 60.0, 80.0]
inh_weights = [-20.0, -25.0, -30.0]
spike_times_list = []


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

        reduced_size = int(0.5 * N_E)
        subset_E = random.sample(range(N_E), min(reduced_size, N_E))
        subset_I = random.sample(range(N_I), min(reduced_size, N_I))

        self._populations.append(nest.Create("iaf_psc_alpha", len(subset_E)))
        self._populations.append(nest.Create("iaf_psc_alpha", len(subset_I)))
        self._populations.append(nest.Create("iaf_psc_alpha", N_thalamo))

    def connect_network(self):
        cluster_strength = self._network_params.get("cluster_strength", 1.0)
        syn_params_exc = {"weight": self._synapse_params["exc_weight"] * cluster_strength}
        syn_params_inh = {"weight": self._synapse_params["inh_weight"] * cluster_strength}
        syn_params_thalamo = {"weight": self._synapse_params["thalamo_weight"] * cluster_strength}

        nest.Connect(self._populations[0], self._populations[1], "all_to_all", syn_spec=syn_params_exc)
        nest.Connect(self._populations[1], self._populations[0], "all_to_all", syn_spec=syn_params_inh)
        nest.Connect(self._populations[2], self._populations[0], "all_to_all", syn_spec=syn_params_thalamo)

    def adjust_input_currents(self):
        nest.SetStatus(self._populations[0], {"I_e": 1800.0})
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


def alter_network(cluster_strength, network_params, synapse_params):
    network_params["cluster_strength"] = cluster_strength
    ei_clustered_network = EIClusteredNetwork(network_params, synapse_params, {"N_thalamo": network_params["N_thalamo"]})
    ei_clustered_network.setup_network()
    ei_clustered_network.simulate()
    spike_counts = ei_clustered_network.get_recordings()

    # Generate spike times list for different cluster strengths
    max_cluster_strength = network_params[
        "N_populations"]  # Maximum cluster strength should not exceed the total number of populations
    spike_times_list = generate_spike_times_list(network_params["N_neurons"], network_params["N_populations"],
                                                 network_params["simtime"], max_cluster_strength)

    return spike_counts

def generate_spike_times(num_neurons, sim_duration):
    spike_times = []
    for _ in range(num_neurons):
        spike_count = np.random.poisson(5)  # Assuming an average of 5 spikes per neuron
        spikes = np.random.uniform(0, sim_duration, spike_count)
        spike_times.append(spikes)
    return spike_times

def generate_population_spike_times(num_neurons, num_populations, sim_duration, cluster_strength):
    population_spike_times = []
    for _ in range(num_populations):
        population_spike_times.append(generate_spike_times(num_neurons, sim_duration))
    if cluster_strength > 1:
        # Apply clustering by repeating the spike times for each population
        clustered_spike_times = []
        for times in population_spike_times:
            for _ in range(cluster_strength):
                clustered_spike_times.append(times)
        return clustered_spike_times
    else:
        return population_spike_times

def generate_spike_times_list(num_neurons, num_populations, sim_duration, max_cluster_strength):
    spike_times_list = []
    for cluster_strength in range(1, max_cluster_strength + 1):
        population_spike_times = generate_population_spike_times(num_neurons, num_populations, sim_duration, cluster_strength)
        spike_times_list.append(population_spike_times)
    return spike_times_list

# Define parameters
num_neurons = 1000
num_populations = 4
sim_duration = 2000.0
max_cluster_strength = num_populations  # Maximum cluster strength should not exceed the total number of populations

# Generate spike times list for different cluster strengths
spike_times_list = generate_spike_times_list(num_neurons, num_populations, sim_duration, max_cluster_strength)

# Plot raster plots for different cluster strengths
for i, spike_times in enumerate(spike_times_list):
    plt.figure(figsize=(10, 6))
    plot_spike_raster(exc_weights, inh_weights, spike_times, sim_duration)
    plt.title(f'Spike Raster Plot for Cluster Strength = {i+1}')



def calculate_fano_factor(spike_counts):
    spike_counts_array = np.array([len(events[0]) for events in spike_counts if events is not None])
    mean_spike_count = np.mean(spike_counts_array)
    variance_spike_count = np.var(spike_counts_array)
    return variance_spike_count / mean_spike_count if mean_spike_count != 0 else float('nan')


def calculate_cv(isis):
    mean_isi = np.mean(isis)
    std_isi = np.std(isis)
    return std_isi / mean_isi if mean_isi != 0 else float('nan')


def plot_spike_counts(cluster_strengths, spike_counts_list):
    markers = ['o', 's', '^']
    colors = ['b', 'g', 'r']
    labels = ['Excitatory', 'Inhibitory', 'Thalamic']  # Label legend

    plt.figure(figsize=(10, 6))
    for i, counts in enumerate(spike_counts_list):
        marker = markers[i]
        color = colors[i]
        for j, strength in enumerate(cluster_strengths):
            mean_count = np.mean(counts[j])  # Taking mean count for each cluster strength
            plt.plot(strength, mean_count, marker=marker, color=color)

    plt.xlabel('Cluster Strength')
    plt.ylabel('Average Total Spike Counts')
    plt.title('Average Total Spike Counts vs. Cluster Strength')

    # Adding legend for population types
    patches = [plt.Line2D([0], [0], marker=markers[i], color='w', markerfacecolor=colors[i], markersize=10, label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Collecting data for the table
    table_data = []
    for i, strength in enumerate(cluster_strengths):
        row_data = [strength]
        for j in range(len(labels)):
            mean_count = np.mean(spike_counts_list[j][i])
            row_data.append(mean_count)
        table_data.append(row_data)

    headers = ["Cluster Strength"] + labels

    plt.figure(figsize=(8, 4))
    plt.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    plt.axis('off')
    plt.show()


def alter_synaptic_weights(synapse_params, network_params, thalamo_params):
    ei_clustered_network = EIClusteredNetwork(network_params, synapse_params, thalamo_params)
    ei_clustered_network.setup_network()
    ei_clustered_network.simulate()
    spike_counts = ei_clustered_network.get_recordings()
    spike_times_list = []

    return spike_times_list


def generate_spike_times(num_neurons, sim_duration):
    spike_times = []
    for _ in range(num_neurons):
        spike_count = np.random.poisson(5)  # Assuming an average of 5 spikes per neuron
        spikes = np.random.uniform(0, sim_duration, spike_count)
        spike_times.append(spikes)
    return spike_times

# Generate spike times for each population
population_0_spike_times = generate_spike_times(1000, 2000.0)
population_1_spike_times = generate_spike_times(1000, 2000.0)
population_2_spike_times = generate_spike_times(1000, 2000.0)
population_3_spike_times = generate_spike_times(1000, 2000.0)

# Combine spike times into spike_times_list
spike_times_list = [population_0_spike_times, population_1_spike_times, population_2_spike_times, population_3_spike_times]


def plot_spike_raster(exc_weights, inh_weights, spike_times_list, sim_duration):
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y'] * (len(exc_weights) * len(inh_weights) // len(['b', 'g', 'r', 'c', 'm', 'y']) + 1)
    used_labels = {}  # Dictionary to keep track of used labels

    for i, spike_times in enumerate(spike_times_list):
        if spike_times is not None:
            color_index = i % len(colors)
            exc_weight = exc_weights[i // len(inh_weights)]
            inh_weight = inh_weights[i % len(inh_weights)]
            label = f'Exc Weight: {exc_weight}, Inh Weight: {inh_weight}'
            if label not in used_labels:
                # Flatten the spike times list into a 1D array
                flat_spike_times = np.concatenate([times for times in spike_times])
                flat_neurons = np.concatenate([np.ones_like(times) * i for i, times in enumerate(spike_times)])
                plt.scatter(flat_spike_times, flat_neurons, s=1, color=colors[color_index], label=label)
                used_labels[label] = True
            else:
                flat_spike_times = np.concatenate([times for times in spike_times])
                flat_neurons = np.concatenate([np.ones_like(times) * i for i, times in enumerate(spike_times)])
                plt.scatter(flat_spike_times, flat_neurons, s=1, color=colors[color_index])

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.title('Spike Raster Plot')
    if used_labels:  # Check if labels are available
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.xlim(0, sim_duration)  # Limit x-axis to the duration of the simulation
    plt.ylim(-0.5, len(spike_times_list) - 0.5)  # Set y-axis limits to cover all populations
    plt.yticks(np.arange(len(spike_times_list)), ['Population 0', 'Population 1', 'Population 2', 'Population 3'])
    plt.tight_layout()
    plt.show()


def plot_task_related_activity():
    t = np.linspace(-1.5, 0, 1000)
    rate_pyr = 25
    rate_pv = 30

    pyr_curve = rate_pyr * np.exp(-(t - 0.208)**2 / 0.2)
    pv_curve = rate_pv * np.exp(-(t + 0.208)**2 / 0.2)

    pyr_curve /= np.max(pyr_curve)
    pv_curve /= np.max(pv_curve)

    pyr_curve *= 100
    pv_curve *= 100

    plt.plot(t, pyr_curve, 'r', label='Pyr')
    plt.plot(t, pv_curve, 'b', label='PV')

    plt.axvline(x=0.208, color='r', linestyle='--', label='Pyr onset')
    plt.axvline(x=-0.208, color='b', linestyle='--', label='PV onset')

    plt.fill_between(t, pyr_curve - 5, pyr_curve + 5, color='r', alpha=0.3)
    plt.fill_between(t, pv_curve - 5, pv_curve + 5, color='b', alpha=0.3)

    plt.axhline(y=10, color='g', linestyle='--')
    plt.axhline(y=20, color='m', linestyle='--')

    avg_diff_ms = np.abs(np.mean(pyr_curve / 100) - np.mean(pv_curve / 100)) * 1000
    plt.text(-1.55, 75, f'Diff. betw. avg. profiles = {avg_diff_ms:.2f} ms', fontsize=10, color='black')

    plt.xlabel('Time to MO (s)')
    plt.ylabel('%Peak')
    plt.title('Task-related activity around movement onset')

    plt.legend()
    plt.grid(True)
    plt.show()


plot_task_related_activity()


def calculate_fano_and_cv(cluster_strengths, spike_counts_list):
    fano_factors = []
    cvs = []

    for spike_counts in spike_counts_list:
        if spike_counts is not None:
            fano_factor = calculate_fano_factor(spike_counts)
            fano_factors.append(fano_factor)

            isis = [np.diff(np.array(events[0])) for events in spike_counts if len(events[0]) > 1]
            isis = np.concatenate(isis) if len(isis) > 0 else np.array([])
            cv = calculate_cv(isis)
            cvs.append(cv)
        else:
            fano_factors.append(float('nan'))
            cvs.append(float('nan'))

    print("Fano Factors:", fano_factors)
    print("CVs:", cvs)


network_params = {
    "N_ex": 1000,
    "N_in": 250,
    "N_thalamo": 100,
    "simtime": 500.0,
    "cluster_strength": 0.5
}

synapse_params = {
    "exc_weight": 60.0,
    "inh_weight": -25.0,
    "thalamo_weight": 2.0
}

cluster_strengths = [0.1, 0.5, 1.0]
spike_counts_list = [alter_network(strength, network_params, synapse_params) for strength in cluster_strengths]
plot_spike_counts(cluster_strengths, spike_counts_list)

network_params["simtime"] = 2000.0
thalamo_params = {"N_thalamo": 100}

spike_times_list = []
for exc_weight in exc_weights:
    for inh_weight in inh_weights:
        synapse_params["exc_weight"] = exc_weight
        synapse_params["inh_weight"] = inh_weight
        spike_times = alter_synaptic_weights(synapse_params, network_params, thalamo_params)
        spike_times_list.append(spike_times)

calculate_fano_and_cv(cluster_strengths, spike_counts_list)
