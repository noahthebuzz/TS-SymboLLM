import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tslearn.piecewise import SymbolicAggregateApproximation
from tslearn.preprocessing import TimeSeriesScalerMeanVariance



##########################################################
### PLOT FUNCTIONS
##########################################################

# TODO:
# Multi TSD Plot with each TSD in a separate subplot
# Multi TSD Plot with each TSD in a separate subplot and a shared y-axis
def plot_single_tsd(data: dict, abstraction_level: str, title: str, xlabel: str, ylabel: str, save: bool = False, location: str = "./plots/") -> None:
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(data.keys(), data.values(), marker='o', color='b')
    ax.set_title(title + f" ({abstraction_level})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True, which='both', linewidth='0.5', color='gray')
    max_data_value = max(data.values())
    max_yticks = np.arange(0, round(max_data_value*1.25, -1), 1)[::5]
    if len(max_yticks) >= 10:
        ax.set_yticks(max_yticks[::len(max_yticks)//10])
    if len(data) >= 11:
        ax.set_xticks(list(data.keys())[::(len(data)//11)])
    plt.xticks(rotation=22.5)

    if save:
        title = title.replace(" ", "_")
        plt.savefig(f"{location}{title}_plot.png")


def plot_multi_tsd(data: list[dict], labels: list[dict], abstraction_level: str, title: str, xlabel: str, ylabel: str, save: bool = False, location: str = "./plots/") -> None:
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    for i in range(len(data)):
        ax.plot(data[i].keys(), data[i].values(), marker='o', color=colors[i], label=labels[i])
    ax.set_title(title + f" ({abstraction_level})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True, which='both', linewidth='0.5', color='gray')
    plt.xticks(rotation=22.5)
    ax.legend(loc='best', fontsize=10)

    # Determine number of ticks in both x and y axis
    # data have the same dimensions
    max_data_value = 1
    for dataset in data:
        max_data_value = max(max_data_value, max(dataset.values()))

    max_yticks = np.arange(0, round(max_data_value*1.3, -1), 1)[::5]
    if len(max_yticks) >= 10:
        ax.set_yticks(max_yticks[::len(max_yticks)//10])
    if len(data[0]) >= 11:
        ax.set_xticks(list(data[0].keys())[::(len(data[0])//11)])

    if save:
        title = title.replace(" ", "_")
        plt.savefig(f"{location}{title}_plot.png")


def plot_multi_tsd_separate(data: list[dict], labels: list[dict], abstraction_level: str, title: str, xlabel: str, ylabel: str, save: bool = False, location: str = "./plots/") -> None:
    None


##########################################################
### TIME INSTANCE GENERATION FUNCTION
##########################################################

def _generate_time_instances(start_time: datetime, interval_sec: int, n_instances: int) -> list[str]:
    return [(start_time + timedelta(seconds=interval_sec * i)).strftime("%H:%M:%S") for i in range(n_instances)]


##########################################################
### DATA GENERATION FUNCTIONS
##########################################################

def _generate_random_data(min_value: float, max_value: float, n_instances: int) -> list[float]:
    return np.random.uniform(min_value, max_value, n_instances)

def generate_random_tsd(n_instances: int, interval_sec: int) -> list[dict[str, int]]:
    timestamps = _generate_time_instances(datetime.now(), interval_sec, n_instances)
    data = _generate_random_data(0, 100, n_instances)
    tsd = dict(zip(timestamps, data))
    rounded_data = np.around(data, 0).tolist()
    tsd2 = dict(zip(timestamps, rounded_data))
    return [tsd, tsd2]

###########################################################

def _generate_temperature_data(n_data_points: int, start_temp: float, stable_deviation: float , unstable_start: int, unstable_deviation: float, expo_increase_start: int, final_temp: float) -> list[float]:
    # 0. Ruhige Phase
    ruhige_phase = np.random.normal(loc=start_temp, scale=stable_deviation, size=unstable_start)

    # 1. Stabile Phase
    unstable_phase = np.random.normal(loc=start_temp, scale=unstable_deviation, size=expo_increase_start-unstable_start)

    # 2. Exponentielle Anstiegsphase
    time_increase = np.arange(n_data_points - expo_increase_start)
    increase_phase = start_temp + (final_temp - start_temp) * time_increase ** 2 / (n_data_points - expo_increase_start) ** 2 + np.random.normal(scale=(stable_deviation + unstable_deviation)/1, size=n_data_points - expo_increase_start)

    # 3. Kombinieren der Daten
    temperature_data = np.concatenate([ruhige_phase, unstable_phase, increase_phase])

    return temperature_data.tolist()

def generate_temperature_tsd(n_instances: int, interval_sec: int, time: datetime = datetime.now()) -> list[dict]:
    timestamps = _generate_time_instances(time, interval_sec, n_instances)
    data = _generate_temperature_data(
            n_data_points=n_instances, 
            start_temp=50, 
            stable_deviation=0.75, 
            unstable_start=round(n_instances//2.5), 
            unstable_deviation=3.5, 
            expo_increase_start=round(n_instances//1.75), 
            final_temp=95)
    tsd_raw = dict(zip(timestamps, data))
    rounded_data = np.around(data, 0).tolist()
    tsd_rounded = dict(zip(timestamps, rounded_data))
    tsd_sax = approximate_tsd(data=data, timestamps=timestamps)
    return [tsd_raw, tsd_rounded, tsd_sax]

###########################################################

def approximate_tsd(data: list[int], timestamps: list[str], n_paa_segments: int = 10, n_sax_symbols = 8) -> dict[str, int]:

    dataset = np.array(data).reshape(1, -1)
    #print(f"Dataset:\n{dataset}")

    scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  # Rescale time series
    normalized_dataset = scaler.fit_transform(dataset)
    sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
    sax_values = sax.fit_transform(normalized_dataset)

    reduced_timestamps = timestamps[::round(len(timestamps)/n_paa_segments)]

    return dict(zip(reduced_timestamps, sax_values[0].ravel().tolist()))


def get_sax_string(data: dict) -> dict[str, str]:
    keys = []
    values = []
    for datum in data.values():
        for key in datum.keys():
            keys.append(key)
        for value in datum.values():
            values.append(value)

    alphabet = 'abcdefghijklmnopqrstuvwxyz'[:len(values)]
    sax_string = [''.join([alphabet[int(i)] for i in values])][0]

    return dict(zip(keys, sax_string))


def get_sax_values(data: dict) -> dict[str, int]:
    keys = []
    values = []
    for datum in data.values():
        for key in datum.keys():
            keys.append(key)
        for value in datum.values():
            values.append(value)

    # match the letter in values to the corresponding number
    # like a = 1, b = 2, c = 3, ...
    sax_values = []
    for i in values:
        sax_values.append(ord(i) - ord('a') + 1)

    return dict(zip(keys, sax_values))



###########################################################

def _generate_capacity_data(n_data_points: int, start_capacity: float, stable_deviation: float, jump_start: int, jump_capacity:float, jump_deviation: float, decrease_start: int, final_capacity: float) -> list[float]:
    # 0. Ruhige Phase
    ruhige_phase = np.random.normal(loc=start_capacity, scale=stable_deviation, size=jump_start)

    # 1. Jump Phase
    jump_phase = np.random.normal(loc=jump_capacity, scale=jump_deviation, size=decrease_start-jump_start)

    # 2. Decrease Phase
    time_decrease = np.arange(n_data_points - decrease_start)
    decrease_phase = jump_capacity - (jump_capacity - final_capacity) * time_decrease / (n_data_points - decrease_start) + np.random.normal(scale=(stable_deviation + jump_deviation)/1, size=n_data_points - decrease_start)

    # 3. Kombinieren der Daten
    capacity_data = np.concatenate([ruhige_phase, jump_phase, decrease_phase])

    return capacity_data.tolist()

def generate_capacity_tsd(n_instances: int, interval_sec: int, time: datetime = datetime.now()) -> list[dict]:
    timestamps = _generate_time_instances(time, interval_sec, n_instances)
    data = _generate_capacity_data(
                n_data_points=n_instances,
                start_capacity=30,
                stable_deviation=2.0,
                jump_start=round(n_instances//2.5),
                jump_capacity=75,
                jump_deviation=3.5,
                decrease_start=round(n_instances//2.125),
                final_capacity=30,
    )
    tsd_raw = dict(zip(timestamps, data))
    rounded_data = np.around(data, 0).tolist()
    tsd_rounded = dict(zip(timestamps, rounded_data))
    tsd_sax = approximate_tsd(data=data, timestamps=timestamps)
    return [tsd_raw, tsd_rounded, tsd_sax]

###########################################################

def _generate_sequence_data(n_instances: int,  type_of_sequence: int, every_n_th_number: int = 3) -> list[int]:    
    if every_n_th_number < 1:
        every_n_th_number = 1

    if type_of_sequence == 0 or type_of_sequence == 1:
        # return every n-th number of even/odd numbers
        return [i for i in range(type_of_sequence, n_instances * 2 * every_n_th_number, 2 * every_n_th_number)]
    
    elif type_of_sequence == 2:
        # return every n-th number of squares
        return [i**2 for i in range(0, n_instances * every_n_th_number, every_n_th_number)]
    
    elif type_of_sequence == 3:
        # return every n-th number of the oscillating harmonic sequence
        return [round(1/i, 2) if i % 2 == 0 else round(-1/i, 2) for i in range(1, n_instances * every_n_th_number + 1, every_n_th_number)]
    
    elif type_of_sequence == 4:
        # return every n-th number of the prime numbers
        primes = [2]
        i = 3
        while len(primes) < n_instances * every_n_th_number:
            for j in range(2, i):
                if i % j == 0:
                    break
            else:
                primes.append(i)
            i += 1
        return [primes[i] for i in range(0, n_instances * every_n_th_number, every_n_th_number)]
    
def generate_sequence_tsd(n_instances: int, interval_sec: int, sequence: int, time: datetime = datetime.now()) -> dict:
    timestamps = _generate_time_instances(time, interval_sec, n_instances)
    data = _generate_sequence_data(n_instances=n_instances, type_of_sequence=sequence)
    tsd = dict(zip(timestamps, data))
    return tsd
        
###########################################################