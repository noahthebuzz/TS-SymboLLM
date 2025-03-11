import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta



##########################################################
### PLOT FUNCTIONS
##########################################################

# TODO:
# Single TSD Plot
# Multi TSD Plot
# Multi TSD Plot with each TSD in a separate subplot
# Multi TSD Plot with each TSD in a separate subplot and a shared y-axis
def single_tsd_plot(data: dict, title: str, xlabel: str, ylabel: str, save: bool = False, location: str = "./plots"):
    None

# TODO: Remove
def plot_tsd(data: dict, title: str, xlabel: str, ylabel: str, save: bool = False, show: bool = False):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(data.keys(), data.values(), marker='o', color='b')
    ax.set_title(title)
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
        time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        title = title.replace(" ", "_")
        plt.savefig(f"./plots/{title}.png")


def plot_multi_tsd(data1: dict, data2:dict, title: str, xlabel: str, ylabel: str, save: bool = False, show: bool = False):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(data1.keys(), data1.values(), marker='o', color='r', label='Temperature')
    ax.plot(data2.keys(), data2.values(), marker='o', color='b', label='Capacity')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True, which='both', linewidth='0.5', color='gray')
    plt.xticks(rotation=22.5)
    ax.legend(loc='best', fontsize=10)

    # Determine number of ticks in both x and y axis
    # data1 and data2 have the same dimensions
    max_data_value = max(data1.values())
    max_yticks = np.arange(0, round(max_data_value*1.25, -1), 1)[::5]
    if len(max_yticks) >= 10:
        ax.set_yticks(max_yticks[::len(max_yticks)//10])
    if len(data1) >= 11:
        ax.set_xticks(list(data1.keys())[::(len(data1)//11)])

    if save:
        time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        title = title.replace(" ", "_")
        plt.savefig(f"./plots/{title}.png")

    if show:
        plt.show()
    else:
        plt.close()


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
    rounded_data = list(np.around(data, 0))
    tsd2 = dict(zip(timestamps, rounded_data))
    # TODO implement plot_tsd
    #plot_tsd(tsd, "Random Data", "Time (HH:MM:SS)", "Value (int)", save=True, show=False)
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
    tsd = dict(zip(timestamps, data))
    rounded_data = list(np.around(data, 0))
    tsd2 = dict(zip(timestamps, rounded_data))
    # TODO implement plot_tsd
    return [tsd, tsd2]

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
    tsd = dict(zip(timestamps, data))
    rounded_data = list(np.around(data, 0))
    tsd2 = dict(zip(timestamps, rounded_data))
    # TODO implement plot_tsd
    return [tsd, tsd2]

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
    # TODO implement plot_tsd
    return tsd
        
###########################################################

