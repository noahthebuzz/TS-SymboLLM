import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import ollama
from ollama import GenerateResponse


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
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_time_instances(start_time: datetime, interval_sec: int, n_instances: int) -> list[str]:
    return [(start_time + timedelta(seconds=interval_sec * i)).strftime("%H:%M:%S") for i in range(n_instances)]


def generate_random_data(min_value: int, max_value: int, n_instances: int) -> list[int]:
    return np.random.randint(min_value, max_value, n_instances).tolist()


def generate_temperature_data(n_data_points: int, start_temp: float, stable_deviation: float , unstable_start: int, unstable_deviation: float, expo_increase_start: int, final_temp: float, rounded_values: bool = False) -> list[int]:
    # 0. Ruhige Phase
    ruhige_phase = np.random.normal(loc=start_temp, scale=stable_deviation, size=unstable_start)

    # 1. Stabile Phase
    unstable_phase = np.random.normal(loc=start_temp, scale=unstable_deviation, size=expo_increase_start-unstable_start)

    # 2. Exponentielle Anstiegsphase
    time_increase = np.arange(n_data_points - expo_increase_start)
    increase_phase = start_temp + (final_temp - start_temp) * time_increase ** 2 / (n_data_points - expo_increase_start) ** 2 + np.random.normal(scale=(stable_deviation + unstable_deviation)/1, size=n_data_points - expo_increase_start)

    # 3. Kombinieren der Daten
    temperature_data = np.concatenate([ruhige_phase, unstable_phase, increase_phase])

    #print(f"temperature_data has {len(temperature_data)} data points.")
    # TODO: UNCOMMENT
    #plot_data(data=temperature_data, plot_label="Temperatur", x_label="Zeit (t)", y_label="Temperatur (°C)", title="Temperaturdaten mit stabiler Phase und exponentiellem Anstieg", output_file="temperaturverlauf_exponentiell.png")

    if rounded_values:
        return temperature_data.round().tolist()
    return temperature_data.tolist()


def generate_sequence_data(n_instances: int,  type_of_sequence: int, every_n_th_number: int = 3) -> list[int]:    
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
        

def generate_tsd(kind_of_data: str, n_instances: int, interval_sec: int, sequence: int = 0) -> dict:
    start_time = datetime.now()
    time_instances = generate_time_instances(start_time, interval_sec, n_instances)

    data = []

    if kind_of_data == "random":
        data = generate_random_data(0, 100, n_instances)
        tsd = dict(zip(time_instances, data))
        plot_tsd(tsd, "Random Data", "Time (HH:MM:SS)", "Value (int)", save=True, show=False)

    elif kind_of_data == "temperature":
        data = generate_temperature_data(n_data_points=n_instances, start_temp=50, stable_deviation=0.75, unstable_start=round(n_instances//2.5), unstable_deviation=3.5, expo_increase_start=round(n_instances//1.75), final_temp=95, rounded_values=True)
        tsd = dict(zip(time_instances, data))
        plot_tsd(tsd, "Temperature Data of Processor XY", "Time (HH:MM:SS)", "Temperature (°C)", save=True, show=False)

    elif kind_of_data == "sequence":
        data = generate_sequence_data(n_instances=n_instances, type_of_sequence=sequence)
        tsd = dict(zip(time_instances, data))
        plot_tsd(tsd, f"Sequence Data ({['even numbers', 'odd numbers', 'squared numbers', 'oscillating harmonic numbers', 'prime numbers'][sequence]})", "Time (HH:MM:SS)", "Value (int)", save=True, show=False)

    return tsd


def pull_model_from_ollama(model: str) -> bool:
    try:
        response = ollama.pull(model=model, store=True)
        progress_states = set()
        for progress in response:
            if progress.get('status') in progress_states:
                continue
            progress_states.add(progress.get('status'))
            print(progress.get('status'))
        print("\nModel pulled successfully.\n")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False



if __name__ == "__main__":
    None