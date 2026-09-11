'''
Optional showcase: fabricates synthetic time-series datasets (and diagrams)
so you can try ts-symbollm without bringing your own data. Not part of the
installable ts_symbollm package — run it directly from a checkout:

    python examples/showcase/generate.py

See examples/showcase/README.md for details.
'''

from datetime import datetime

import numpy as np

import datagen

from ts_symbollm import plotting
from ts_symbollm.config import config
from ts_symbollm.representation import get_sax_string


####################################################################
### PLOTTING HELPER
####################################################################

def _plot(data_by_label: dict, abstraction_level: str, filename: str, xlabel: str, ylabel: str, location: str) -> None:
    series_data = {label: list(series.items()) for label, series in data_by_label.items()}
    plotting.plot_series(
        series_data,
        title=f"{filename} ({abstraction_level})",
        xlabel=xlabel,
        ylabel=ylabel,
        output_dir=location,
        filename=filename,
    )


####################################################################
### GENERATE DATA
####################################################################

def _generate_random_data():
    raw_path = "prompts/single/float/"
    rounded_path = "prompts/single/integer/"
    filename = "random"
    datasets = datagen.generate_random_tsd(n_instances=np.random.randint(low=1, high=101), interval_sec=60)
    raw_data, rounded_data = {"data": datasets[0]}, {"data": datasets[1]}

    _plot({filename: raw_data["data"]}, "raw", filename, "Time (HH:MM:SS)", "Value", raw_path)
    _plot({filename: rounded_data["data"]}, "rounded", filename, "Time (HH:MM:SS)", "Value", rounded_path)

    # Write data to json files
    config.write_prompt_json(data=raw_data    , path=raw_path + filename + "_test.json")
    config.write_prompt_json(data=rounded_data, path=rounded_path + filename + "_test.json")


def _generate_temperature_data():
    raw_path = "prompts/single/float/"
    rounded_path = "prompts/single/integer/"
    paasax_path = "prompts/single/paasax/"
    filename = "temperature"
    datasets = datagen.generate_temperature_tsd(n_instances=300, interval_sec=1)
    raw_data, rounded_data, paasax_data = {"data": datasets[0]}, {"data": datasets[1]}, {"data": datasets[2]}

    _plot({filename: raw_data["data"]}, "raw", filename, "Time (HH:MM:SS)", "Temperature (°C)", raw_path)
    _plot({filename: rounded_data["data"]}, "rounded", filename, "Time (HH:MM:SS)", "Temperature (°C)", rounded_path)
    _plot({filename: paasax_data["data"]}, "paasax", filename, "Time (HH:MM:SS)", "Temperature (°C)", paasax_path)

    paasax_string_data = {"data": get_sax_string(paasax_data)}

    # Write data to json files
    config.write_prompt_json(data=raw_data    , path=raw_path + filename + "_test.json")
    config.write_prompt_json(data=rounded_data, path=rounded_path + filename + "_test.json")
    config.write_prompt_json(data=paasax_string_data, path=paasax_path + filename + "_test.json")


def _generate_sequence_data():
    dir, name = "prompts/single/", "sequence_"
    # 0: even numbers, 1: odd numbers, 2: squared numbers, 3: oscillating harmonic numbers, 4: prime numbers
    sequences = ['even', 'odd', 'squared', 'harmosc', 'prime']
    paths = ['integer/', 'integer/', 'integer/', 'float/', 'integer/']
    for i in range(5):
        filename = f"{name}{sequences[i]}"
        path = f"{dir}{paths[i]}{filename}_test.json"
        series = datagen.generate_sequence_tsd(n_instances=10, interval_sec=3, sequence=i)
        data = {"data": series}

        _plot({filename: series}, "raw", filename, "Time (HH:MM:SS)", f"{sequences[i].capitalize()} Values", dir + paths[i])

        # Write data to json files
        config.write_prompt_json(data=data, path=path)


def _generate_multi_cpu_data():
    n_instances, interval_sec, time = 300, 1, datetime.now()
    temp_data = datagen.generate_temperature_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    cap_data = datagen.generate_capacity_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    raw_data = [cap_data[0], temp_data[0]]
    rounded_data = [cap_data[1], temp_data[1]]
    paasax_data = [cap_data[2], temp_data[2]]

    paths = ["prompts/multi/float/", "prompts/multi/integer/", "prompts/multi/paasax/"]
    filename = "cpu_temp_and_cap"
    labels = ["Capacity (%)", "Temperature (°C)"]
    ylabel = "Temperature (°C) / Capacity (%)"

    _plot(dict(zip(labels, raw_data)), "raw", filename, "Time (HH:MM:SS)", ylabel, paths[0])
    _plot(dict(zip(labels, rounded_data)), "rounded", filename, "Time (HH:MM:SS)", ylabel, paths[1])
    _plot(dict(zip(labels, paasax_data)), "paasax", filename, "Time (HH:MM:SS)", ylabel, paths[2])

    sax_string = [get_sax_string({"data": paasax_data[0]}), get_sax_string({"data": paasax_data[1]})]

    # Write data to json files
    config.write_prompt_json(data={"data_1": temp_data[0], "data_2": cap_data[0]}, path=f"{paths[0]}{filename}_test.json")
    config.write_prompt_json(data={"data_1": temp_data[1], "data_2": cap_data[1]}, path=f"{paths[1]}{filename}_test.json")
    config.write_prompt_json(data={"data_1": sax_string[1], "data_2": sax_string[0]}, path=f"{paths[2]}{filename}_test.json")


def _generate_multi_3_data():
    n_instances, interval_sec, time = 300, 1, datetime.now()
    temp_data = datagen.generate_temperature_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    cap_data = datagen.generate_capacity_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    rpm_data = datagen.generate_fanspeed_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    raw_data_plot = [cap_data[0], temp_data[0], rpm_data[3]]
    paasax_data = [cap_data[2], temp_data[2], rpm_data[2]]

    paths = ["prompts/multi/float/", "prompts/multi/paasax/"]
    filename = "multi_3"
    labels = ["Capacity (%)", "Temperature (°C)", "RPM (X*100)"]
    ylabel = "Temperature (°C) / Capacity (%) / Fanspeed (RPM)"

    _plot(dict(zip(labels, raw_data_plot)), "raw", filename, "Time (HH:MM:SS)", ylabel, paths[0])
    _plot(dict(zip(labels, paasax_data)), "paasax", filename, "Time (HH:MM:SS)", ylabel, paths[1])

    sax_string = [get_sax_string({"data": paasax_data[0]}), get_sax_string({"data": paasax_data[1]}), get_sax_string({"data": paasax_data[2]})]

    # Write data to json files
    config.write_prompt_json(data={"data_1": temp_data[0], "data_2": cap_data[0], "data_3": rpm_data[0]}, path=f"{paths[0]}{filename}_test.json")
    config.write_prompt_json(data={"data_1": sax_string[1], "data_2": sax_string[0], "data_3": sax_string[2]}, path=f"{paths[1]}{filename}_test.json")


def _generate_multi_4_data():
    n_instances, interval_sec, time = 300, 1, datetime.now()
    temp_data = datagen.generate_temperature_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    cap_data = datagen.generate_capacity_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    rpm_data = datagen.generate_fanspeed_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    volt_data = datagen.generate_voltage_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    raw_data_plot = [cap_data[0], temp_data[0], rpm_data[3], volt_data[3]]
    paasax_data = [cap_data[2], temp_data[2], rpm_data[2], volt_data[2]]

    paths = ["prompts/multi/float/", "prompts/multi/paasax/"]
    filename = "multi_4"
    labels = ["Capacity (%)", "Temperature (°C)", "RPM (X*100)", "Voltage (%)"]
    ylabel = "Temperature (°C) / Capacity (%) / Fanspeed (RPM) / Voltage (%)"

    _plot(dict(zip(labels, raw_data_plot)), "raw", filename, "Time (HH:MM:SS)", ylabel, paths[0])
    _plot(dict(zip(labels, paasax_data)), "paasax", filename, "Time (HH:MM:SS)", ylabel, paths[1])

    sax_string = [get_sax_string({"data": paasax_data[0]}), get_sax_string({"data": paasax_data[1]}), get_sax_string({"data": paasax_data[2]}), get_sax_string({"data": paasax_data[3]})]

    # Write data to json files
    config.write_prompt_json(data={"data_1": temp_data[0], "data_2": cap_data[0], "data_3": rpm_data[0], "data_4": volt_data[0]}, path=f"{paths[0]}{filename}_test.json")
    config.write_prompt_json(data={"data_1": sax_string[1], "data_2": sax_string[0], "data_3": sax_string[2], "data_4": sax_string[3]}, path=f"{paths[1]}{filename}_test.json")


def _generate_multi_5_data():
    n_instances, interval_sec, time = 300, 1, datetime.now()
    temp_data = datagen.generate_temperature_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    cap_data = datagen.generate_capacity_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    rpm_data = datagen.generate_fanspeed_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    volt_data = datagen.generate_voltage_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    roomtemp_data = datagen.generate_roomtemp_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    raw_data_plot = [cap_data[0], temp_data[0], rpm_data[3], volt_data[3], roomtemp_data[0]]
    paasax_data = [cap_data[2], temp_data[2], rpm_data[2], volt_data[2], roomtemp_data[2]]

    paths = ["prompts/multi/float/", "prompts/multi/paasax/"]
    filename = "multi_5"
    labels = ["Capacity (%)", "Temperature (°C)", "RPM (X*100)", "Voltage (%)", "Room Temperature (°C)"]
    ylabel = "Temperature (°C) / Capacity (%) / Fanspeed (RPM) / Voltage (%) / Room Temperature (°C)"

    _plot(dict(zip(labels, raw_data_plot)), "raw", filename, "Time (HH:MM:SS)", ylabel, paths[0])
    _plot(dict(zip(labels, paasax_data)), "paasax", filename, "Time (HH:MM:SS)", ylabel, paths[1])

    sax_string = [get_sax_string({"data": paasax_data[0]}), get_sax_string({"data": paasax_data[1]}), get_sax_string({"data": paasax_data[2]}), get_sax_string({"data": paasax_data[3]}), get_sax_string({"data": paasax_data[4]})]

    # Write data to json files
    config.write_prompt_json(data={"data_1": temp_data[0], "data_2": cap_data[0], "data_3": rpm_data[0], "data_4": volt_data[0], "data_5": roomtemp_data[0]}, path=f"{paths[0]}{filename}_test.json")
    config.write_prompt_json(data={"data_1": sax_string[1], "data_2": sax_string[0], "data_3": sax_string[2], "data_4": sax_string[3], "data_5": sax_string[4]}, path=f"{paths[1]}{filename}_test.json")

###########################################################

def generate_data(random: bool = False, temperature: bool = False, sequence: bool = False, cpu_temp_cap_tsd: bool = False, multi_3_tsd: bool = False, multi_4_tsd: bool = False, multi_5_tsd: bool = False):
    if random:
        _generate_random_data()

    if temperature:
        _generate_temperature_data()

    if sequence:
        _generate_sequence_data()

    if cpu_temp_cap_tsd:
        _generate_multi_cpu_data()

    if multi_3_tsd:
        _generate_multi_3_data()

    if multi_4_tsd:
        _generate_multi_4_data()

    if multi_5_tsd:
        _generate_multi_5_data()


if __name__ == "__main__":
    generate_data(temperature=True)
