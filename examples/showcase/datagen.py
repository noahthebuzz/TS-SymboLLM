'''
Synthetic time-series generators for the optional showcase (see
examples/showcase/README.md). These exist purely to fabricate demo
datasets when you don't have your own data handy — the installable
ts_symbollm package does not depend on any of this.
'''

import numpy as np
from datetime import datetime, timedelta

from ts_symbollm.representation import approximate_tsd


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

def _generate_fanspeed_data(n_data_points: int, start_rpm: float, stable_deviation: float, jump_start: int, jump_rpm: float, decrease_start: int, final_rpm: float) -> list[float]:
    # 0. Ruhige Phase
    ruhige_phase = np.random.normal(loc=start_rpm, scale=stable_deviation, size=jump_start)

    # 1. Jump Phase
    jump_phase = np.random.normal(loc=jump_rpm, scale=stable_deviation, size=decrease_start-jump_start)

    # 2. Decrease Phase
    time_decrease = np.arange(n_data_points - decrease_start)
    decrease_phase = jump_rpm - (jump_rpm - final_rpm) * time_decrease / (n_data_points - decrease_start) + np.random.normal(scale=(stable_deviation)/1, size=n_data_points - decrease_start)

    # 3. Kombinieren der Daten
    fanspeed_data = np.concatenate([ruhige_phase, jump_phase, decrease_phase])

    return fanspeed_data.tolist()

def generate_fanspeed_tsd(n_instances: int, interval_sec: int, time: datetime = datetime.now()) -> list[dict]:
    timestamps = _generate_time_instances(time, interval_sec, n_instances)
    data = _generate_fanspeed_data(
                n_data_points=n_instances,
                start_rpm=40.0,
                stable_deviation=3.0,
                jump_start=round(n_instances//2.5),
                jump_rpm=85.0,
                decrease_start=round(n_instances//2.125),
                final_rpm=10.0,
    )
    tsd_raw = dict(zip(timestamps, [100 * x for x in data]))
    tsd_raw_plot = dict(zip(timestamps, data))
    rounded_data = np.around(data, 0).tolist()
    tsd_rounded = dict(zip(timestamps, rounded_data))
    tsd_sax = approximate_tsd(data=data, timestamps=timestamps)
    return [tsd_raw, tsd_rounded, tsd_sax, tsd_raw_plot]

###########################################################

def _generate_voltage_data(n_data_points: int, start_capacity: float, stable_deviation: float, jump_start: int, jump_capacity: float, jump_deviation: float, decrease_start: int, final_capacity: float) -> list[float]:
    # 0. Ruhige Phase
    ruhige_phase = np.random.normal(loc=start_capacity, scale=stable_deviation, size=jump_start)

    # 1. Jump Phase
    jump_phase = np.random.normal(loc=jump_capacity, scale=jump_deviation, size=decrease_start-jump_start)

    # 2. Decrease Phase
    time_decrease = np.arange(n_data_points - decrease_start)
    decrease_phase = jump_capacity - (jump_capacity - final_capacity) * time_decrease / (n_data_points - decrease_start) + np.random.normal(scale=(stable_deviation + jump_deviation)/1, size=n_data_points - decrease_start)

    # 3. Kombinieren der Daten
    voltage_data = np.concatenate([ruhige_phase, jump_phase, decrease_phase])

    return voltage_data.tolist()

def generate_voltage_tsd(n_instances: int, interval_sec: int, time: datetime = datetime.now()) -> list[dict]:
    timestamps = _generate_time_instances(time, interval_sec, n_instances)
    data = _generate_voltage_data(
                n_data_points=n_instances,
                start_capacity=35,
                stable_deviation=3.25,
                jump_start=round(n_instances//2.5),
                jump_capacity=80,
                jump_deviation=5.5,
                decrease_start=round(n_instances//2.125),
                final_capacity=60
    )
    tsd_raw = dict(zip(timestamps, data))
    tsd_raw_plot = dict(zip(timestamps, data))
    rounded_data = np.around(data, 0).tolist()
    tsd_rounded = dict(zip(timestamps, rounded_data))
    tsd_sax = approximate_tsd(data=data, timestamps=timestamps)
    return [tsd_raw, tsd_rounded, tsd_sax, tsd_raw_plot]

###########################################################

def _generate_roomtemp_data(n_data_points: int, const_temp: float, stable_deviation: float) -> list[float]:
    # 0. Ruhige Phase
    ruhige_phase = np.random.normal(loc=const_temp, scale=stable_deviation, size=n_data_points)

    # 1. Kombinieren der Daten
    roomtemp_data = ruhige_phase

    return roomtemp_data.tolist()

def generate_roomtemp_tsd(n_instances: int, interval_sec: int, time: datetime = datetime.now()) -> list[dict]:
    timestamps = _generate_time_instances(time, interval_sec, n_instances)
    data = _generate_roomtemp_data(
                n_data_points=n_instances,
                const_temp=15,
                stable_deviation=0.5
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
