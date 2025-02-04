# Space for helper functions and classes
import numpy as np
import matplotlib.pyplot as plt


def simulate_temperature_data(n_data_points: int, stable_temp: float = 55, stable_deviation: float = 0.5, unstable_start: int = 150, unstable_deviation: float = 1.5, increase_start: int = 225, final_temp: float = 90):
    # 1. Ruhige Phase
    ruhige_phase = np.random.normal(loc=stable_temp, scale=stable_deviation, size=unstable_start)

    # 1. Stabile Phase
    stable_phase = np.random.normal(loc=stable_temp, scale=unstable_deviation, size=increase_start-unstable_start)

    # 2. Exponentielle Anstiegsphase
    time_increase = np.arange(n_data_points - increase_start)
    increase_phase = stable_temp + (final_temp - stable_temp) * time_increase ** 2 / (n_data_points - increase_start) ** 2 + np.random.normal(scale=(stable_deviation + unstable_deviation)/2, size=n_data_points - increase_start)

    # 3. Kombinieren der Daten
    temperature_data = np.concatenate([ruhige_phase, stable_phase, increase_phase])

    print(f"temperature_data has {len(temperature_data)} data points.")

    # Plot erstellen
    plt.figure(figsize=(10, 5))
    plt.plot(temperature_data, label="Temperaturverlauf")
    plt.axvline(x=increase_start, color='red', linestyle='--', label='Start des Anstiegs')
    plt.xlabel("Zeit")
    plt.ylabel("Temperatur (°C)")
    plt.title("Temperaturdaten mit stabiler Phase und exponentiellem Anstieg")
    plt.legend()

    # Plot als PNG-Datei speichern
    output_file = "temperaturverlauf_exponentiell.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight')  # Hohe Auflösung
    plt.close()  # Schließt den Plot, um Ressourcen zu schonen

    print(f"Der Plot wurde als '{output_file}' gespeichert.")

    return temperature_data


def generate_data(data: str, n_data_points: int = 300, params: dict = None):
    if data == "temperature":
        if params is not None:
            return simulate_temperature_data(n_data_points, stable_temp=params["stable_temp"], stable_deviation=params["stable_deviation"], unstable_start=params["unstable_start"], unstable_deviation=params["unstable_deviation"], increase_start=params["increase_start"], final_temp=params["final_temp"])
        else:
            return simulate_temperature_data(n_data_points)
    else:
        raise ValueError(f"Daten können für'{data}' nicht generiert werden.")
    


if __name__ == "__main__":
    simulate_temperature_data()
