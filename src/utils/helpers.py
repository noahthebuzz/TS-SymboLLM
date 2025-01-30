# Space for helper functions and classes
import numpy as np
import matplotlib.pyplot as plt


def temperature_plot():
    # Parameter
    n_data_points = 300   # Gesamtanzahl der Datenpunkte
    stable_temp = 55      # Durchschnittstemperatur in der stabilen Phase
    stable_deviation = 0.5       # Kleine Abweichung der Temperatur
    unstable_start = 150
    unstable_deviation = 1.5       # Kleine Abweichung der Temperatur
    increase_start = 225  # Punkt, ab dem die Temperatur steigt
    final_temp = 90       # Endtemperatur

    # 1. Ruhige Phase
    ruhige_phase = np.random.normal(loc=stable_temp, scale=stable_deviation, size=unstable_start)

    # 1. Stabile Phase
    stable_phase = np.random.normal(loc=stable_temp, scale=unstable_deviation, size=increase_start-unstable_start)

    # 2. Exponentielle Anstiegsphase
    time_increase = np.arange(n_data_points - increase_start)
    increase_phase = stable_temp + (final_temp - stable_temp) * time_increase ** 2 / (n_data_points - increase_start) ** 2

    # 3. Kombinieren der Daten
    temperature_data = np.concatenate([ruhige_phase, stable_phase, increase_phase])

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
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Hohe Auflösung
    plt.close()  # Schließt den Plot, um Ressourcen zu schonen

    print(f"Der Plot wurde als '{output_file}' gespeichert.")


if __name__ == "__main__":
    temperature_plot()
