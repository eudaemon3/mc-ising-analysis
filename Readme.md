# mc-ising-analysis

Das Thema dieses Projekts ist eine Analyse des 2D-Ising-Modells mithilfe von Markov Chain Monte Carlo (MCMC).

## Überblick

Die Analyse besteht aus den folgenden Komponenten:
- `Lattice2D`-Klasse zur Erzeugung eies 2D-Spin-Gitters
- `IsingModel`-Klasse zur Simulation und Analyse des Systems
- `AnalysisResult`-Klasse für die Darstellung der berechneten Resultate.
- Unterstützung für offene und periodische Randbedingungen sowie sweeps über das gesamte Grid.
- Analyse von Magnetisierung, Energie, Suszeptibilität, spezifischer Wärme und Autokorrelation


## Anforderungen

- Python 3.11+
- numpy
- scipy
- matplotlib
- ipykernel
- scienceplots
- numba
- mpi4py


## Beispiel

Das bereitgestellte Skript in `src/main.py` demonstriert die Untersuchung von Phasenübergängen in einem  ferromagnetischen System. 

Es werden zwei unabhängige Simulationsreihen auf einem $64 \times 64$ Gitter über eine Temperaturachse von 12 Messpunkten ($\beta J \in [0.20, 0.75]$) durchgeführt:
1. **System A**: Initialisierung mit $75\%$ Spin-Up-Überschuss (`spin_prob=0.25`).
2. **System B**: Initialisierung mit $75\%$ Spin-Down-Überschuss (`spin_prob=0.75`).

Beide Konfigurationen durchlaufen denselben Temperaturbereich, wodurch das Verschwinden der Magnetisierung beim Überschreiten einer kritischen Temperatur (Curie-Punkt) durch die Simulation korrekt abbgebildet werden kann. 

Bei der Analyse der Zeitreihe werden außerdem noch folgende Punkte berücksichtigt:
* **Burn-In-Phase:** Die ersten $20\%$ der insgesamt $80.000$ Monte-Carlo-Schritte werden verworfen, um sicherzustellen, dass sich das System im thermischen Gleichgewicht befindet.

* **Thinning:** Um die Korrelation aufeinanderfolgender Zustände zu eliminieren, berechnet der Algorithmus auf Basis der Autokorrelationszeit der MEssreihe $\tau_\text{int}$ eine Ausdünnungs-Schrittweite (*Thinning Stride*). Dadurch werden nur statistisch unabhängige Messpunkte in die finale Fehleranalyse mit einbezogen.

## Nutzung
### 1. Standard-Ausführung (Seriell)
Falls kein MPI-Cluster genutzt wird, kann das Skript direkt über den Python-Interpreter gestartet werden.
```bash
python src/main.py
```
### 2. Parallelisierte Ausführung (MPI)
Mit `MPI` ist es zusätzlich möglich die Berechnungen der einzelnen Temperaturen auf die verfügbaren CPU-Kerne aufzuteilen und so die gesamte Anlyse drastisch zu beschleunigen.
```bash
mpiexec -n 4 python src/main.py
```

## Resultate

Im Verzeichnis `results/` sind sind die Resultate für 6 Beispiele angeführt. Diese Grafiken zeigen die kontinuierliche Entwicklung und Optimierung des Projekts an.

* **Beispiel 1: Analyse mit Sweeps** 
Erste Analyse des Ising-Modells und die dazugehörige Zeitreihe einer Messung unter Verwendung von vollständigen Lattice-Sweeps.

* **Beispiel 2: Single-Spin-Updates & Burn-In-Zeit** 
Analyse des Modells und eine spezifische Zeitreihe für eine feste Temperatur mittels Single-Spin-Updates, welche die benötigte Burn-In-Zeit bis zum Erreichen des thermischen Gleichgewichts veranschaulicht.

* **Beispiel 3: Critical Slowing Down**
Darstellung der Autokorrelationszeit $\tau_\text{int}$ in Abhängigkeit von der inversen Temperatur $\beta J = 1/T$. Der Plot zeigt extrem hohe Autokorrelationszeiten direkt am Phasenübergang.

* **Beispiel 4: Observablen-Fehler bei 80k Schritten**
Ising-Modell-Analyse inklusive statistischer Fehlerbalken bei einer Simulationslänge von 80.000 Schritten. Aufgrund der geringen Anzahl an unkorellierten Messpunkten entstehen extrem großen Fehlern im Bereich des Phasenübergangs.

* **Beispiel 5: Block-Jackknife-Resampling**
Erweiterte und genauere Fehlerberechnung der Observablen mittels Jackknife-Resampling.

* **Beispiel 6: Finale Analyse mit 320k Schritten** 
Finale Analyse des Ising Modells mit einer verlängerten Simulationszeit von 320.000 Schritten. Berücksichtigung aller implementierten Features, wie: Lattice Sweeps, Fehlerberechnung mit Jackknife, Thinning zur Dekorrrelation und einer Burn-In-Phase für valide Ergebnisse.