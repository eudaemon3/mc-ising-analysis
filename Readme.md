# mc-ising-analysis-framework

Ein Python-Framework zur Simulation und Analyse des 2D-Ising-Modells mithilfe von Monte-Carlo.

## Überblick

Dieses Projekt bietet:
- eine `Lattice2D`-Klasse zur Erzeugung eies 2D-Spin-Gitters
- eine schlanke `IsingModel`-Klasse zur Simulation und Analyse des Systems
- eine `IsingAnalysisResult`-Dataclass für alle Analyseergebnisse
- eine `IsingResultPlotter`-Klasse für die Darstellung der Resultate
- Unterstützung für offene und periodische Randbedingungen
- Analyse von Magnetisierung, Energie, Suszeptibilität, spezifischer Wärme und Autokorrelation


## Anforderungen

- Python 3.11+
- numpy
- scipy
- matplotlib
- ipykernel
- scienceplots
- numba

## Nutzung

Das Beispiel in `src/main.py` nutzt eine kleine Konfiguration mit zwei Startzuständen: ein Gitter mit 75% Spin up und ein Gitter mit 75% Spin down.
Beide werden über denselben Temperaturbereich analysiert, damit der Übergang zu höheren Temperaturen sichtbar wird.

Die Resultate werden als 2x2-Übersichtsplot, ein zweigeteiltes Autokorrelationsdiagramm und ein Plot für die Autokorrelationszeit erzeugt.
Das Sample-Fenster wird dabei automatisch aus der Autokorrelationszeit bestimmt.

Die wichtigsten Einstiegspunkte sind:

- `IsingModel.run_analysis()`
- `IsingAnalysisResult`
- `IsingResultPlotter`


## Hinweise

Das Projekt ist als Demonstration einer Monte-Carlo-Simulation des 2D-Ising-Modells gedacht und kann leicht erweitert werden, um z.B. weitere Messgrößen, Visualisierungen oder unterschiedliche Gittergrößen zu unterstützen.
