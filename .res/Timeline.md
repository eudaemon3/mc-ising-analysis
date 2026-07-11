# Projektidee – Fortgeschrittene Programmierung in der Physik

## Inhalt
Inhalte bzw. Konzepte welche mein Projekt beinhalten soll:

- **Monte-Carlo** Methoden
- **Optimierung** von einem Algorithmus z.B. durch Implementation auf der GPU(CUDA) oder Parallelisierung 
- Als Ergebnis würde ich gerne eine **physikalisch sinnvolle Größe** berechnen oder ein **System simulieren**.

Außerdem würde ich mich besonders für ein Projekt mit **Monte-Carlo aus der Materialphysik** interessieren.

## Entscheidung: Ising Modell
Monte-Carlo-Simulation eines 2D-Spin-Systems.

Inhalte:
- Metropolis-Algorithmus
- Berechnung von Observablen wie Magnetisierung und Energie
- Markov chain monte carlo
- Paralellisieruing

## Info
Programmiersprache des Projekts ist **Python** und optional **C**.

---

# Meeting 13.04.2026
Implementationen und Aufgaben:
- [x] sweeps hinzufügen, um Korrelation zu minimieren
- [x] **Autokorrelationsanalyse** -> Zeit bis *Thermalisierung*
- [x] statistische Analyse der Daten -> check knive, timing analyse
- [x] Parallelisierung using MPI 

Hilfreiche Ressourcen : [Basic concepts in Computational Physics](https://www.if.ufrj.br/~sandra/MetComp/livros/BasicConceptsInComputationalPhysics-Sticler-Schachinger.pdf)

## Results:
```
example 04: 
time = 80000

beta_j  tau_int(max)  stride  effective_samples
 0.20         2.09      11               5818
 0.25         3.43      18               3555
 0.30         7.31      37               1729
 0.35        18.16      91                703
 0.40        86.28     432                148
 0.45      7517.75   37589                  1
 0.50       128.68     644                 99
 0.55        24.07     121                528
 0.60        10.46      53               1207
 0.65         4.97      25               2560
 0.70         3.69      19               3368
```

```
example 05 (jackknive):
time = 80000

beta_j  tau_int(max)  stride  effective_samples
 0.20         2.15      11               5818
 0.25         3.71      19               3368
 0.30         6.44      33               1939
 0.35        17.96      90                711
 0.40        94.79     474                135
 0.45      1336.72    6684                  9
 0.50       124.46     623                102
 0.55        25.39     127                503
 0.60        34.92     175                365
 0.65        13.02      66                969
 0.70         2.99      15               4266
```
