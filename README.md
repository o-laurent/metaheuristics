# Readme d'évaluation

Il s'agit du rendu de projet du cours IA308 de l'ENSTA Paris dispensé par Johann Dreo. L'objectif était de compléter un module en implémentant des heuristiques permettant de résoudre un problème d'optimisation - la maximisation du recouvrement d'une zone carrée par des capteurs.

Mon objectif était de faire l'algorithme le plus léger possible. L'algorithme de départ a été complexifié pour permettre son fonctionnement sur des durées plus longues grâce à un redémarrage automatique. Ses performances sont cependant légèrement décevantes sur cette échelle de temps.

## Dépendances

Installez les dépendances avec `pip install -r requirements.txt`.

## Les fichiers intéressants

Les fichiers intéressants sont les suivants:

### Le Code

Les fichiers modifiés sont les suivants:

-   Un fichier contenant les [algorithmes généraux](sho/algo.py) comme le recuit simulé dans les deux encodages ainsi qu'un algorithme génétique (malheureusement pas terminé).
-   Les fichiers contenant les heuristiques d'initialisation (ainsi que la réparation des solutions pour num) et des fonctions _to_sensors_ ainsi que _neighb_square_ optimisées en [bit](sho/bit.py) et [sho/num.py](num). Les optimisations utilisent np.where au lieu de boucler sur le tableau.
-   Des fichiers permettant de définir des [probas d'acceptation](sho/proba.py), des [températures](sho/temp.py) et les [make](sho/make.py) associés.
-   Un dossier définissant les sous-fonctions liées aux [populations](sho/pop_based).

### L'analyse des résultats

Quelques fichiers d'analyse ont été créés:

-   Un fichier permettant de générer des graphes d'[évaluation](evaluation.py), comme des tranches d'EAF (de façon optimisée) ainsi que l'ECDF (version sans enveloppe convexe). L'ECDF est disponible en version heatmap ainsi qu'en version "seaborn", en fait nuage de points.
-   Des fichiers permettant de générer des historiques en faisant varier un paramètre ([bit](variying_sa_cst_bit.py), [num](variying_sa_cst_num.py)) et de les [comparer](compare_encoding.py).
-   Un [fichier](meta_optim.py) optimisant les paramètres du recuit simulé grâce aux Parzen-Tree Estimators de la librairie optuna.
-   Des [figures](sho/analysis/EAF_Slice.png) générées par les algorithmes présentés ci-dessus.

### Remarque

pb.py contient la fonction coverage optimisée par Arthur Liu.
