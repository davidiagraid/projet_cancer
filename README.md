# Projet classification de tumeurs du foie

- MIL.py : Permet de lancer la méthode Multiple Instance learning en CLI. 
    Exemple : python  MIL.py --Val False --Test True --n_runs_test 5  
    Paramètres : 
    - Val (True ou False). Si c'est True, le choix du meilleur K est lancé
    - Test (True ou False). Si c'est True, le Test est lancé avec K fixé à 100 car c'est la meilleure valeur
    - n_runs_val : nombre d'occurences indépendantes pour la validation
    - n_runs_val : Pareil mais pour le test

- sliced_data.ipynb : notebook qui permet de générer les données pour l'inférence sur les données par slices. Laissez vous guider
- tree_classification.ipynb :  notebook qui permet de tester les classifieurs sous forme d'arbre décisionnels.

- extract_and_save.ipynb : Notebook qui permet de convertir les fichiers excel en tenseurs utilisables pour la régression logistique et PARAFAC (données de départ de Laurent Le Brusquet donnée lors du cours de AMDA)

- tensor_analysis.ipynb : Notebook qui contient toute l'analyse tensorielle des données avec notamment PARAFAC et la régression logistique tensorielle