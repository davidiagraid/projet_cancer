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