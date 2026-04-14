# Regression_coffe.py :
## 1. Introductions :
### 1. Cas pratique :
Nous avons à étudier les chiffres et données de ventes d'un coffee shop pour étudier les modèles d'IA (Classification et Regression)
Ici on cherche à trouver grace à toutes les infirlations disponibles, le prix du café.

### 2. Nos données :
- https://www.kaggle.com/datasets/navjotkaushal/coffee-sales-dataset
- Notre dataset dispose de 3000 lignes et 11 colonnes. Les colomnes sont hour_of_day,cash_type,money,coffee_name,Time_of_Day,Weekday,Month_name,Weekdaysort,Monthsort,Date,Time

## 2. Notre algorithme :
### 1. Les modèles :
- Baseline:
  - Notre moyenne, c'est notre points de comparaison.
- Linear Regression :
  - On le prend comme point de départ, il fait une ligne droite entre les points.
- SVR : 
  - On le prend comme pour la regression linéaire mais sachant que celui la est capable de faire des formes géométrique sans passer par le bruit.
- Random Forest 
  - On le prend car il s'adapate aux datasets et prend généralement des formes plus complexes et adaptées pour toutes les catégories.
- Gradient Boosting
  - On le prend car il s'entraine et s'adapte à ses échecs et réussites.
- XGBoost
  - On le prend pour "la science" car il est actuellemnt l'un des meilleurs algorithmes de machine learning (Selon https://towardsdatascience.com/), il est une sorte de forme améliorer de Gradient Boosting que je ne saurais expliqué mais il peut prendre en compte des micro paramètres.

### 2. Process :
On utilise uniquement coffee_name et le mois pour trouver le prix, 
utilisation de get_dummies pour ne pas avoir les problèmes de LabelEncoder (hierarchisation fausse)
Enfin on les encodes en ne gardant que les ints.

## 3. Nos résultats :
- Baseline :
  - (RMSE ~4,82, MSE~23,21, MAE~4,15,)
- Linear Regression :
  - Mauvais résultats (RMSE ~1,67, MSE~2,78, MAE~1,45) car il recherche une relation direct (par xemple : Plus il est tard, plus le café est cher)
- SVR :
  - Mauvais résultats (RMSE ~1,68, MSE~2,81, MAE~1,22) pour les mêmes raison que Linear Regression car i lcherche a englobé tout les resultats dans son spectre et aurait du mal avec les différents types de cafés.
- Random Forest
  - Très bon résultats (RMSE ~0,64, MSE~0,41, MAE~0,29) car il crée plein de chemin (arbres) qui essayent de découper les données et d'après donner son avis pour imager le tout (Si c'est X et que le mois est Y alors le prix sera Z)
- Gradient Boosting
  - Très bon résultats (RMSE ~0,65, MSE~0,42, MAE~0,35), ici les arbres ne sont pas indépendants et chaque nouvel arbres essaye de régler le problèmes des autres. Le tuning permet aussi de trouver automatiquement le meilleur ou l'un des meilleurs sets d'hyperparamètres. Il réduit les erreur petit à petit.
    - Tuning trop permisives qui avait laiser les arbres trop complexes -> overfitting.
- XGBoost
  - Très bon résultats (RMSE ~0,79, MSE~0,62, MAE~0,50) Ici la préicsion est la meilleur car il s'inspire de Gradient Boosting mais il a un algorithme plus complexe qui arrives à trouver micro paramètres.
    - Problème d'overfitting (RMSE ~0,74, MSE~0,55, MAE~0,27 avant limitation)

## 4. Les évolutions et nos choix :
- Nous avons décider après avoir vu plusieurs itérations et leurs statistiques de ne garder que le nom du café et le mois pour trouver son prix. En effet, cela simplifie le tout et nous aurions pu faire plus dur en nous basant sur tout sauf le nom car la, il fait plus ou moin du nom = prix. Le mois lui a une ptit importance autour des 11% car il semblerait que le prix ait évolué au curs de l'année. Le jour et l'heure de la journée n'avaient pas d'importance (ce qui nous donnes l'info que pas d'Happy Hour / surge pricing) et ont donc été retiré pour optimiser le tout.
- Nous avons éssayer le tuning sur le Gradient Boosting mais dans notre cas, les changements ne semble pas énormes et font peut être même perdre trop de temps par rapport aux résultats mais cela a été intéressant de voir comment faire.
- XGBoost rend la tache trop complexe pour notre petit dataset, c'est pourquoi ces résultats sont si bas.
- On a utilisé pd.get_dummies car il fallait de base transformer des strings en int. Nous n'avons pas utilisé LabelEncoder car celui aurait converti chaque café en un int ce qui aurait pu entrainer des erreurs comme faires des moyennes de café en les transformant ou crée une hierarchie.
  - pd.get_dummies crée des nouvelle colomne avec un boolean par exemple latte = 1 sur les latte.
  - LabelEncoder est bon pour les données croissantes (ex argent)
-
## 5. Conclusion :
On peut voir que les arbres sont ceux qui donnent les meilleurs résultats, on a fait du tuning sur le Gradient Boosting, XGBoost est trop complexe pour notre petit jeux de données.
On a utilisé K-Fold Cross-Validation pour vérifier si overfitting est cela avait détecté sur Gradiet Boosting. 
Il est assez simple d trouver le prix du café grace au nom et au mois (pour l'inflation)








# Classification_coffe.py :
### 1. Cas pratique :
Nous avons à étudier les chiffres et données de ventes d'un coffee shop pour étudier les modèles d'IA (Classification et Regression)
Ici nous cherchons grace à toutes les autres données, à trouver le nom / type du café.
### 2. Nos données :
- https://www.kaggle.com/datasets/navjotkaushal/coffee-sales-dataset
- Notre dataset dispose de 3000 lignes et 11 colonnes. Les colomnes sont hour_of_day,cash_type,money,coffee_name,Time_of_Day,Weekday,Month_name,Weekdaysort,Monthsort,Date,Time

## 2. Notre algorithme :
### 1. Les modèles :
- Baseline:
  - Accuracy: 0.228
  - Mit en temps que référence comme base
- Decision Tree
  - Accuracy: 0.623
  - Bon score, attention over fitting
- Random Forest
  - Accuracy: 0.641
  -Très stable, moins sensible au bruit que l'Arbre de Décision.
- Gradient Boosting
  - Accuracy: 0.658
  - On note un score parfait (1.00) sur l'Espresso, ce qui suggère que cette boisson a une signature de données très spécifique.
- XGBoost
  - Accuracy: 0.638
  - Performance solide, mais nécessite un réglage d'hyperparamètres.
- Naive Bayes
  - Accuracy: 0.527
  - Moins efficace, car il suppose l'indépendance des variables.

### 2. Process Des données :
### 3. Tuning :
  On a décidé de s'entrainer à faire du tuning sur Gradient Boosting pour voir comment cela marchait pour la classification.

## 3. Nos résultats :
On se rend donc compte que les meilleurs résultats sont ceux de Random Forest, Gradient Boosting et XGBoost, pour finir de les classer il faudrait calculer les performances / couts de chaque IA.
Les meilleurs IA sont celles qui utilisent les arbres.

## 4. Les évolutions et nos choix :
Petit à petit nous avont retiré des features non utiles ou très peu dans notre objectif, nous avons utilisé ce que nous avons appris avec la regression.

## 5. Evolution Possible :
Etendre le tuning à tout les modèles, améliorer le tuning pour que celui-ci se note avec l'accuracy. 

## 6. Conclusion :

