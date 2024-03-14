# Workshop MLops
Ce workshop consiste a partir de 2 scripts pythoin de tranformaton des donne et d'entrainement / selection d'un medoele predictif a construire un pipeline de mise en production d'un service de prevision du DPE d'un etablissement en fonction de paramtres de consommation et d'usage de l'energie.

## MLops
Le MLops consiste a mettre des modeles de machine learning en production.
On passe d'un POC sur un notebook jupyter à un système robuste, stable et controllé de gestion et d'execution des modeles.

Le MLops se joue sur 2 niveaux

la pratique: avec un focus sur la tracabilité des elements (data, model, experiences) et la fiabilit≥é des operations (tests)

les outils. On decompose chaque etape du lifecycle d'un modele et de ses artifacts (dataset) pour en faire une etape a part entiere avec sa prqtiaue et ses plateformes et outils.

A ceci on rajoute:
Une couche d'automatisation des etapes du cycle de vie du modele et des données
et
l'automatisation de l'allocation des ressources alloué a l'exploitation des modeles

## La data
En amont on travaille sur la data. On va donc orchestrer
- le stockage: on passe d'un csv a une base donne ou un data store (BigQuery, postgresql, S3, GCP Storage, ......)
- l'extraction: Airflow
- le nettoyage et le feature engineering: Airflow
- la validation des data (Pydantic, Feast, ...)
- la versionalisation des datasets (CCI/CD)

Toutes les etapes d'obtention et de transformation des données sont controllées et versionnée et automatisée.


## Le modele
De meme, les etapes d'entrainement et de mise en production d'un modele sont automatisées
- entrainement et selection (MLflow)
- le monitoring (Seldon) et la detection de drift

## Les ressources
La mise a disposition ou la mise en veille des instances de calcul, des machines virtuelles, est automatisée
- terraform: IaaS, l'infrastructure est gerée comme du code avec un versionnage et une automatisation
- Kubernetes, Kubeflow

# Le workshop
Ce workshop part donc d'une version POC d'entraînement d'un modele de ML classique et peitt a petit transforme ces scripts locaux en un systeme de production.
### dataset : DPE de l'Ademe
On utilise le dataset de l'Ademe sur les DPE dans le tertiaire.
Ce dataset est accessible publiquement sur le site de l'Ademe.
Un extrait du modele avec 10k enregistrements est notre point de depart.

### Tracabilité des modeles et de leur selection avec MLflow
On part de 2 scripts python, de transformation des données et d'entraînement / selection du modele.

On va d'abord intrduire la tracvabilité de l;entraienemnt des modele avec MLflow.
- installation de MLflow
- log des parametres, metriques, artifact et modele
- autolog
- sauvegarde du modele
- predire a partir du modele

### Creer une API avec FastAPI
On va decoupler la parti sleection / opttimisation du modele de son exploitation.
Pour cela on cree un API avec FAstAPI qui va nous permettre
- de creer un endpoint ou seront envoyé les donnees d/entree
- de valider ces données (Pydantic)
- de charger le modele
- de predire le DPE


## Du Local au Cloud
A ce stade, nous avons donc des data de base, un modele entrainé en local et une API aussi en locale.

Nous allons petit a petit passer nous affranchr du local pour apsser online sur une infrastructure cloud.

### Le modele
Commencons par le modele.
Nous allons mettre le modele optimisé sur une solution de storage de type S3 ou Google Storage.
et modifier ensuite notre API pour qu'elle interroge le modele disponible online.


### L'application FastAPI

Pour deployer l'endpoint Fast API, la bonne prqtieu est de dockeriser l'application pour ...

### Le serveur MLflow
pendant qu'on y est on va aussi dockerizer le serveur MLflow qui devient alors collaboratif

## Une interface
A  ce stade nous pouvons interorger le modele en liogne de commande.
mais ce n'est pas l'approche favorite de tout le mopnde. Une intercace web serait plus adpatée

Sttreamlit offre une façon simple et rapide pour mettre en ligne une application simple ou l'utilisteur peut saisir ses données, et ainsi obtenir une prediction de son DPE.













