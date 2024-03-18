# Modèle de coloration de manga

## Prérequis

- Python installé
- Librairie pytorch ( https://pytorch.org/ )
- Installer les poids ( https://drive.google.com/file/d/1WPh5zppEproC_YDjSYPmkQ9Z50ikiwIK/view?usp=drive_link )

## Installation

- Cloner le répertoire

## Inférence

Il suffit de mettre l'image en noir et blanc dans le dossier, nommée 'image.jpg' par défaut, et de lancer le fichier `infer.py`. Celui-ci enregistre le sortie sous le nom `image_colorized.png`.

## Entrainement

Pour entrainer l'IA, il faut avoir WandB et pandas d'installé en plus (librairies python).

Par défaut, en lançant `train.py`, l'IA s'entraine de zero et enregistre ses poids à chaque epoch. Si vous souhaitez reprendre l'entrainement à partir de poids préentrainés, il suffit de décommenter les lignes 69 et 70 de `train.py`. Les informations seront affichées sur WandB.
