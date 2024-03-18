# Modèle de coloration de manga

Ce modèle est en cours de développement et n'est pas encore terminé.
Kaggle ayant servi de dataset pour l'entrainement : https://www.kaggle.com/datasets/ultraamvking/colored-manga

L'architecture est basée sur Pix2Pix, dont le papier est à ce lien : https://arxiv.org/abs/1611.07004

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

### Mise en place du dataset

- Installer la libraire opencv
- Mettre les images dans `dataset/rgb/`
- Créer un dosser `dataset/bw/`
- Lancer `utils/setup_dataset.py`

Le dataset est alors prêt à être utilisé.

## Exemples de résultats

De gauche à droite, on a :
- l'image en noir et blanc
- l'image colorée d'origine
- l'image colorée par le modèle

![media_images_Validation Image_2100_8c707e38b91c65e0012c](https://github.com/Aul16/ColorationManga/assets/39156836/f9641e32-5cdd-4674-994a-f5c1498bc33c)


## Limitations

Le dataset en noir et blanc étant crée par un algorithme à partir de celui en rgb pour créer des paires d'images, ce modèle peine à généraliser sur des images de mangas quelconques. Un nouveau modèle d'apprentissage non supervisé est en cours de développement pour régler ce problème.
