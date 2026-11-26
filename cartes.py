"""
Les cartes : lecture / écriture en CSV et tirage d'une carte au hasard.

Une carte = un fichier CSV dans le dossier cartes/, une ligne par élément :
    type,x1,y1,x2,y2,epaisseur
    mur,146.0,90.0,146.0,155.0,12          segment épais : de (x1, y1) à (x2, y2),
    muret,300.0,140.0,300.0,250.0,12       (muret : mur bas, on voit par-dessus)
    fenetre,146.0,155.0,146.0,195.0,4      n'importe quel angle (y vers le bas)
    palette,212.3,80.1,254.7,122.5,12
    objectif,300,180,,,                    point : seulement x1, y1
    depart_survivant,500,350,,,
    depart_tueur,30,30,,,                  (plusieurs : le tueur part de l'un d'eux au hasard)

Une carte chargée est un dictionnaire :
    {"nom", "murs", "murets", "fenetres", "palettes"   (listes de geometrie.Segment),
     "objectifs", "departs_tueur"            (listes de Vector2),
     "depart_survivant"                      (Vector2)}

Les CSV sont produits par generer_cartes.py.
"""
import csv
import glob
import os
import random

import pygame

from geometrie import Segment

DOSSIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cartes")
COLONNES = ["type", "x1", "y1", "x2", "y2", "epaisseur"]
OBSTACLES = {"mur": "murs", "muret": "murets", "fenetre": "fenetres", "palette": "palettes"}


def carte_vide(nom):
    return {"nom": nom, "murs": [], "murets": [], "fenetres": [], "palettes": [], "objectifs": [],
            "depart_survivant": None, "departs_tueur": []}


def lire_carte(chemin):
    carte = carte_vide(os.path.splitext(os.path.basename(chemin))[0])
    with open(chemin, newline="", encoding="utf-8") as f:
        for ligne in csv.DictReader(f):
            genre = ligne["type"]
            x, y = float(ligne["x1"]), float(ligne["y1"])
            if genre in OBSTACLES:
                b = (float(ligne["x2"]), float(ligne["y2"]))
                carte[OBSTACLES[genre]].append(Segment((x, y), b, float(ligne["epaisseur"])))
            elif genre == "objectif":
                carte["objectifs"].append(pygame.Vector2(x, y))
            elif genre == "depart_survivant":
                carte["depart_survivant"] = pygame.Vector2(x, y)
            elif genre == "depart_tueur":
                carte["departs_tueur"].append(pygame.Vector2(x, y))
            else:
                raise ValueError(f"{chemin} : type inconnu {genre!r}")
    return carte


def ecrire_carte(carte, chemin):
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        ecrivain = csv.writer(f)
        ecrivain.writerow(COLONNES)
        for genre, cle in OBSTACLES.items():
            for o in carte[cle]:
                ecrivain.writerow([genre, round(o.a.x, 1), round(o.a.y, 1),
                                   round(o.b.x, 1), round(o.b.y, 1), o.ep])
        points = ([("objectif", p) for p in carte["objectifs"]]
                  + [("depart_survivant", carte["depart_survivant"])]
                  + [("depart_tueur", p) for p in carte["departs_tueur"]])
        for genre, p in points:
            ecrivain.writerow([genre, round(p.x), round(p.y), "", "", ""])


_cartes = None          # toutes les cartes, lues une seule fois


def toutes_les_cartes():
    global _cartes
    if _cartes is None:
        fichiers = sorted(glob.glob(os.path.join(DOSSIER, "*.csv")))
        if not fichiers:
            raise FileNotFoundError(f"Aucune carte dans {DOSSIER} : lance python generer_cartes.py")
        _cartes = [lire_carte(f) for f in fichiers]
    return _cartes


def carte_aleatoire():
    """Une des cartes du dossier, tirée au hasard."""
    return random.choice(toutes_les_cartes())
