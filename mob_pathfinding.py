"""
Le cerveau du mob : il suit le chemin calculé par le pathfinding (chemin.py).
Pas d'apprentissage.

Trois pièces :
  - DIRECTIONS     : numéro d'action -> vecteur de déplacement
  - observer()     : état du jeu (pixels) -> numéro de situation s
  - MobPathfinding : choisit l'action qui suit le chemin, avec un softmax.

La TEMPÉRATURE est le bouton de difficulté :
  température ~ 0  -> le mob prend toujours la bonne direction (parfait)
  température haute -> il se trompe souvent (presque au hasard)
C'est ce réglage que le système bayésien ajustera plus tard.
"""
import math
import numpy as np
import pygame

# --- Les 9 actions ----------------------------------------------------------
diag = 1 / math.sqrt(2)          # pour que les diagonales aient aussi une longueur 1
DIRECTIONS = [
    pygame.Vector2(0, 0),      # 0 : rester
    pygame.Vector2(0, -1),     # 1 : haut
    pygame.Vector2(diag, -diag),   # 2 : haut-droite
    pygame.Vector2(1, 0),      # 3 : droite
    pygame.Vector2(diag, diag),    # 4 : bas-droite
    pygame.Vector2(0, 1),      # 5 : bas
    pygame.Vector2(-diag, diag),   # 6 : bas-gauche
    pygame.Vector2(-1, 0),     # 7 : gauche
    pygame.Vector2(-diag, -diag),  # 8 : haut-gauche
]
N_ACTIONS = len(DIRECTIONS)

# --- Les situations -----------------------------------------------------------
N_ANGLES = 8                 # direction du chemin : 8 parts de 45°
SEUILS_DISTANCE = [100, 300]   # proche < 100 px <= moyen < 300 px <= loin
N_DISTANCES = len(SEUILS_DISTANCE) + 1
N_ETATS = N_ANGLES * N_DISTANCES   # 8 x 3 = 24 situations


def secteur_de(vecteur):
    """Numéro du secteur (0..7) dans lequel pointe un vecteur.
    On décale d'un demi-secteur pour que chaque secteur soit centré sur une
    direction : le secteur 0 = "droite" couvre de -22,5° à +22,5°."""
    largeur = 360 / N_ANGLES
    angle = math.degrees(math.atan2(vecteur.y, vecteur.x))
    return int(((angle + largeur / 2) % 360) // largeur)


def observer(position, cible, longueur):
    """Résume la situation en un entier entre 0 et N_ETATS - 1, à partir du
    CHEMIN vers la proie, donné par le pathfinding (Grille.guider) :
    cible = prochain point du chemin, longueur = longueur du chemin."""
    secteur = secteur_de(cible - position)                # direction du prochain point
    d = longueur
    distance = sum(d >= seuil for seuil in SEUILS_DISTANCE)   # 0 proche, 1 moyen, 2 loin
    return secteur * N_DISTANCES + distance


# Direction centrale de chaque secteur, alignée sur secteur_de() :
# secteur 0 = droite, puis sens horaire à l'écran (y vers le bas).
_DIRECTION_SECTEUR = [DIRECTIONS[a] for a in (3, 4, 5, 6, 7, 8, 1, 2)]
assert len(_DIRECTION_SECTEUR) == N_ANGLES

# Note de chaque action dans chaque secteur : cos(angle entre l'action et le
# chemin) x ECHELLE. Vaut ECHELLE pour la bonne direction, 0 pour "rester",
# -ECHELLE pour le demi-tour. ECHELLE fixe "l'unité" de la température.
ECHELLE = 1.5
NOTES = np.array([[DIRECTIONS[a].dot(d) * ECHELLE for a in range(N_ACTIONS)]
                  for d in _DIRECTION_SECTEUR])          # tableau 8 x 9


class MobPathfinding:
    def agir(self, s, temperature):
        """Choisit l'action qui suit le chemin (softmax sur les notes)."""
        notes = NOTES[s // N_DISTANCES]                  # seul le secteur compte
        if temperature <= 0.01:
            return int(notes.argmax())                   # parfait : toujours la meilleure
        x = notes / temperature
        proba = np.exp(x - x.max())
        proba /= proba.sum()
        return int(np.random.choice(N_ACTIONS, p=proba))
