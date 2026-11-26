"""
Les adversaires contrôlés par l'ordinateur (provisoires, en attendant les IA).
Ils ne trichent pas : ils ne savent que ce qu'ils VOIENT (jeu.voit).

  - TueurBot     : poursuit le survivant s'il le voit ; sinon suit les traces de
                   pas qu'il voit ; sinon utilise sa CROYANCE (croyance.py) :
                   nuage serré -> il vise là où le survivant devrait être quand il
                   y arrivera ; nuage étalé -> il fouille la zone la plus probable.
                   (Avec croyance=False : dernière position vue, puis rondes.)
                   Saute les fenêtres et casse les palettes sur son chemin,
                   attaque dès que le survivant est dans son cône.
  - SurvivantBot : s'il voit le tueur (ou l'a vu il y a moins de DELAI_CALME s),
                   il court et choisit où aller (ou quoi sauter, ou s'il baisse
                   une palette) pour que le tueur ait le plus de chemin à faire.
                   Sinon il MARCHE (discret, pas de traces) vers un objectif.

Les deux ont une méthode decider(etat, dt) appelée à chaque image :
  TueurBot.decider     -> (direction, espace, attaque)
  SurvivantBot.decider -> (direction, espace, marche)
"""
import math
import random

import numpy as np
import pygame

import croyance as croyance_mod
import jeu
from croyance import Croyance
from mob_pathfinding import DIRECTIONS, MobPathfinding, observer

V = pygame.Vector2
FREQ_DECISION = 0.1        # le tueur décide toutes les 0,1 s (pas à chaque image)
ATTEINT = 20               # px : un point est "atteint" quand on est plus près que ça

# Utilisation de la croyance par le tueur bot
SEUIL_ENTROPIE = 6.0       # bits : en dessous, nuage serré -> prédiction ; au-dessus -> traque
PREVISION_MAX = 12         # pas de 0,1 s : on prédit au plus 1,2 s à l'avance
RAYON_ZONE = 8             # cases (~100 px) : taille d'une "zone" pendant la traque
ECHELLE_DISTANCE = 500     # px : une zone 500 px plus loin compte e ≈ 2,7 fois moins


class TueurBot:
    """temperature : 0 = parfait, plus haut = plus maladroit.
    croyance : True = il utilise sa croyance bayésienne (croyance.py) quand il ne
    voit pas le survivant ; False = ancien comportement (dernière position vue, rondes)."""

    def __init__(self, temperature=0.0, croyance=True):
        self.temperature = temperature
        self.utilise_croyance = croyance
        self.croyance = None           # créée au premier appel (il faut l'état de la manche)
        self.cerveau = MobPathfinding()
        self.chrono = 0.0
        self.direction = V(0, 0)
        self.mode = "ronde"            # "poursuite", "traces", "prediction", "traque",
                                       # (sans croyance : "recherche", "ronde")
        self.derniere_vue = None       # où il a vu le survivant pour la dernière fois
        self.derniere_trace = -1.0     # instant de la dernière trace atteinte
        self.ronde = None              # point de ronde visé
        self.but_traque = None         # zone visée pendant la traque (case)

    # ----- Sans croyance : l'ancien comportement --------------------------------------
    def point_de_ronde(self, etat):
        """Un objectif (une fois sur deux) ou une case libre au hasard."""
        if etat["carte"]["objectifs"] and random.random() < 0.5:
            return V(random.choice(etat["carte"]["objectifs"]))
        grille = jeu.grille_tueur(etat)
        j, i = random.choice(np.argwhere(~grille.bloquee))
        return grille.centre((i, j))

    def but_sans_croyance(self, etat):
        moi = etat["tueur"]
        if self.derniere_vue is not None:
            if moi.distance_to(self.derniere_vue) < ATTEINT:
                self.derniere_vue = None           # personne ici : on abandonne
            else:
                self.mode = "recherche"
                return self.derniere_vue
        if self.ronde is None or moi.distance_to(self.ronde) < ATTEINT:
            self.ronde = self.point_de_ronde(etat)
        self.mode = "ronde"
        return self.ronde

    # ----- Avec croyance ---------------------------------------------------------------
    def but_croyance(self, etat):
        """Nuage serré (entropie basse) : PRÉDICTION, on vise l'endroit le plus probable
        au moment où on y arrivera. Nuage étalé : TRAQUE, on vise la zone qui contient
        le plus de probabilité, sans aller trop loin."""
        c, moi = self.croyance, etat["tueur"]
        if c.entropie() < SEUIL_ENTROPIE:
            self.mode, self.but_traque = "prediction", None
            ici = c.plus_probable()
            pas = int(moi.distance_to(ici) / jeu.VITESSE_TUEUR / croyance_mod.PAS_TEMPS)
            return c.plus_probable(c.prevoir(min(pas, PREVISION_MAX))) if pas > 0 else ici

        self.mode = "traque"
        grille = jeu.grille_tueur(etat)
        masse = somme_fenetre(c.carte(), RAYON_ZONE)                # probabilité autour de chaque case
        chemin = grille.carte_distances(grille.case_de(moi)) * grille.taille
        score = np.where(np.isfinite(chemin), masse * np.exp(-chemin / ECHELLE_DISTANCE), 0)
        j, i = np.unravel_index(score.argmax(), score.shape)
        # Hystérésis : on garde la zone visée tant qu'elle reste presque aussi bonne
        # (sinon le tueur hésiterait sans cesse entre deux zones)
        if self.but_traque is not None:
            bj, bi = self.but_traque
            if (score[bj, bi] >= 0.7 * score[j, i]
                    and moi.distance_to(grille.centre((bi, bj))) > ATTEINT):
                j, i = bj, bi
        self.but_traque = (j, i)
        return grille.centre((i, j))

    # ----- Décision ----------------------------------------------------------------------
    def destination(self, etat):
        """Où aller, d'après ce que le tueur voit (et croit)."""
        moi = etat["tueur"]
        if jeu.voit_adversaire(etat, "tueur"):
            self.mode = "poursuite"
            self.derniere_vue = V(etat["survivant"])
            self.but_traque = None
            return etat["survivant"]

        # Traces visibles plus récentes que la dernière atteinte : on va à la plus récente
        traces = [(t, p) for p, t in etat["traces"]
                  if t > self.derniere_trace and jeu.voit(etat, "tueur", p)]
        if traces:
            t, p = max(traces, key=lambda tp: tp[0])
            if moi.distance_to(p) < ATTEINT:
                self.derniere_trace = t
            self.mode = "traces"
            return p

        return self.but_croyance(etat) if self.utilise_croyance else self.but_sans_croyance(etat)

    def decider(self, etat, dt):
        # La croyance est tenue à jour à chaque image (elle fait un pas toutes les 0,1 s)
        if self.utilise_croyance:
            if self.croyance is None or self.croyance.etat is not etat:
                self.croyance = Croyance(etat)
            self.croyance.mettre_a_jour(etat, dt)

        self.chrono -= dt
        if self.chrono > 0:
            return self.direction, False, False
        self.chrono = FREQ_DECISION

        # 1. Attaquer si le survivant est dans le cône d'attaque
        if jeu.dans_cone(etat, jeu.ATTAQUE_PORTEE, jeu.ATTAQUE_ANGLE):
            return self.direction, False, True

        # 2. Suivre le chemin vers la destination (il dépend des palettes baissées)
        but = self.destination(etat)
        cible, longueur = jeu.grille_tueur(etat).guider(etat["tueur"], but)
        s = observer(etat["tueur"], cible, longueur)
        self.direction = DIRECTIONS[self.cerveau.agir(s, self.temperature)]

        # 3. Sauter / casser si l'élément à portée est sur le chemin
        espace = False
        element = jeu.element_proche(etat, "tueur")
        if element and cible != etat["tueur"]:
            obstacle = etat["carte"]["fenetres" if element[0] == "fenetre" else "palettes"][element[1]]
            devant = etat["tueur"] + (cible - etat["tueur"]).normalize() * 3 * jeu.RAYON_TUEUR
            espace = obstacle.coupe(etat["tueur"], devant)
        return self.direction, espace, False


def somme_fenetre(a, r):
    """Pour chaque case, la somme de `a` sur le carré de (2r+1) x (2r+1) cases autour
    (image intégrale : 4 lectures par case au lieu de (2r+1)² additions)."""
    ny, nx = a.shape
    P = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    j0, j1 = np.clip(np.arange(ny) - r, 0, ny), np.clip(np.arange(ny) + r + 1, 0, ny)
    i0, i1 = np.clip(np.arange(nx) - r, 0, nx), np.clip(np.arange(nx) + r + 1, 0, nx)
    return P[np.ix_(j1, i1)] - P[np.ix_(j0, i1)] - P[np.ix_(j1, i0)] + P[np.ix_(j0, i0)]


class SurvivantBot:
    """Un survivant plus ou moins bon selon son niveau (entre 0 et 1).

    En FUITE (il voit le tueur, ou l'a vu il y a moins de DELAI_CALME s), il
    court et compare plusieurs options :
      - courir dans une des 16 directions pendant 0,4 s ;
      - sauter la fenêtre ou la palette à côté de lui ;
      - baisser la palette à côté de lui.
    Note de chaque option : la longueur du chemin que le tueur devra faire,
    depuis l'endroit où le survivant l'a vu en dernier, pour le rejoindre
    (pathfinding du tueur), moins une pénalité près des bords de l'arène.
    Puis il choisit avec un softmax :
      niveau = 1 : réagit toutes les 0,1 s, prend presque toujours la meilleure option
      niveau = 0 : réagit toutes les 1,0 s, choisit presque au hasard

    Au CALME, il marche (pas de traces) vers un objectif et y reste.
    """

    CANDIDATES = [V(1, 0).rotate(22.5 * k) for k in range(16)]
    ANTICIPATION = 0.4          # secondes : jusqu'où il regarde devant lui
    N_PAS = 4                   # la prévision est faite en 4 petits pas (pour ne pas traverser un mur)
    MARGE_MUR = 60              # pixels : en dessous, le bord de l'arène est jugé dangereux
    DELAI_CALME = 4.0           # secondes sans voir le tueur avant de se remettre à marcher

    def __init__(self, niveau):
        self.niveau = niveau
        self.temps_reaction = 0.1 + 0.9 * (1 - niveau)          # secondes entre deux décisions
        self.temperature = 5 + 150 * (1 - niveau) ** 2          # bruit du choix (en pixels)
        self.chrono = 0.0
        self.direction = V(0, 0)
        self.marche = True
        self.menace = None          # où il a vu le tueur pour la dernière fois
        self.vu_a = -math.inf       # quand
        self.objectif = None

    def prevoir(self, etat, direction):
        """Position après ANTICIPATION secondes de course dans cette direction."""
        p = V(etat["survivant"])
        obstacles = jeu.obstacles(etat)
        pas = direction * jeu.VITESSE_SURVIVANT * self.ANTICIPATION / self.N_PAS
        for _ in range(self.N_PAS):
            p += pas
            jeu.garder_dans_arene(p, jeu.RAYON_SURVIVANT)
            jeu.sortir_des_obstacles(p, jeu.RAYON_SURVIVANT, obstacles)
        return p

    def noter(self, p, grille):
        """Plus la note est haute, mieux c'est d'être en p (loin de la menace)."""
        d = grille.distance(self.menace, p)
        if d == math.inf:
            d = 2000                        # le tueur ne peut pas y aller : excellent
        dist_mur = min(p.x, p.y, jeu.LARGEUR - p.x, jeu.HAUTEUR - p.y)
        return d - 2.0 * max(0, self.MARGE_MUR - dist_mur)

    def fuir(self, etat):
        grille = jeu.grille_tueur(etat)
        options = [(self.noter(self.prevoir(etat, c), grille), c, False) for c in self.CANDIDATES]

        element = jeu.element_proche(etat, "survivant")
        if element:
            genre, k = element
            if genre == "palette" and etat["palettes"][k] == "debout":
                # Et si je la baissais ? On regarde avec la grille "palette baissée"
                imaginees = list(etat["palettes"])
                imaginees[k] = "baissee"
                note = self.noter(etat["survivant"], jeu.grille_tueur(etat, imaginees))
            else:
                obstacle = etat["carte"]["fenetres" if genre == "fenetre" else "palettes"][k]
                _, arrivee = jeu.preparer_saut(etat["survivant"], obstacle, jeu.RAYON_SURVIVANT)
                note = self.noter(arrivee, grille)
            options.append((note, V(0, 0), True))

        meilleure = max(o[0] for o in options)
        poids = [math.exp((o[0] - meilleure) / self.temperature) for o in options]   # softmax
        _, direction, espace = random.choices(options, weights=poids)[0]
        return direction, espace

    def se_promener(self, etat):
        """Au calme : marcher vers un objectif (le plus loin de la menace) et y rester."""
        moi = etat["survivant"]
        objectifs = etat["carte"]["objectifs"]
        if self.objectif is None and objectifs:
            if self.menace is None:
                self.objectif = V(random.choice(objectifs))
            else:
                self.objectif = V(max(objectifs, key=lambda o: o.distance_to(self.menace)))
        if self.objectif is None or moi.distance_to(self.objectif) < ATTEINT:
            return V(0, 0)
        cible, _ = jeu.grille_tueur(etat).guider(moi, self.objectif)
        return (cible - moi).normalize() if cible != moi else V(0, 0)

    def decider(self, etat, dt):
        self.chrono -= dt
        if self.chrono > 0:
            return self.direction, False, self.marche     # pas encore réagi
        self.chrono = self.temps_reaction

        if jeu.voit_adversaire(etat, "survivant"):
            self.menace, self.vu_a = V(etat["tueur"]), etat["temps"]

        if self.menace is not None and etat["temps"] - self.vu_a < self.DELAI_CALME:
            self.marche = False
            self.objectif = None                # on en choisira un autre une fois au calme
            self.direction, espace = self.fuir(etat)
        else:
            self.marche = True
            self.direction, espace = self.se_promener(etat), False
        return self.direction, espace, self.marche
