"""
Croyance bayésienne du tueur : où est probablement le survivant ?

C'est un FILTRE BAYÉSIEN sur une grille (filtre à histogramme / HMM) :
  état x      = (mode, direction, case)
                mode      : 0 = court, 1 = marche (accroupi, discret), 2 = arrêté
                direction : 8 directions
                case      : case (i, j) de la grille
  croyance b  = tableau b[mode, d, j, i] = P(survivant dans cet état)

À chaque pas de PAS_TEMPS secondes :
  1. PRÉDICTION  b⁻ = Σ p(x | x') b(x')    (le survivant a bougé : modèle de transition)
  2. MISE À JOUR b  ∝ p(z | x) · b⁻(x)     (ce que le tueur observe : modèle d'observation)

Modèle de transition :
  - MODE : une chaîne de Markov (TRANSITIONS_MODES) ; il garde le plus souvent le
    même mode d'un pas à l'autre (un survivant accroupi le reste) ;
  - MARCHE ALÉATOIRE PERSISTANTE dans son mode : il garde sa direction avec la
    probabilité PERSISTANCE, sinon tourne de 45° ; il avance d'une case avec une
    probabilité qui dépend de sa vitesse (course, marche, 0 à l'arrêt), plus
    lentement sur une fenêtre ou une palette baissée (il faut sauter) ;
  - il ne traverse pas les murs ni les murets : s'il est bloqué, il change de direction.

Modèle d'observation :
  - le tueur VOIT le survivant      -> toute la croyance sur sa case, son mode et sa direction ;
  - il ne le voit pas               -> les cases visibles sont multipliées par EPSILON
                                       (information négative), puis on normalise ;
    sauf collé derrière un muret    -> un survivant qui marche ou est arrêté y est
                                       INVISIBLE (vraisemblance 1) ; qui court, EPSILON ;
  - il voit une trace posée à τ     -> seules les cases à moins de v·(t − τ) (en
                                       chemin) de la trace restent probables.

Évaluation (le survivant bot contre le tueur bot, en comparant à la vraie position) :
    python croyance.py
"""
import math

import numpy as np
import pygame
from matplotlib.path import Path

import jeu
import vision
from chemin import Grille

PAS_TEMPS = 0.1                # secondes entre deux mises à jour
PERSISTANCE = 0.85             # probabilité de garder sa direction à chaque pas
EPSILON = 0.02                 # P(ne pas le voir | il est dans une case visible)

# Les modes du survivant
COURSE, MARCHE, ARRET = 0, 1, 2
NOMS_MODES = ["course", "marche", "arrêt"]
VITESSE_MODE = [1.0, jeu.FACTEUR_MARCHE, 0.0]      # en fraction de la vitesse de course
# TRANSITIONS_MODES[m][n] = P(mode n au pas suivant | mode m maintenant), par pas de 0,1 s
TRANSITIONS_MODES = np.array([
    [0.97, 0.02, 0.01],        # il court : il continue le plus souvent
    [0.03, 0.95, 0.02],        # il marche
    [0.02, 0.03, 0.95],        # il est arrêté
])
MODES_INITIAUX = [0.4, 0.3, 0.3]    # au départ, on ne sait pas
# P(ne pas le voir | collé derrière un muret, du côté opposé), selon le mode :
# en marche ou arrêté il est accroupi, donc invisible ; en courant on le verrait
INVISIBLE_MURET = np.array([EPSILON, 1.0, 1.0])
DELTA_TRACE = 0.05             # vraisemblance (faible) des cases incompatibles avec une trace
LENTEUR_ELEMENT = 0.35         # sur une fenêtre / palette baissée, il avance 3 fois moins vite

# Les 8 directions, dans l'ordre des angles (0°, 45°, ...), y vers le bas
DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def decaler(a, di, dj):
    """r[j, i] = a[j - dj, i - di] : déplace tout le tableau de (di, dj) cases (0 qui entre par le bord)."""
    r = np.zeros_like(a)
    ny, nx = a.shape
    r[max(dj, 0):ny + min(dj, 0), max(di, 0):nx + min(di, 0)] = \
        a[max(-dj, 0):ny + min(-dj, 0), max(-di, 0):nx + min(-di, 0)]
    return r


_grilles = {}           # nom de la carte -> grille du survivant (murs seuls)


class Croyance:
    def __init__(self, etat):
        self.etat = etat              # la manche suivie
        carte = etat["carte"]
        if carte["nom"] not in _grilles:
            _grilles[carte["nom"]] = Grille(jeu.LARGEUR, jeu.HAUTEUR, carte["murs"] + carte["murets"],
                                            rayon=jeu.RAYON_SURVIVANT)
        self.grille = g = _grilles[carte["nom"]]
        self.taille = g.taille
        # Centres des cases (pour savoir lesquelles sont dans le champ de vision)
        X, Y = np.meshgrid((np.arange(g.nx) + 0.5) * g.taille, (np.arange(g.ny) + 0.5) * g.taille)
        self.centres = np.column_stack((X.ravel(), Y.ravel()))
        # Cases sur chaque fenêtre / palette (là où il faut sauter)
        marge = jeu.RAYON_SURVIVANT + 2
        self.sur_fenetre = [f.distance_grille(X, Y) < marge for f in carte["fenetres"]]
        self.sur_palette = [p.distance_grille(X, Y) < marge for p in carte["palettes"]]
        # Murets : cases où le survivant peut se cacher (collé au muret), et de quel
        # côté du muret elles sont (signe de t dans le repère du muret)
        # (+ une case de marge : la bande "collé" est plus étroite qu'une case, sinon
        # elle ne contiendrait que des cases bloquées)
        coin = jeu.RAYON_SURVIVANT + jeu.COLLE_MURET + g.taille
        self.murets = [(m, m.distance_grille(X, Y) < coin,
                        np.sign((X - m.a.x) * m.n.x + (Y - m.a.y) * m.n.y)) for m in carte["murets"]]
        self._cle = None
        # Vitesse : en courant, une fraction de case par pas
        self.avance = jeu.VITESSE_SURVIVANT * PAS_TEMPS / self.taille

        # Croyance initiale : un Dirac sur le départ du survivant, directions
        # équiprobables, modes selon MODES_INITIAUX
        self.b = np.zeros((3, 8, g.ny, g.nx))
        i, j = g.case_de(pygame.Vector2(carte["depart_survivant"]))
        self.b[:, :, j, i] = np.array(MODES_INITIAUX)[:, None] / 8
        self.chrono = 0.0
        self.vu_avant = None          # (position, instant) de la dernière fois qu'il l'a vu
        self.derniere_trace = -1.0    # instant de la dernière trace prise en compte

    # ----- Les cases franchissables (dépendent des palettes et fenêtres) ----------
    def _preparer(self, etat):
        """Pour chaque direction : probabilité d'avancer depuis chaque case (0 si bloqué)."""
        cle = (tuple(etat["palettes"]), tuple(jeu.fenetre_fermee(etat, k) for k in range(len(etat["passages"]))))
        if cle == self._cle:
            return
        self._cle = cle
        libre = ~self.grille.bloquee.copy()
        lent = np.zeros_like(libre)
        for k, masque in enumerate(self.sur_fenetre):
            if cle[1][k]:
                libre &= ~masque              # fenêtre fermée pour le survivant : un mur
            else:
                lent |= masque
        for k, masque in enumerate(self.sur_palette):
            if etat["palettes"][k] == "baissee":
                lent |= masque                # il faut la sauter
        self.libre = libre
        vitesse = np.where(lent, self.avance * LENTEUR_ELEMENT, self.avance)
        self.proba_avancer = []
        for di, dj in DIRECTIONS:
            arrivee_libre = decaler(libre, -di, -dj)          # la case voisine dans cette direction est libre ?
            possible = libre & arrivee_libre
            if di and dj:                                     # en diagonale : pas de coin coupé
                possible &= decaler(libre, -di, 0) & decaler(libre, 0, -dj)
            pas = vitesse / (math.sqrt(2) if di and dj else 1)  # une diagonale est plus longue
            self.proba_avancer.append(np.where(possible, np.minimum(pas, 1), 0.0))

    # ----- 1. Prédiction ----------------------------------------------------------
    def _bouger(self, b, vitesse):
        """Un pas de marche aléatoire persistante pour un mode donné.
        b : tableau (8, ny, nx) ; vitesse : fraction de la vitesse de course."""
        # a) Tourner : garder la direction, ou tourner de ±45°
        tourne = (1 - PERSISTANCE) / 2
        b = PERSISTANCE * b + tourne * (np.roll(b, 1, axis=0) + np.roll(b, -1, axis=0))
        # b) Avancer d'une case (ou être bloqué par un mur et changer de direction)
        nouveau = np.zeros_like(b)
        rebond = np.zeros_like(b[0])
        for d, (di, dj) in enumerate(DIRECTIONS):
            p = self.proba_avancer[d] * vitesse
            bloque = np.where(p > 0, 0.0, self.avance * vitesse) * b[d]   # voulait avancer, mur devant
            avance = b[d] * p
            nouveau[d] += b[d] - avance - bloque                # reste sur place
            nouveau[d] += decaler(avance, di, dj)               # arrive dans la case voisine
            rebond += bloque
        return nouveau + rebond / 8                             # les bloqués repartent au hasard

    def predire(self, b):
        """Un pas du modèle de transition : changer de mode, puis bouger."""
        # a) Mode : nouveau[n] = Σ_m P(n | m) · b[m]
        b = np.tensordot(TRANSITIONS_MODES.T, b, axes=1)
        # b) Chaque mode bouge à sa vitesse (à l'arrêt, il ne bouge pas)
        return np.stack([self._bouger(b[m], VITESSE_MODE[m]) if VITESSE_MODE[m] > 0 else b[m]
                         for m in range(3)])

    # ----- 2. Mise à jour -------------------------------------------------------------
    def _normaliser(self, etat, visibles):
        total = self.b.sum()
        if total < 1e-12:
            # Toutes les hypothèses ont été éliminées (le modèle s'est trompé) :
            # on repart d'une croyance uniforme sur les cases libres non visibles
            uniforme = (self.libre & ~visibles).astype(float)
            self.b = np.broadcast_to(uniforme, self.b.shape).copy()
            total = self.b.sum()
        self.b /= total

    def observer(self, etat):
        g = self.grille
        # a) Le tueur voit le survivant : Dirac sur sa case, direction d'après son mouvement
        if jeu.voit_adversaire(etat, "tueur"):
            i, j = g.case_de(etat["survivant"])
            self.b[:] = 0
            bouge = self.vu_avant is not None and etat["survivant"].distance_to(self.vu_avant[0]) > 1
            # Le mode se voit : immobile, qui marche (lent, accroupi) ou qui court
            mode = (MARCHE if etat["marche"] else COURSE) if bouge else ARRET
            if bouge:
                angle = (etat["survivant"] - self.vu_avant[0]).as_polar()[1]
                self.b[mode, round(angle / 45) % 8, j, i] = 1
            else:
                self.b[mode, :, j, i] = 1 / 8
            self.vu_avant = (pygame.Vector2(etat["survivant"]), etat["temps"])
            return
        self.vu_avant = None

        # b) Il ne le voit pas : information négative sur les cases qu'il voit
        poly = vision.polygone_vision(etat, "tueur")
        visibles = Path(poly).contains_points(self.centres).reshape(g.ny, g.nx)
        # Derrière un muret (de l'autre côté par rapport au tueur), un survivant
        # qui marche ou est arrêté est accroupi, donc invisible
        cachettes = np.zeros_like(visibles)
        for muret, colle, cote in self.murets:
            _, t_tueur = muret.local(etat["tueur"])
            cachettes |= colle & (cote != np.sign(t_tueur))
        self.b[:, :, visibles & ~cachettes] *= EPSILON
        self.b[:, :, visibles & cachettes] *= INVISIBLE_MURET[:, None, None]

        # c) Traces vues depuis la dernière fois : on prend la plus récente
        nouvelles = [(t, p) for p, t in etat["traces"]
                     if t > self.derniere_trace and jeu.voit(etat, "tueur", p)]
        if nouvelles:
            t, p = max(nouvelles, key=lambda tp: tp[0])
            self.derniere_trace = t
            distances = g.carte_distances(g.case_de(p)) * g.taille        # chemin depuis la trace (px)
            portee = jeu.VITESSE_SURVIVANT * (etat["temps"] - t) + 2 * g.taille
            self.b *= np.where(distances <= portee, 1.0, DELTA_TRACE)[None, None]

        self._normaliser(etat, visibles)

    # ----- Le cycle complet -----------------------------------------------------------
    def mettre_a_jour(self, etat, dt):
        """À appeler à chaque image : fait un pas toutes les PAS_TEMPS secondes."""
        self.chrono -= dt
        while self.chrono <= 0:
            self.chrono += PAS_TEMPS
            self._preparer(etat)
            self.b = self.predire(self.b)
            self.observer(etat)

    # ----- Ce qu'on en tire ----------------------------------------------------------
    def carte(self):
        """P(survivant dans chaque case), en sommant sur les modes et les directions."""
        return self.b.sum(axis=(0, 1))

    def modes(self):
        """P(court), P(marche), P(arrêté)."""
        return self.b.sum(axis=(1, 2, 3))

    def prevoir(self, pas):
        """Croyance sur la position dans `pas` pas de temps, sans nouvelle observation."""
        b = self.b
        for _ in range(pas):
            b = self.predire(b)
        return b.sum(axis=(0, 1))

    def plus_probable(self, carte=None):
        """Centre de la case la plus probable (le mode, ou MAP)."""
        carte = self.carte() if carte is None else carte
        j, i = np.unravel_index(carte.argmax(), carte.shape)
        return self.grille.centre((i, j))

    def entropie(self):
        """En bits : 0 = on sait où il est ; grand = on n'en sait rien."""
        p = self.carte().ravel()
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    def masse_autour(self, position, rayon):
        """Probabilité que le survivant soit à moins de `rayon` px de `position`."""
        carte = self.carte()
        d = np.hypot(self.centres[:, 0] - position.x, self.centres[:, 1] - position.y).reshape(carte.shape)
        return float(carte[d <= rayon].sum())


# =============================================================================
# Évaluation : la croyance comparée à la vraie position du survivant
# =============================================================================
if __name__ == "__main__":
    import argparse
    import random
    import time

    from bots import SurvivantBot, TueurBot

    p = argparse.ArgumentParser(description="Évalue la croyance du tueur en simulation.")
    p.add_argument("--manches", type=int, default=10)
    p.add_argument("--graine", type=int, default=0)
    args = p.parse_args()
    random.seed(args.graine)
    np.random.seed(args.graine)

    DT = 1 / 30
    mesures = []                 # quand le tueur NE voit PAS le survivant
    debut = time.time()
    for _ in range(args.manches):
        etat = jeu.start()
        tueur, survivant, croyance = TueurBot(0.2), SurvivantBot(0.8), Croyance(etat)
        perdu_depuis = None
        while not jeu.mettre_a_jour(etat, survivant.decider(etat, DT), tueur.decider(etat, DT), DT):
            avant = croyance.chrono
            croyance.mettre_a_jour(etat, DT)
            if croyance.chrono <= avant:                  # pas de nouveau pas de filtre
                continue
            if jeu.voit_adversaire(etat, "tueur"):
                perdu_depuis = None
                continue
            perdu_depuis = etat["temps"] if perdu_depuis is None else perdu_depuis
            carte = croyance.carte()
            i, j = croyance.grille.case_de(etat["survivant"])
            libres = int(croyance.libre.sum())
            mesures.append({
                "depuis": etat["temps"] - perdu_depuis,
                "masse_50": croyance.masse_autour(etat["survivant"], 50),
                # même mesure pour une croyance "uniforme" (on ne sait rien) :
                "uniforme_50": min(1.0, math.pi * 50 ** 2 / (libres * croyance.taille ** 2)),
                "log_score": math.log2(max(carte[j, i], 1e-12)),
                "rang": int((carte > carte[j, i]).sum()) + 1,
                "entropie": croyance.entropie(),
            })

    import pandas as pd
    m = pd.DataFrame(mesures)
    m["periode"] = pd.cut(m["depuis"], [0, 1, 3, 6, 12, 1000], right=False,
                          labels=["0-1 s", "1-3 s", "3-6 s", "6-12 s", "12 s +"])
    resume = m.groupby("periode", observed=True).agg(
        pas=("masse_50", "size"),
        masse_50=("masse_50", "mean"),
        uniforme_50=("uniforme_50", "mean"),
        rang_median=("rang", "median"),
        entropie=("entropie", "mean"))
    print(f"{args.manches} manches, {len(m)} pas où le tueur ne voit pas le survivant "
          f"[{time.time() - debut:.0f} s]\n")
    print("Depuis combien de temps il l'a perdu de vue :")
    print(resume.round(3).to_string())
    print("\nmasse_50    : probabilité donnée à la zone de 50 px autour de la vraie position")
    print("uniforme_50 : la même chose pour une croyance qui ne sait rien (référence)")
    print("rang_median : rang de la vraie case parmi les ~4000 (1 = la plus probable)")
