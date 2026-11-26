"""
Pathfinding : trouver par où passer pour aller d'un point à un autre
en contournant les obstacles. Aucun apprentissage ici, que de l'algorithmique.

Idée :
  1. On découpe l'arène en cases de TAILLE_CASE pixels.
  2. Une case est "bloquée" si un cercle de rayon `rayon` posé en son centre
     toucherait un obstacle (un segment épais, voir geometrie.py, de n'importe quel angle).
  3. Certaines cases sont "lentes" (fenêtre à sauter, palette à casser) : les
     traverser coûte plus cher.
  4. On calcule le plus court chemin entre deux cases (comme Dijkstra :
     les pas en diagonale coûtent √2, les cases lentes plus).
  5. On "lisse" le chemin : le point à viser est le point du chemin le plus
     loin qu'on peut atteindre en ligne droite sans toucher d'obstacle.

Utilisation :
    grille = Grille(LARGEUR, HAUTEUR, murs, rayon=RAYON_TUEUR, zones_lentes=[(zone, 0.2)])
    # murs : liste de Segment ; zone : un Segment (les cases DEDANS sont lentes)
    cible, longueur = grille.guider(position_tueur, position_survivant)
    # cible    : le point vers lequel se diriger maintenant (Vector2)
    # longueur : la longueur du trajet, en pixels
    grille.distance(a, b)   # longueur du plus court chemin de a à b

Démo :  python chemin.py   (le survivant suit la souris, le chemin du tueur est dessiné)
"""
import math
from collections import OrderedDict

import numpy as np
import pygame

TAILLE_CASE = 12                         # pixels par case
MAX_CARTES = 400                         # cartes de distances gardées en mémoire

# Les 8 voisins d'une case : (décalage en colonne, décalage en ligne, coût)
VOISINS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
           (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
           (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]


class Grille:
    def __init__(self, largeur, hauteur, obstacles, rayon, zones_lentes=(), taille_case=TAILLE_CASE):
        """obstacles : liste de Segment. zones_lentes : liste de (Segment, facteur de
        vitesse) ; traverser une case lente coûte 1/facteur fois plus cher."""
        self.taille = taille_case
        self.nx = math.ceil(largeur / taille_case)       # nombre de colonnes
        self.ny = math.ceil(hauteur / taille_case)       # nombre de lignes
        # Pour la ligne de vue : obstacles élargis du rayon (1 px de tolérance pour
        # qu'un personnage collé contre un obstacle puisse "voir" en s'en éloignant)
        self._pour_vue = [o.elargi(rayon - 1) for o in obstacles]

        # Centres de toutes les cases d'un coup (tableaux numpy lignes x colonnes)
        X, Y = np.meshgrid((np.arange(self.nx) + 0.5) * taille_case,
                           (np.arange(self.ny) + 0.5) * taille_case)
        # bloquee[j, i] : un cercle de rayon `rayon` en (i, j) toucherait-il un obstacle ?
        self.bloquee = np.zeros((self.ny, self.nx), dtype=bool)
        for o in obstacles:
            self.bloquee |= o.distance_grille(X, Y) < rayon
        # cout_case[j, i] : 1 normalement, 1/facteur dans une zone lente
        self.cout_case = np.ones((self.ny, self.nx))
        for zone, facteur in zones_lentes:
            dedans = zone.distance_grille(X, Y) == 0
            self.cout_case[dedans] = np.maximum(self.cout_case[dedans], 1 / facteur)

        self._preparer_pas()
        self._cartes = OrderedDict()             # case d'arrivée -> carte des distances

    # ----- Conversions pixels <-> cases ---------------------------------------
    def centre(self, case):
        i, j = case
        return pygame.Vector2((i + 0.5) * self.taille, (j + 0.5) * self.taille)

    def case_de(self, position):
        """Case qui contient la position. Si elle est bloquée (on frôle un
        obstacle), on prend la case libre la plus proche."""
        i = min(max(int(position.x // self.taille), 0), self.nx - 1)
        j = min(max(int(position.y // self.taille), 0), self.ny - 1)
        if not self.bloquee[j, i]:
            return (i, j)
        libres = [(i + di, j + dj) for di in range(-3, 4) for dj in range(-3, 4)
                  if self._dans_grille(i + di, j + dj) and not self.bloquee[j + dj, i + di]]
        if not libres:
            return (i, j)
        return min(libres, key=lambda c: self.centre(c).distance_to(position))

    # ----- Géométrie ------------------------------------------------------------
    def _dans_grille(self, i, j):
        return 0 <= i < self.nx and 0 <= j < self.ny

    def visible(self, a, b):
        """Peut-on aller de a à b en ligne droite sans toucher d'obstacle ?"""
        return not any(o.coupe(a, b) for o in self._pour_vue)

    # ----- Plus court chemin -------------------------------------------------------
    #
    # La "carte des distances" vers une case d'arrivée donne, pour CHAQUE case,
    # la longueur du plus court chemin jusqu'à elle. Aller de n'importe où à
    # cette arrivée, c'est ensuite "descendre la pente".
    #
    # Pour la calculer vite en Python, on traite toute la grille d'un coup avec
    # numpy : à chaque tour, chaque case regarde ses 8 voisines et garde
    # min(sa distance, distance de la voisine + coût du pas). On répète jusqu'à
    # ce que plus rien ne change. Même résultat que Dijkstra, mais en quelques ms.
    # Les cartes sont gardées en mémoire (les MAX_CARTES plus récentes).

    def _preparer_pas(self):
        """Pour chaque direction : tableau du coût du pas depuis chaque case
        (math.inf si on ne peut pas faire ce pas)."""
        libre = np.pad(~self.bloquee, 1, constant_values=False)
        lent = np.pad(self.cout_case, 1, constant_values=1.0)
        ici = (slice(1, self.ny + 1), slice(1, self.nx + 1))
        self._pas = []
        for di, dj, pas in VOISINS:
            la = (slice(1 + dj, self.ny + 1 + dj), slice(1 + di, self.nx + 1 + di))
            possible = libre[ici] & libre[la]
            if di and dj:
                # En diagonale, on interdit de "couper" le coin d'un obstacle
                possible &= libre[1:self.ny + 1, 1 + di:self.nx + 1 + di]
                possible &= libre[1 + dj:self.ny + 1 + dj, 1:self.nx + 1]
            cout = pas * (lent[ici] + lent[la]) / 2      # lenteur moyenne des deux cases
            self._pas.append((di, dj, np.where(possible, cout, math.inf)))

    def _voisins(self, case):
        """Cases voisines accessibles, avec le coût du pas."""
        i, j = case
        return [((i + di, j + dj), cout[j, i]) for di, dj, cout in self._pas
                if cout[j, i] < math.inf]

    def carte_distances(self, arrivee):
        """Tableau d[j, i] = longueur (en cases) du plus court chemin entre la
        case (i, j) et `arrivee` ; math.inf si c'est impossible."""
        if arrivee in self._cartes:
            self._cartes.move_to_end(arrivee)
            return self._cartes[arrivee]

        d = np.full((self.ny, self.nx), math.inf)
        d[arrivee[1], arrivee[0]] = 0.0
        while True:
            dp = np.pad(d, 1, constant_values=math.inf)
            nouveau = d
            for di, dj, cout in self._pas:
                voisine = dp[1 + dj:self.ny + 1 + dj, 1 + di:self.nx + 1 + di]
                nouveau = np.minimum(nouveau, cout + voisine)
            if np.array_equal(nouveau, d):
                break
            d = nouveau

        self._cartes[arrivee] = d
        if len(self._cartes) > MAX_CARTES:
            self._cartes.popitem(last=False)     # on oublie la plus ancienne
        return d

    def plus_court_chemin(self, depart, arrivee):
        """Liste de cases de `depart` à `arrivee` (incluses), ou None si impossible."""
        carte = self.carte_distances(arrivee)
        if carte[depart[1], depart[0]] == math.inf:
            return None
        chemin = [depart]
        case = depart
        while case != arrivee:
            # On descend la pente : le voisin qui minimise (pas + distance restante)
            case = min(self._voisins(case), key=lambda v: v[1] + carte[v[0][1], v[0][0]])[0]
            chemin.append(case)
        return chemin

    # ----- Ce que les bots utilisent ------------------------------------------------
    def distance(self, depart, arrivee):
        """Longueur (en pixels, lenteurs comprises) du chemin de depart à arrivee.
        math.inf si arrivee est inatteignable."""
        i, j = self.case_de(pygame.Vector2(arrivee))
        return self.carte_distances(self.case_de(pygame.Vector2(depart)))[j, i] * self.taille

    def guider(self, depart, arrivee):
        """Renvoie (cible, longueur) :
        cible    = le point à viser maintenant pour aller de depart à arrivee
        longueur = longueur du trajet en pixels (≈ distance à vol d'oiseau si rien ne gêne)
        """
        depart, arrivee = pygame.Vector2(depart), pygame.Vector2(arrivee)

        # Cas simple : rien entre les deux, on va tout droit
        if self.visible(depart, arrivee):
            return arrivee, depart.distance_to(arrivee)

        cases = self.plus_court_chemin(self.case_de(depart), self.case_de(arrivee))
        if cases is None:                    # inatteignable : on vise tout droit
            return arrivee, depart.distance_to(arrivee)

        # Points du chemin en pixels (on remplace la dernière case par la vraie position)
        points = [self.centre(c) for c in cases[1:-1]] + [arrivee]

        # Lissage : on vise le point le plus loin encore visible en ligne droite
        cible = points[0]
        for p in points:
            if self.visible(depart, p):
                cible = p
            else:
                break

        # Longueur : depart -> premier point, puis la suite du chemin
        longueur = depart.distance_to(points[0])
        for p, q in zip(points, points[1:]):
            longueur += p.distance_to(q)
        return cible, longueur


# =============================================================================
# Démo : le survivant suit la souris, clic = déplacer le tueur
# =============================================================================
if __name__ == "__main__":
    import jeu
    import main as affichage

    pygame.init()
    ecran = pygame.display.set_mode((jeu.LARGEUR, jeu.HAUTEUR))
    pygame.display.set_caption("PAC - démo du pathfinding du tueur (clic : déplacer le tueur, N : autre carte)")
    horloge = pygame.time.Clock()
    police = pygame.font.Font(None, 24)

    etat = jeu.start()
    grille = jeu.grille_tueur(etat)
    tueur = pygame.Vector2(etat["carte"]["departs_tueur"][0])
    en_cours = True
    while en_cours:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                en_cours = False
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                tueur = pygame.Vector2(ev.pos)
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_n:
                etat = jeu.start()
                grille = jeu.grille_tueur(etat)
        survivant = pygame.Vector2(pygame.mouse.get_pos())

        ecran.fill(affichage.FOND)
        # Cases bloquées (zone interdite au centre du tueur) en gris foncé
        for j in range(grille.ny):
            for i in range(grille.nx):
                if grille.bloquee[j, i]:
                    pygame.draw.rect(ecran, (45, 45, 55),
                                     (i * grille.taille, j * grille.taille, grille.taille, grille.taille))
        affichage.dessiner_carte(ecran, police, etat)

        # Chemin complet (petits points) et cible lissée (trait jaune)
        debut = pygame.time.get_ticks()
        cases = grille.plus_court_chemin(grille.case_de(tueur), grille.case_de(survivant))
        cible, longueur = grille.guider(tueur, survivant)
        duree = pygame.time.get_ticks() - debut
        for c in cases or []:
            pygame.draw.circle(ecran, (90, 90, 120), grille.centre(c), 2)
        pygame.draw.line(ecran, (240, 200, 60), tueur, cible, 2)

        pygame.draw.circle(ecran, affichage.ROUGE, tueur, jeu.RAYON_TUEUR)
        pygame.draw.circle(ecran, affichage.BLEU, survivant, jeu.RAYON_SURVIVANT)
        texte = (f"longueur du trajet : {longueur:5.0f} px   "
                 f"vol d'oiseau : {tueur.distance_to(survivant):5.0f} px   calcul : {duree} ms")
        ecran.blit(police.render(texte, True, affichage.BLANC), (20, jeu.HAUTEUR - 30))
        pygame.display.flip()
        horloge.tick(60)

    pygame.quit()
