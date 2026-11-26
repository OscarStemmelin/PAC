"""
Géométrie des obstacles : le SEGMENT ÉPAIS (un rectangle qui peut être tourné
de n'importe quel angle). Murs, fenêtres, palettes et rochers en sont tous.

Un segment épais est défini par :
    a, b  : les deux bouts de sa ligne centrale
    ep    : son épaisseur
Pour faire les calculs, on se place dans son REPÈRE LOCAL :
    s = position le long du segment (0 en a, longueur en b)      axe u
    t = position en travers (0 sur la ligne centrale)            axe n (normale)
Dans ce repère, le segment est un simple rectangle [0, L] x [-ep/2, ep/2].
"""
import math

import numpy as np
import pygame

V = pygame.Vector2


class Segment:
    def __init__(self, a, b, ep):
        self.a, self.b, self.ep = V(a), V(b), ep
        self.longueur = self.a.distance_to(self.b)
        self.u = (self.b - self.a) / self.longueur        # direction du segment
        self.n = self.u.rotate(90)                        # normale (en travers)
        self.centre = (self.a + self.b) / 2
        # Rectangle pygame qui l'englobe (horizontal) : sert de test rapide
        xs = [c.x for c in self.coins()]
        ys = [c.y for c in self.coins()]
        self.boite = pygame.Rect(math.floor(min(xs)) - 1, math.floor(min(ys)) - 1,
                                 math.ceil(max(xs) - min(xs)) + 3, math.ceil(max(ys) - min(ys)) + 3)

    @property
    def vertical(self):
        """Plutôt vertical (pour les textes, etc.)."""
        return abs(self.u.y) > abs(self.u.x)

    def coins(self):
        e = self.n * self.ep / 2
        return [self.a + e, self.b + e, self.b - e, self.a - e]

    def local(self, p):
        """Coordonnées (s, t) du point p dans le repère du segment."""
        d = V(p) - self.a
        return d.dot(self.u), d.dot(self.n)

    def point(self, s, t):
        """Inverse de local : (s, t) -> point de la carte."""
        return self.a + self.u * s + self.n * t

    def point_proche(self, p):
        """Point du segment épais le plus proche de p (p lui-même s'il est dedans)."""
        s, t = self.local(p)
        return self.point(min(max(s, 0), self.longueur), min(max(t, -self.ep / 2), self.ep / 2))

    def distance(self, p):
        """Distance entre p et le segment épais (0 si p est dedans)."""
        return V(p).distance_to(self.point_proche(p))

    def contient(self, p):
        s, t = self.local(p)
        return 0 <= s <= self.longueur and abs(t) <= self.ep / 2

    def distance_grille(self, X, Y):
        """Même chose que distance, mais pour des tableaux numpy de points (X, Y)."""
        dx, dy = X - self.a.x, Y - self.a.y
        s = dx * self.u.x + dy * self.u.y
        t = dx * self.n.x + dy * self.n.y
        hors_s = np.maximum(np.abs(s - self.longueur / 2) - self.longueur / 2, 0)
        hors_t = np.maximum(np.abs(t) - self.ep / 2, 0)
        return np.hypot(hors_s, hors_t)

    def coupe(self, p, q):
        """Le trajet en ligne droite de p à q traverse-t-il le segment épais ?"""
        if not self.boite.clipline(p, q):                 # test rapide : loin de la boîte
            return False
        s0, t0 = self.local(p)
        s1, t1 = self.local(q)
        entree, sortie = 0.0, 1.0                         # partie de [p, q] à l'intérieur
        for debut, delta, bas, haut in ((s0, s1 - s0, 0, self.longueur),
                                        (t0, t1 - t0, -self.ep / 2, self.ep / 2)):
            if abs(delta) < 1e-12:
                if debut < bas or debut > haut:
                    return False
            else:
                k1, k2 = (bas - debut) / delta, (haut - debut) / delta
                entree, sortie = max(entree, min(k1, k2)), min(sortie, max(k1, k2))
                if entree > sortie:
                    return False
        return True

    def elargi(self, marge):
        """Le même segment, plus long et plus épais de `marge` de chaque côté."""
        return Segment(self.a - self.u * marge, self.b + self.u * marge, self.ep + 2 * marge)

    def epaissi(self, marge):
        """Plus épais de `marge` de chaque côté, même longueur."""
        return Segment(self.a, self.b, self.ep + 2 * marge)

    def distance_segment(self, autre):
        """Distance entre deux segments épais (0 s'ils se touchent ou se croisent)."""
        mes_coins, ses_coins = self.coins(), autre.coins()
        for i in range(4):
            if autre.coupe(mes_coins[i], mes_coins[i - 1]) or self.coupe(ses_coins[i], ses_coins[i - 1]):
                return 0.0
        # Deux formes convexes disjointes : la distance est atteinte en un coin
        return min(min(autre.distance(c) for c in mes_coins), min(self.distance(c) for c in ses_coins))

    def __repr__(self):
        return f"Segment(({self.a.x:.0f}, {self.a.y:.0f}), ({self.b.x:.0f}, {self.b.y:.0f}), {self.ep})"
