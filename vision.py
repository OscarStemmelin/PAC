"""
Zone visible par un personnage, pour l'affichage (le brouillard autour).

Méthode : le LANCER DE RAYONS. Depuis l'œil, on lance des rayons dans toutes
les directions du champ de vision (1 par degré). Chaque rayon s'arrête sur le
premier bord de mur qu'il touche. En reliant les points d'arrêt, on obtient le
polygone de ce qui est visible. Seuls les murs arrêtent les rayons (on voit à
travers les fenêtres et les palettes), comme jeu.voit().

Calcul de l'intersection rayon / bord :
  rayon : o + t·d   (t >= 0 : distance)      bord : e1 + u·s   (0 <= u <= 1)
  o + t·d = e1 + u·s  =>  t = (w × s) / (d × s),  u = (w × d) / (d × s),  w = e1 - o
  (a × b = ax·by - ay·bx). On fait tous les rayons et tous les bords d'un coup avec numpy.
"""
import math

import numpy as np
import pygame

import jeu

_bords = {}             # nom de la carte -> tableau des bords de murs (x1, y1, x2, y2)


def bords_murs(carte):
    """Les 4 côtés de chaque mur, plus les bords de l'arène."""
    if carte["nom"] not in _bords:
        L, H = jeu.LARGEUR, jeu.HAUTEUR
        liste = [(0, 0, L, 0), (L, 0, L, H), (L, H, 0, H), (0, H, 0, 0)]
        for mur in carte["murs"]:
            coins = mur.coins()
            for i in range(4):
                a, b = coins[i - 1], coins[i]
                liste.append((a.x, a.y, b.x, b.y))
        _bords[carte["nom"]] = np.array(liste, dtype=float)
    return _bords[carte["nom"]]


def lancer_rayons(origine, angles, bords):
    """Point d'arrêt de chaque rayon (angles en degrés, y vers le bas)."""
    a = np.radians(angles)
    dx, dy = np.cos(a)[:, None], np.sin(a)[:, None]            # rayons en lignes
    x1, y1, x2, y2 = bords.T                                   # bords en colonnes
    sx, sy = x2 - x1, y2 - y1
    wx, wy = x1 - origine.x, y1 - origine.y
    denom = dx * sy - dy * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wx * sy - wy * sx) / denom
        u = (wx * dy - wy * dx) / denom
    touche = (np.abs(denom) > 1e-9) & (t > 1e-6) & (u >= 0) & (u <= 1)
    t = np.where(touche, t, np.inf).min(axis=1)                # premier bord touché
    t = np.minimum(t, math.hypot(jeu.LARGEUR, jeu.HAUTEUR))
    return np.column_stack((origine.x + dx[:, 0] * t, origine.y + dy[:, 0] * t))


def polygone_vision(etat, qui):
    """Liste de points du polygone que `qui` voit."""
    oeil = etat[qui]
    bords = bords_murs(etat["carte"])
    if qui == "survivant":                                     # 360° : un tour complet
        points = lancer_rayons(oeil, np.arange(0, 360, 1.0), bords)
        return [tuple(p) for p in points]
    regard = etat["regard"].as_polar()[1]                      # angle du regard (degrés)
    demi = jeu.VISION_TUEUR / 2
    points = lancer_rayons(oeil, np.arange(regard - demi, regard + demi + 0.5, 1.0), bords)
    return [tuple(oeil)] + [tuple(p) for p in points]          # éventail depuis l'œil
