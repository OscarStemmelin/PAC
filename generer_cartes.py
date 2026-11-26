"""
Génère les cartes du jeu dans le dossier cartes/ (une carte = un CSV, voir cartes.py).

Lancer :  python generer_cartes.py               (10 cartes, graine 0)
          python generer_cartes.py --n 20 --graine 3

carte_01 est la carte faite à la main ; les autres sont générées ainsi :

1. MODULES : des petits ensembles d'obstacles dessinés à la main (un L, un T,
   une cabane...), tenant dans un carré de 160 x 160 px. Chaque palette ou
   fenêtre y est placée sur une ligne [mur >= L_MIN][élément][mur >= L_MIN] :
   la règle "entre deux murs assez longs" est garantie par construction.
2. ASSEMBLAGE : on pose un module (ou rien) dans chaque case d'une grille
   grossière 4 x 3, tourné d'un angle pris dans ANGLES (0/90/180/270°), avec un éventuel
   miroir et un petit décalage. On essaie plusieurs tirages jusqu'à ce que le
   module laisse un passage assez large avec ses voisins et les bords ; sinon
   la case reste vide. Une carte sur deux est symétrique gauche/droite.
3. VÉRIFICATION : on rejette la carte et on recommence si
   - une zone est inaccessible ;
   - une palette ou une fenêtre n'impose pas un vrai détour quand elle est fermée ;
   - il y a trop peu de palettes ou de fenêtres ;
   - on ne trouve pas de place pour les départs et les 2 objectifs.
"""
import argparse
import math
import os
import random

import pygame

import jeu
from cartes import DOSSIER, carte_vide, ecrire_carte
from chemin import Grille
from geometrie import Segment

V = pygame.Vector2
EP = 12                        # épaisseur des murs et des palettes
EP_FENETRE = 4                 # une fenêtre est un trait fin
OUVERTURE = 40                 # longueur d'une fenêtre
LONGUEUR_PALETTE = 50         # un peu plus longue qu'une fenêtre (40)
L_MIN = 50                     # longueur minimale d'un mur de chaque côté d'une palette / fenêtre
NB_PALETTES = 6                # exactement autant sur chaque carte (nombres pairs :
NB_FENETRES = 6                # les cartes symétriques en ont forcément un nombre pair)
NB_MURETS_MIN = 2              # au moins autant de murets (murs bas) par carte

DEGAGEMENT = 2 * jeu.RAYON_TUEUR + 6     # passage minimal entre deux modules, ou avec un bord
DETOUR_MIN = 80                # px : détour minimal imposé par une palette / fenêtre fermée
DISTANCE_MUR_OBJECTIF = 30     # px : un objectif est loin des murs (on peut le toucher)

# Grille grossière : centres des cases où poser les modules
COLONNES_X = [140, 380, 620, 860]
LIGNES_Y = [125, 350, 575]
DECALAGE_MAX = 20
ESSAIS_PAR_MODULE = 40
# Rotations possibles des modules. Seulement des angles droits : murs horizontaux
# et verticaux. (Le jeu gère n'importe quel angle : mettre par ex. range(0, 360, 15).)
ANGLES = [0, 90, 180, 270]

# --- Les modules ----------------------------------------------------------------------
# Coordonnées locales (centre du module en (0, 0)), y vers le bas.
#   ("ligne", départ, axe, motif) : une ligne droite (axe "x" ou "y") découpée en
#                                   morceaux (genre, longueur) ; départ = sur l'axe du mur
#   ("bloc", x, y, l, h)          : un rocher plein (coin haut-gauche, largeur, hauteur)
MUR_FENETRE = [("mur", 60), ("fenetre", OUVERTURE), ("mur", 60)]          # 160 px
MUR_PALETTE = [("mur", 55), ("palette", LONGUEUR_PALETTE), ("mur", 55)]   # 160 px

MODULES = {
    # Un coin : fenêtre sur un bras, palette sur l'autre
    "L": [("ligne", (-74, -80), "y", MUR_FENETRE),
          ("ligne", (-80, 74), "x", MUR_PALETTE)],
    # Un T : barre avec palette, pied avec fenêtre
    "T": [("ligne", (-80, -60), "x", MUR_PALETTE),
          ("ligne", (50, -60), "y", [("mur", 50), ("fenetre", OUVERTURE), ("mur", 50)])],
    # Deux murs parallèles : un couloir
    "couloir": [("ligne", (-80, -50), "x", MUR_PALETTE),
                ("ligne", (-80, 50), "x", MUR_FENETRE)],
    # Cabane ouverte d'un côté
    "cabane": [("ligne", (-80, -74), "x", MUR_FENETRE),
               ("ligne", (-74, -80), "y", MUR_PALETTE),
               ("ligne", (74, -80), "y", [("mur", 160)])],
    "mur_fenetre": [("ligne", (-80, 0), "x", MUR_FENETRE)],
    "mur_palette": [("ligne", (-80, 0), "x", MUR_PALETTE)],
    # Rocher + palette + mur
    "rocher_palette": [("bloc", -80, -30, 60, 60),
                       ("ligne", (-20, 0), "x", [("palette", LONGUEUR_PALETTE), ("mur", 90)])],
    "rocher": [("bloc", -35, -35, 70, 70)],
    # Murs plus longs (180 px)
    "long_mur_fenetre": [("ligne", (-90, 0), "x", [("mur", 70), ("fenetre", OUVERTURE), ("mur", 70)])],
    "long_mur_palette": [("ligne", (-90, 0), "x", [("mur", 65), ("palette", LONGUEUR_PALETTE), ("mur", 65)])],
    "long_L": [("ligne", (-84, -90), "y", [("mur", 180)]),
               ("ligne", (-90, 84), "x", [("mur", 65), ("palette", LONGUEUR_PALETTE), ("mur", 65)])],
    # Murets (murs bas : on voit par-dessus, on se cache derrière en marchant)
    "muret": [("ligne", (-70, 0), "x", [("muret", 140)])],
    "murets_paralleles": [("ligne", (-60, -40), "x", [("muret", 120)]),
                          ("ligne", (-60, 40), "x", [("muret", 120)])],
    "muret_L": [("ligne", (-64, -70), "y", [("muret", 130)]),
                ("ligne", (-70, 64), "x", [("muret", 130)])],
    "mur_et_muret": [("ligne", (-80, -30), "x", MUR_FENETRE),
                     ("ligne", (-60, 40), "x", [("muret", 120)])],
    "vide": [],
}
POIDS = {"L": 3, "T": 2, "couloir": 2, "cabane": 1, "mur_fenetre": 1,
         "mur_palette": 1, "rocher_palette": 1, "rocher": 1,
         "long_mur_fenetre": 1, "long_mur_palette": 1, "long_L": 1,
         "muret": 2, "murets_paralleles": 1, "muret_L": 1, "mur_et_muret": 1,
         "vide": 1}


def transformer(p, angle, miroir, centre):
    """Point local -> point de la carte : miroir, rotation de `angle` degrés, translation."""
    x, y = p
    if miroir:
        x = -x
    return centre + V(x, y).rotate(angle)


def construire_module(module, angle, miroir, centre):
    """Liste de (genre, Segment) du module placé sur la carte."""
    pieces = []
    for piece in module:
        if piece[0] == "bloc":
            # Un rocher = un segment très épais (sa ligne centrale traverse le bloc)
            _, x, y, l, h = piece
            a = transformer((x, y + h / 2), angle, miroir, centre)
            b = transformer((x + l, y + h / 2), angle, miroir, centre)
            pieces.append(("mur", Segment(a, b, h)))
            continue
        _, depart, axe, motif = piece
        sens = V(1, 0) if axe == "x" else V(0, 1)
        t = 0
        for genre, longueur in motif:
            a = transformer(V(depart) + sens * t, angle, miroir, centre)
            t += longueur
            b = transformer(V(depart) + sens * t, angle, miroir, centre)
            pieces.append((genre, Segment(a, b, EP_FENETRE if genre == "fenetre" else EP)))
    return pieces


def miroir_carte(segment):
    """Le symétrique d'un segment par rapport au milieu de la carte (gauche <-> droite)."""
    return Segment((jeu.LARGEUR - segment.a.x, segment.a.y),
                   (jeu.LARGEUR - segment.b.x, segment.b.y), segment.ep)


def loin_des_bords(segment):
    return all(DEGAGEMENT <= c.x <= jeu.LARGEUR - DEGAGEMENT and
               DEGAGEMENT <= c.y <= jeu.HAUTEUR - DEGAGEMENT for c in segment.coins())


def bien_place(nouvelles, existantes):
    """Le module ne touche aucun autre module et laisse un passage assez large."""
    for _, s in nouvelles:
        if not loin_des_bords(s):
            return False
        if any(s.distance_segment(e) < DEGAGEMENT for e in existantes):
            return False
    return True


def assembler(rng, nom):
    carte = carte_vide(nom)
    symetrique = rng.random() < 0.5
    colonnes = range(2) if symetrique else range(4)
    deja = []                                   # segments déjà posés (autres modules)
    for ci in colonnes:
        for y in LIGNES_Y:
            module = MODULES[rng.choices(list(POIDS), weights=list(POIDS.values()))[0]]
            if not module:
                continue
            for _ in range(ESSAIS_PAR_MODULE):
                centre = V(COLONNES_X[ci] + rng.uniform(-DECALAGE_MAX, DECALAGE_MAX),
                           y + rng.uniform(-DECALAGE_MAX, DECALAGE_MAX))
                angle = rng.choice(ANGLES)
                pieces = construire_module(module, angle, rng.random() < 0.5, centre)
                if symetrique:
                    pieces_et_miroir = pieces + [(g, miroir_carte(s)) for g, s in pieces]
                    # le module doit aussi laisser un passage avec son propre reflet
                    ok = bien_place(pieces, deja) and bien_place(
                        [(g, miroir_carte(s)) for g, s in pieces], deja + [s for _, s in pieces])
                else:
                    pieces_et_miroir, ok = pieces, bien_place(pieces, deja)
                if ok:
                    for genre, s in pieces_et_miroir:
                        carte[genre + "s"].append(s)
                        deja.append(s)
                    break
    for cle, nombre in (("palettes", NB_PALETTES), ("fenetres", NB_FENETRES)):
        ajuster_nombre(carte, cle, nombre, symetrique, rng)
    return carte, symetrique


def ajuster_nombre(carte, cle, nombre, symetrique, rng):
    """Transforme en murs les palettes / fenêtres en trop (au hasard), pour en
    garder exactement `nombre`. Sur une carte symétrique, on les transforme par
    paires (l'élément et son reflet) pour garder la symétrie. S'il en manque,
    on ne fait rien : verifier() rejettera la carte."""
    elements = carte[cle]
    while len(elements) > nombre:
        choisi = rng.choice(elements)
        groupe = [choisi]
        if symetrique:
            reflet = miroir_carte(choisi).centre
            groupe += [e for e in elements if e is not choisi and e.centre.distance_to(reflet) < 1]
        if len(elements) - len(groupe) < nombre:
            return                                   # impossible sans casser la symétrie
        for e in groupe:
            elements.remove(e)
            carte["murs"].append(Segment(e.a, e.b, EP))     # même place, devient un mur


# --- Vérifications ------------------------------------------------------------------------
def obstacles_tous(carte):
    return carte["murs"] + carte["murets"] + carte["fenetres"] + carte["palettes"]


def grille(carte, en_plus=()):
    """Grille du tueur : fenêtres et palettes franchissables (sauf celles `en_plus`)."""
    return Grille(jeu.LARGEUR, jeu.HAUTEUR, carte["murs"] + carte["murets"] + list(en_plus),
                  rayon=jeu.RAYON_TUEUR)


def detour(element, carte, base):
    """Détour (px) imposé au tueur quand `element` est fermé : chemin entre ses deux
    côtés avec l'élément bloquant, moins sans."""
    a, b = element.centre - element.n * 25, element.centre + element.n * 25
    return grille(carte, [element]).distance(a, b) - base.distance(a, b)


def libre(carte, p, marge):
    return (marge <= p.x <= jeu.LARGEUR - marge and marge <= p.y <= jeu.HAUTEUR - marge
            and all(o.distance(p) >= marge for o in obstacles_tous(carte)))


def placer_points(carte, rng, symetrique, base):
    """Départs et objectifs. Renvoie False si on ne trouve pas de place."""
    milieu = V(jeu.LARGEUR / 2, jeu.HAUTEUR / 2)
    candidats = [V(x, y) for x in range(420, 581, 20) for y in range(250, 451, 20)]
    candidats = [p for p in candidats if libre(carte, p, DEGAGEMENT)]
    if not candidats:
        return False
    depart = min(candidats, key=lambda p: p.distance_to(milieu))
    carte["depart_survivant"] = depart
    carte["departs_tueur"] = [V(x, y) for x in (30, jeu.LARGEUR - 30) for y in (30, jeu.HAUTEUR - 30)
                              if libre(carte, V(x, y), jeu.RAYON_TUEUR + 4)]
    if not carte["departs_tueur"]:
        return False

    # Objectifs : loin des murs, atteignables, loin l'un de l'autre et du départ
    carte_d = base.carte_distances(base.case_de(depart))

    def atteignable(p):
        i, j = base.case_de(p)
        return carte_d[j, i] < math.inf

    for _ in range(2000):
        p = V(rng.uniform(40, jeu.LARGEUR - 40), rng.uniform(40, jeu.HAUTEUR - 40))
        q = V(jeu.LARGEUR - p.x, p.y) if symetrique else \
            V(rng.uniform(40, jeu.LARGEUR - 40), rng.uniform(40, jeu.HAUTEUR - 40))
        if (p.distance_to(q) >= 400 and min(p.distance_to(depart), q.distance_to(depart)) >= 200
                and all(libre(carte, o, DISTANCE_MUR_OBJECTIF) and atteignable(o) for o in (p, q))):
            carte["objectifs"] = [V(round(p.x), round(p.y)), V(round(q.x), round(q.y))]
            return True
    return False


def verifier(carte, rng, symetrique):
    """Renvoie None si la carte est jouable, sinon la raison du rejet."""
    if len(carte["palettes"]) != NB_PALETTES or len(carte["fenetres"]) != NB_FENETRES:
        return "pas le bon nombre de palettes / fenêtres"
    if len(carte["murets"]) < NB_MURETS_MIN:
        return "pas assez de murets"
    base = grille(carte)
    if not placer_points(carte, rng, symetrique, base):
        return "pas de place pour les départs / objectifs"
    d = base.carte_distances(base.case_de(carte["depart_survivant"]))
    if ((~base.bloquee) & (d == math.inf)).any():
        return "zone inaccessible"
    for element in carte["palettes"] + carte["fenetres"]:
        if not DETOUR_MIN <= detour(element, carte, base) < math.inf:
            return "palette / fenêtre inutile ou cul-de-sac"
    return None


# --- La carte faite à la main (l'ancienne carte du jeu) -------------------------------------
def carte_main():
    carte = carte_vide("carte_01")

    def rect(genre, x, y, l, h):
        """Rectangle horizontal/vertical (x, y, largeur, hauteur) -> segment épais."""
        if l >= h:
            seg = Segment((x, y + h / 2), (x + l, y + h / 2), h)
        else:
            seg = Segment((x + l / 2, y), (x + l / 2, y + h), l)
        carte[genre + "s"].append(seg)

    def mur(x, y, longueur, vertical):
        rect("mur", x, y, EP, longueur) if vertical else rect("mur", x, y, longueur, EP)

    def mur_fenetre(x, y, longueur, vertical):
        a = (longueur - OUVERTURE) // 2
        d = (EP - EP_FENETRE) // 2
        if vertical:
            rect("mur", x, y, EP, a)
            rect("mur", x, y + a + OUVERTURE, EP, longueur - a - OUVERTURE)
            rect("fenetre", x + d, y + a, EP_FENETRE, OUVERTURE)
        else:
            rect("mur", x, y, a, EP)
            rect("mur", x + a + OUVERTURE, y, longueur - a - OUVERTURE, EP)
            rect("fenetre", x + a, y + d, OUVERTURE, EP_FENETRE)

    def palette(x, y, vertical):
        # Le passage fait 60 px : 5 px de mur, la palette (50 px), 5 px de mur
        bout = (60 - LONGUEUR_PALETTE) // 2
        if vertical:
            rect("mur", x, y, EP, bout)
            rect("palette", x, y + bout, EP, LONGUEUR_PALETTE)
            rect("mur", x, y + bout + LONGUEUR_PALETTE, EP, bout)
        else:
            rect("mur", x, y, bout, EP)
            rect("palette", x + bout, y, LONGUEUR_PALETTE, EP)
            rect("mur", x + bout + LONGUEUR_PALETTE, y, bout, EP)

    mur_fenetre(140, 90, 170, True);  palette(140, 260, True);  mur(140, 320, 160, False)
    mur_fenetre(848, 90, 170, True);  palette(848, 260, True);  mur(700, 320, 160, False)
    mur_fenetre(110, 580, 190, False); palette(300, 580, False); mur(360, 500, 140, True)
    mur_fenetre(700, 580, 190, False); palette(640, 580, False); mur(628, 500, 140, True)
    # Cabane centrale : fenêtre en haut et à droite, mur plein à gauche
    mur_fenetre(420, 230, 160, False); mur(420, 242, 160, True)
    mur_fenetre(568, 242, 160, True)
    rect("mur", 130, 396, 60, 60); palette(190, 420, False); mur(250, 420, 110, False)
    rect("mur", 810, 396, 60, 60); palette(750, 420, False); mur(640, 420, 110, False)
    rect("mur", 470, 90, 60, 40); rect("mur", 470, 600, 60, 50)
    # Deux murets, en haut à gauche et à droite
    rect("muret", 300, 140, 12, 110); rect("muret", 688, 140, 12, 110)
    return carte


# =============================================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Génère les cartes dans le dossier cartes/.")
    p.add_argument("--n", type=int, default=10, help="nombre total de cartes (carte_01 comprise)")
    p.add_argument("--graine", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.graine)
    os.makedirs(DOSSIER, exist_ok=True)

    # carte_01 : la carte faite à la main (on lui cherche juste des objectifs)
    carte = carte_main()
    assert len(carte["palettes"]) == NB_PALETTES and len(carte["fenetres"]) == NB_FENETRES, \
        "carte_01 n'a pas le bon nombre de palettes / fenêtres"
    placer_points(carte, rng, True, grille(carte))
    carte["depart_survivant"] = V(500, 460)
    ecrire_carte(carte, os.path.join(DOSSIER, "carte_01.csv"))
    print("carte_01 : faite à la main")

    for k in range(2, args.n + 1):
        nom = f"carte_{k:02d}"
        essais, rejets = 0, {}
        while True:
            essais += 1
            carte, symetrique = assembler(rng, nom)
            raison = verifier(carte, rng, symetrique)
            if raison is None:
                break
            rejets[raison] = rejets.get(raison, 0) + 1
        ecrire_carte(carte, os.path.join(DOSSIER, nom + ".csv"))
        print(f"{nom} : {'symétrique' if symetrique else 'asymétrique'}, "
              f"{len(carte['palettes'])} palettes, {len(carte['fenetres'])} fenêtres, "
              f"{essais} essai(s)  {rejets if rejets else ''}")
