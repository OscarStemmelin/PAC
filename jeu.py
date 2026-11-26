"""
Les règles du jeu, sans affichage : la carte, l'état d'une manche et sa mise à jour.
main.py (le jeu à l'écran), bots.py et entrainement.py (simulations) s'en servent.

Règles :
  - Le tueur est un peu plus rapide que le survivant.
  - Il regarde dans une direction (celle où il marche). Devant lui, un cône de
    LUMIÈRE (visible, rouge) et un cône d'ATTAQUE plus petit (invisible).
  - Il attaque avec sa touche d'attaque : si le survivant est dans le cône
    d'attaque (et pas derrière un mur), c'est un coup et le tueur reste immobile
    DUREE_ETOURDI s ; sinon il reste immobile DUREE_ATTAQUE_RATEE s.
    Il gagne s'il porte COUPS_POUR_GAGNER coups avant la fin du temps.
  - Murs (gris) : personne ne passe.
  - Fenêtres (jaunes, fines) : on ne passe qu'en sautant (Espace, à côté). Le saut
    va jusqu'au bout et on ne peut pas s'arrêter ; le tueur saute plus lentement.
    Pendant qu'un des deux saute, l'autre ne peut pas sauter cette fenêtre.
    Après PASSAGES_MAX sauts du survivant, elle se ferme pour lui (rouge).
  - Palettes : debout (rouge), tout le monde passe. Le survivant la baisse
    (Espace, à côté) : elle devient verte, bloque tout le monde ; le survivant
    peut la sauter (Espace), le tueur peut la casser (Espace, DUREE_CASSE s).
  - Les durées (sauts, casse, baisser) sont calées sur Dead by Daylight.
  - Vision : seuls les MURS cachent (on voit à travers fenêtres et palettes).
  - MURETS (murs bas) : personne ne passe, mais on voit par-dessus. Exception :
    le tueur ne voit pas un survivant qui MARCHE collé au muret, de l'autre côté
    (il est "accroupi" derrière). Le survivant, lui, voit toujours par-dessus.
    Le survivant voit tout autour de lui (VISION_SURVIVANT = 360°), le tueur
    seulement devant lui (VISION_TUEUR = 140°).
  - Le survivant peut MARCHER (Maj) : plus lent, mais il ne laisse pas de traces.
    En courant, il laisse des traces de pas qui s'effacent en DUREE_TRACE s ;
    seul le tueur les voit.
"""
import random

import pygame

import cartes
from chemin import Grille

# --- Constantes -------------------------------------------------------------
LARGEUR, HAUTEUR = 1000, 700
DUREE_MANCHE = 90              # secondes : le survivant gagne s'il tient jusque-là
COUPS_POUR_GAGNER = 3

VITESSE_SURVIVANT = 85         # pixels par seconde (même rapport que 125 / 150,
VITESSE_TUEUR = 102            # mais tout est plus petit sur la grande carte)
RAYON_SURVIVANT = 10
RAYON_TUEUR = 12

# Durées des actions, calées sur Dead by Daylight (wiki deadbydaylight.wiki.gg).
# Le survivant fait le saut RAPIDE s'il court, le saut LENT s'il marche.
DUREE_SAUT_FENETRE_RAPIDE = 0.5   # survivant, fenêtre, en courant (DbD : fast vault)
DUREE_SAUT_FENETRE_LENT = 1.5     # survivant, fenêtre, en marchant (DbD : slow vault)
DUREE_SAUT_PALETTE_RAPIDE = 1.1   # survivant, palette baissée, en courant
DUREE_SAUT_PALETTE_LENT = 2.0     # survivant, palette baissée, en marchant
DUREE_SAUT_TUEUR = 1.7            # tueur, fenêtre
DUREE_CASSE = 2.34                # tueur, casser une palette baissée
DUREE_BAISSER = 0.5               # survivant : la palette tombe tout de suite, puis il
                                  # reste immobile ce temps-là (pas de chiffre officiel
                                  # trouvé : estimation)
DUREE_ASSOMME = 2.0               # tueur sous la palette quand elle tombe : étourdi (DbD : 2 s)
PORTEE_ASSOMME = 12               # px entre le bord du tueur et la palette pour être assommé
PASSAGES_MAX = 3               # sauts du survivant avant que la fenêtre se ferme
PORTEE_ACTION = 10             # px entre le bord du personnage et l'élément pour agir

LUMIERE_PORTEE, LUMIERE_ANGLE = 60, 40     # cône rouge visible : longueur (px), demi-angle (°)
ATTAQUE_PORTEE, ATTAQUE_ANGLE = 30, 40     # cône d'attaque, plus petit et invisible
DUREE_ETOURDI = 2.7            # immobilité du tueur après un coup réussi
DUREE_ATTAQUE_RATEE = 1.5      # immobilité du tueur après une attaque ratée

VISION_SURVIVANT = 360         # champ de vision (degrés)
VISION_TUEUR = 140
FACTEUR_MARCHE = 0.55          # le survivant qui marche va à 55 % de sa vitesse
COLLE_MURET = 8                # px entre le bord du survivant et le muret pour être caché
TRACE_INTERVALLE = 0.12        # secondes entre deux traces de pas (en courant)
DUREE_TRACE = 2.0              # secondes avant qu'une trace disparaisse

# --- La carte -----------------------------------------------------------------
# Les cartes sont des CSV dans le dossier cartes/ (voir cartes.py, generer_cartes.py).
# Chaque manche tire une carte au hasard ; elle est rangée dans etat["carte"].
# Toutes les cartes font LARGEUR x HAUTEUR. Murs, fenêtres et palettes sont des
# segments épais (geometrie.Segment) : ils peuvent avoir n'importe quel angle.


def start(carte=None):
    """L'état complet d'une manche tient dans un dictionnaire.
    Sans `carte`, on en tire une au hasard parmi celles du dossier cartes/."""
    if carte is None:
        carte = cartes.carte_aleatoire()
    return {
        "carte": carte,
        "survivant": pygame.Vector2(carte["depart_survivant"]),     # copie : on va la modifier
        "tueur": pygame.Vector2(random.choice(carte["departs_tueur"])),
        "regard": pygame.Vector2(1, 0),          # direction où regarde le tueur
        "temps": 0.0,
        "coups": 0,                              # coups portés par le tueur
        "immobile": 0.0,                         # temps d'immobilité restant du tueur
        "casse": None,                           # palette que le tueur est en train de casser
        "baisse": 0.0,                           # immobilité restante du survivant après une palette
        "assomme": None,                         # instant où le tueur a été assommé par une palette
        "saut": {"survivant": None, "tueur": None},   # saut en cours de chacun
        "palettes": ["debout"] * len(carte["palettes"]),   # "debout", "baissee" ou "cassee"
        "passages": [0] * len(carte["fenetres"]),          # sauts du survivant par fenêtre
        "attaque": None,                         # (touché ?, instant) de la dernière attaque
        "marche": False,                         # le survivant marche-t-il (discret) ?
        "traces": [],                            # traces de pas : (position, instant)
        "chrono_trace": 0.0,
        "fini": False,
        "gagnant": None,                         # "tueur" ou "survivant"
    }


# --- Obstacles ---------------------------------------------------------------------
def fenetre_fermee(etat, k):
    return etat["passages"][k] >= PASSAGES_MAX


def obstacles(etat):
    """Ce qui bloque la marche (des deux) : murs, murets, fenêtres, palettes baissées."""
    c = etat["carte"]
    return c["murs"] + c["murets"] + c["fenetres"] + palettes_baissees(etat)


def palettes_baissees(etat):
    return [p for p, e in zip(etat["carte"]["palettes"], etat["palettes"]) if e == "baissee"]


# Pathfinding du tueur : les fenêtres et les palettes baissées sont des cases
# "lentes" (il faut sauter ou casser) au lieu d'être bloquées. Une grille par
# carte et par combinaison de palettes baissées, construite quand on en a besoin.
CASES_TRAVERSEES = 3           # cases "lentes" autour d'une fenêtre / palette (cf. _zone_traversee)


def _lenteur(duree):
    """Facteur de vitesse des cases lentes pour que les traverser coûte autant
    que `duree` secondes d'action (+ la marche normale sur ces cases)."""
    cases_action = duree * VITESSE_TUEUR / 12           # cases parcourues pendant l'action
    return CASES_TRAVERSEES / (CASES_TRAVERSEES + cases_action)


LENTEUR_FENETRE = _lenteur(DUREE_SAUT_TUEUR)     # ~0,17 : sauter coûte cher
LENTEUR_PALETTE = _lenteur(DUREE_CASSE)          # ~0,13 : casser coûte encore plus
_grilles_tueur = {}


def _zone_traversee(element):
    """L'élément épaissi (dans le sens de la traversée), pour couvrir les cases autour."""
    return element.epaissi(12)


def grille_tueur(etat, palettes=None):
    """Grille du tueur pour l'état actuel des palettes (ou pour `palettes`,
    une liste d'états imaginée : "et si je baissais celle-ci ?")."""
    c = etat["carte"]
    baissees = tuple(e == "baissee" for e in (palettes or etat["palettes"]))
    cle = (c["nom"], baissees)
    if cle not in _grilles_tueur:
        lentes = [(_zone_traversee(f), LENTEUR_FENETRE) for f in c["fenetres"]]
        lentes += [(_zone_traversee(p), LENTEUR_PALETTE) for p, b in zip(c["palettes"], baissees) if b]
        _grilles_tueur[cle] = Grille(LARGEUR, HAUTEUR, c["murs"] + c["murets"],
                                     rayon=RAYON_TUEUR, zones_lentes=lentes)
    return _grilles_tueur[cle]


# --- Déplacements ---------------------------------------------------------------
def garder_dans_arene(position, rayon):
    position.x = max(rayon, min(LARGEUR - rayon, position.x))
    position.y = max(rayon, min(HAUTEUR - rayon, position.y))


def sortir_des_obstacles(position, rayon, liste):
    """Si un cercle chevauche un obstacle (segment épais), on le repousse juste au bord.
    Même idée qu'avec un rectangle, mais dans le repère du segment (s le long, t en travers)."""
    for o in liste:
        if not o.boite.inflate(2 * rayon, 2 * rayon).collidepoint(position):
            continue                                     # test rapide : trop loin
        s, t = o.local(position)
        s_proche = min(max(s, 0), o.longueur)
        t_proche = min(max(t, -o.ep / 2), o.ep / 2)
        if s_proche == s and t_proche == t:
            # Le centre est dans l'obstacle : on sort par le côté le plus proche
            sorties = [
                (s + rayon, -o.u), (o.longueur - s + rayon, o.u),
                (t + o.ep / 2 + rayon, -o.n), (o.ep / 2 - t + rayon, o.n),
            ]
            distance, sens = min(sorties, key=lambda c: c[0])
            position += sens * distance
        else:
            # Le centre est dehors : on recule le long de l'écart si le cercle déborde
            ecart = position - o.point(s_proche, t_proche)
            d = ecart.length()
            if 0 < d < rayon:
                position += ecart / d * (rayon - d)


def marcher(etat, qui, direction, dt, facteur=1.0):
    vitesse, rayon = (VITESSE_SURVIVANT, RAYON_SURVIVANT) if qui == "survivant" else (VITESSE_TUEUR, RAYON_TUEUR)
    etat[qui] += direction * vitesse * facteur * dt
    garder_dans_arene(etat[qui], rayon)
    sortir_des_obstacles(etat[qui], rayon, obstacles(etat))


# --- Sauts (fenêtres et palettes) -----------------------------------------------------
def _a_portee(position, rayon, element):
    """Assez près de l'élément, et EN FACE de lui (pas au bout)."""
    if element.distance(position) > rayon + PORTEE_ACTION:
        return False
    s, _ = element.local(position)
    return 0 <= s <= element.longueur


def element_proche(etat, qui):
    """Ce que `qui` peut faire en appuyant sur Espace maintenant :
    ("fenetre", k) / ("palette", k), ou None. Le plus proche s'il y en a plusieurs."""
    position = etat[qui]
    rayon = RAYON_SURVIVANT if qui == "survivant" else RAYON_TUEUR
    autre = etat["saut"]["tueur" if qui == "survivant" else "survivant"]
    possibles = []
    for k, f in enumerate(etat["carte"]["fenetres"]):
        if autre and autre["fenetre"] == k:
            continue                             # l'autre est en train de la sauter
        if qui == "survivant" and fenetre_fermee(etat, k):
            continue
        possibles.append((f, ("fenetre", k)))
    for k, p in enumerate(etat["carte"]["palettes"]):
        e = etat["palettes"][k]
        if e == "baissee" or (e == "debout" and qui == "survivant"):
            possibles.append((p, ("palette", k)))
    possibles = [(position.distance_to(r.centre), el) for r, el in possibles
                 if _a_portee(position, rayon, r)]
    return min(possibles)[1] if possibles else None


def preparer_saut(position, element, rayon):
    """Points de départ et d'arrivée pour traverser `element` en face de soi,
    perpendiculairement à lui (quel que soit son angle)."""
    s, t = element.local(position)
    s = max(rayon, min(s, element.longueur - rayon))     # en face, sans dépasser des bouts
    cote = 1 if t >= 0 else -1                           # de quel côté on est
    d = element.ep / 2 + rayon + 1
    return element.point(s, cote * d), element.point(s, -cote * d)


def duree_saut(etat, qui, est_fenetre):
    """Durée d'un saut : le tueur saute lentement ; le survivant saute vite s'il
    court, lentement s'il marche."""
    if qui == "tueur":
        return DUREE_SAUT_TUEUR
    if est_fenetre:
        return DUREE_SAUT_FENETRE_LENT if etat["marche"] else DUREE_SAUT_FENETRE_RAPIDE
    return DUREE_SAUT_PALETTE_LENT if etat["marche"] else DUREE_SAUT_PALETTE_RAPIDE


def commencer_saut(etat, qui, element, fenetre=None, palette=None):
    """fenetre / palette : numéro de l'élément sauté (l'un des deux)."""
    rayon = RAYON_SURVIVANT if qui == "survivant" else RAYON_TUEUR
    duree = duree_saut(etat, qui, fenetre is not None)
    debut, fin = preparer_saut(etat[qui], element, rayon)
    etat["saut"][qui] = {"debut": debut, "fin": fin, "duree": duree, "t": 0.0,
                         "fenetre": fenetre, "palette": palette}
    if qui == "tueur":
        etat["regard"] = (fin - debut).normalize()


def avancer_saut(etat, qui, dt):
    """Le saut avance tout seul ; renvoie True quand il est fini."""
    saut = etat["saut"][qui]
    saut["t"] = min(saut["t"] + dt, saut["duree"])
    etat[qui] = saut["debut"].lerp(saut["fin"], saut["t"] / saut["duree"])
    if saut["t"] >= saut["duree"]:
        etat["saut"][qui] = None
        if qui == "survivant" and saut["fenetre"] is not None:
            etat["passages"][saut["fenetre"]] += 1
        return True
    return False


# --- Vision ------------------------------------------------------------------------
def accroupi_contre_muret(etat, muret):
    """Le survivant marche, collé à ce muret ?"""
    return etat["marche"] and muret.distance(etat["survivant"]) < RAYON_SURVIVANT + COLLE_MURET


def cache_par_muret(etat, oeil):
    """Le survivant est-il caché à `oeil` par un muret (il marche collé au muret,
    et le muret est entre eux) ?"""
    return any(accroupi_contre_muret(etat, m) and m.coupe(oeil, etat["survivant"])
               for m in etat["carte"]["murets"])


def voit(etat, qui, point, cible_survivant=False):
    """`qui` ("survivant" ou "tueur") voit-il ce point ? Il faut qu'il soit dans
    son champ de vision et qu'aucun MUR ne soit entre les deux (les fenêtres,
    les palettes et les murets ne cachent rien).
    cible_survivant : le point est le survivant -> le tueur ne le voit pas non
    plus s'il est accroupi derrière un muret (il marche collé au muret)."""
    oeil = etat[qui]
    vers = pygame.Vector2(point) - oeil
    champ = VISION_SURVIVANT if qui == "survivant" else VISION_TUEUR
    if champ < 360 and vers.length() > RAYON_TUEUR:
        regard = etat["regard"]                    # seul le tueur a une direction de regard
        ecart = (regard.angle_to(vers) + 180) % 360 - 180
        if abs(ecart) > champ / 2:
            return False
    if any(m.coupe(oeil, point) for m in etat["carte"]["murs"]):
        return False
    return not (cible_survivant and qui == "tueur" and cache_par_muret(etat, oeil))


def voit_adversaire(etat, qui):
    if qui == "tueur":
        return voit(etat, "tueur", etat["survivant"], cible_survivant=True)
    return voit(etat, "survivant", etat["tueur"])


def laisser_traces(etat, direction, dt):
    """En courant, le survivant laisse une trace toutes les TRACE_INTERVALLE s."""
    if direction.length() > 0 and not etat["marche"]:
        etat["chrono_trace"] -= dt
        if etat["chrono_trace"] <= 0:
            etat["chrono_trace"] = TRACE_INTERVALLE
            ecart = pygame.Vector2(random.uniform(-4, 4), random.uniform(-4, 4))
            etat["traces"].append((etat["survivant"] + ecart, etat["temps"]))


def effacer_vieilles_traces(etat):
    """Les traces plus vieilles que DUREE_TRACE disparaissent (à chaque pas de jeu,
    même quand le survivant saute ou est immobile)."""
    while etat["traces"] and etat["temps"] - etat["traces"][0][1] > DUREE_TRACE:
        etat["traces"].pop(0)


# --- Attaque du tueur -------------------------------------------------------------------
def dans_cone(etat, portee, angle):
    """Le survivant est-il dans le cône (portée, demi-angle) devant le tueur,
    sans mur ni palette baissée entre les deux ? Les fenêtres ne protègent pas
    (on peut frapper à travers) ; une palette baissée protège, sauf celle que
    le survivant est en train de sauter (il est dessus : on peut le toucher)."""
    vers = etat["survivant"] - etat["tueur"]
    d = vers.length()
    if d > RAYON_TUEUR + RAYON_SURVIVANT:        # collé : touché dans tous les cas
        if d - RAYON_SURVIVANT > RAYON_TUEUR + portee:
            return False
        ecart = (etat["regard"].angle_to(vers) + 180) % 360 - 180   # entre -180° et 180°
        if abs(ecart) > angle:
            return False
    saut = etat["saut"]["survivant"]
    sautee = etat["carte"]["palettes"][saut["palette"]] if saut and saut["palette"] is not None else None
    caches = etat["carte"]["murs"] + [p for p in palettes_baissees(etat) if p is not sautee]
    return not any(o.coupe(etat["tueur"], etat["survivant"]) for o in caches)


# --- Un pas de jeu ----------------------------------------------------------------
def mettre_a_jour(etat, commande_survivant, commande_tueur, dt):
    """Fait avancer le jeu de dt secondes.
    commande_survivant = (direction, espace, marche)
    commande_tueur     = (direction, espace, attaque)
    Elles viennent du clavier (main) ou d'un bot (bots.py).
    Renvoie True à l'instant où la manche se termine."""
    if etat["fini"]:
        return False

    fenetres, palettes = etat["carte"]["fenetres"], etat["carte"]["palettes"]

    # 1. Le survivant
    direction, espace, etat["marche"] = commande_survivant
    if etat["saut"]["survivant"]:
        avancer_saut(etat, "survivant", dt)          # pas de contrôle pendant un saut
    elif etat["baisse"] > 0:
        etat["baisse"] -= dt                         # immobile juste après avoir baissé une palette
    else:
        element = element_proche(etat, "survivant") if espace else None
        if element and element[0] == "fenetre":
            commencer_saut(etat, "survivant", fenetres[element[1]], fenetre=element[1])
        elif element and etat["palettes"][element[1]] == "debout":
            # La palette tombe tout de suite ; le survivant est ensuite immobile DUREE_BAISSER s
            k = element[1]
            etat["palettes"][k] = "baissee"
            etat["baisse"] = DUREE_BAISSER
            # Tueur trop près (et pas en plein saut) : assommé, sa casse éventuelle est interrompue
            if (not etat["saut"]["tueur"]
                    and palettes[k].distance(etat["tueur"]) < RAYON_TUEUR + PORTEE_ASSOMME):
                etat["immobile"] = max(etat["immobile"], DUREE_ASSOMME)
                etat["casse"] = None
                etat["assomme"] = etat["temps"]
            # Ceux qui sont dessus sont repoussés d'un côté
            sortir_des_obstacles(etat["survivant"], RAYON_SURVIVANT, [palettes[k]])
            if not etat["saut"]["tueur"]:
                sortir_des_obstacles(etat["tueur"], RAYON_TUEUR, [palettes[k]])
        elif element:
            commencer_saut(etat, "survivant", palettes[element[1]], palette=element[1])
        else:
            marcher(etat, "survivant", direction, dt, FACTEUR_MARCHE if etat["marche"] else 1.0)
            laisser_traces(etat, direction, dt)

    # 2. Le tueur
    direction, espace, attaque = commande_tueur
    if etat["saut"]["tueur"]:
        avancer_saut(etat, "tueur", dt)
    elif etat["immobile"] > 0:
        etat["immobile"] -= dt
        if etat["immobile"] <= 0 and etat["casse"] is not None:
            etat["palettes"][etat["casse"]] = "cassee"
            etat["casse"] = None
    elif attaque:
        touche = dans_cone(etat, ATTAQUE_PORTEE, ATTAQUE_ANGLE)
        etat["attaque"] = (touche, etat["temps"])
        if touche:
            etat["coups"] += 1
            etat["immobile"] = DUREE_ETOURDI
        else:
            etat["immobile"] = DUREE_ATTAQUE_RATEE
    else:
        element = element_proche(etat, "tueur") if espace else None
        if element and element[0] == "fenetre":
            commencer_saut(etat, "tueur", fenetres[element[1]], fenetre=element[1])
        elif element:                                 # palette baissée : on la casse
            etat["casse"] = element[1]
            etat["immobile"] = DUREE_CASSE
        else:
            marcher(etat, "tueur", direction, dt)
            if direction.length() > 0:
                etat["regard"] = pygame.Vector2(direction)

    # 3. Fin de manche
    etat["temps"] += dt
    effacer_vieilles_traces(etat)
    if etat["coups"] >= COUPS_POUR_GAGNER:
        etat["fini"], etat["gagnant"] = True, "tueur"
    elif etat["temps"] >= DUREE_MANCHE:
        etat["fini"], etat["gagnant"] = True, "survivant"
    return etat["fini"]
