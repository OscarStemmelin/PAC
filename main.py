"""
PAC : un tueur, un survivant. Tu choisis ton rôle, l'ordinateur joue l'autre.

Lancer :  python main.py
Touches :
  S / T          : (menu) jouer le survivant / le tueur
  flèches / ZQSD : bouger
  Espace         : action à côté de toi (sauter une fenêtre ; survivant : baisser
                   ou sauter une palette ; tueur : casser une palette baissée)
  A ou clic      : (tueur) attaquer
  Maj            : (survivant) marcher / courir (en marchant : pas de traces de pas)
  V              : voir toute la carte (pour tester ; normalement, brouillard)
  C              : afficher la croyance du tueur (carte de chaleur : où il pense
                   que le survivant est ; croix blanche = endroit le plus probable)
  R              : nouvelle manche
  + / -          : rendre l'adversaire plus fort / plus faible
  Tab            : retour au menu (changer de rôle)
  Échap          : quitter

Les règles sont dans jeu.py.
"""
import numpy as np
import pygame

import jeu
import vision
from bots import SurvivantBot, TueurBot
from croyance import Croyance

FPS = 60                       # images par seconde visées
TEMPERATURE_INIT = 0.2         # température du tueur bot (0 = parfait)
NIVEAU_INIT = 1                # niveau du survivant bot (entre 0 et 1)

# Couleurs en (Rouge, Vert, Bleu), chaque valeur entre 0 et 255
FOND = (25, 25, 35)
BLEU = (80, 160, 255)
ROUGE = (230, 70, 70)
ROUGE_SOMBRE = (120, 45, 45)
BLANC = (240, 240, 240)
GRIS = (150, 150, 160)
GRIS_OBSTACLE = (95, 95, 110)
MURET = (150, 135, 100)               # muret : mur bas, couleur pierre
MURET_BORD = (95, 85, 60)
JAUNE = (230, 200, 60)
ROUGE_FENETRE = (200, 50, 50)
ROUGE_PALETTE = (200, 50, 50)
VERT_PALETTE = (70, 190, 90)
PALETTE_CASSEE = (70, 60, 50)
LUMIERE = (230, 50, 50, 55)    # rouge transparent (4e valeur = opacité)
OBJECTIF = (200, 160, 255)
RAYON_OBJECTIF = 9
FLASH_ATTAQUE = (240, 220, 60, 150)   # jaune, plus opaque
DUREE_FLASH_ATTAQUE = 0.25            # secondes d'affichage du cône d'attaque
BROUILLARD = (0, 0, 0, 185)           # ce qu'on ne voit pas est assombri
TRACE = (255, 110, 40)                # traces de pas (vues par le tueur)


def direction_clavier(touches):
    """Touches enfoncées -> direction (vecteur de longueur 0 ou 1)."""
    direction = pygame.Vector2(0, 0)
    if touches[pygame.K_LEFT] or touches[pygame.K_q]:
        direction.x -= 1
    if touches[pygame.K_RIGHT] or touches[pygame.K_d]:
        direction.x += 1
    if touches[pygame.K_UP] or touches[pygame.K_z]:
        direction.y -= 1
    if touches[pygame.K_DOWN] or touches[pygame.K_s]:
        direction.y += 1
    if direction.length() > 0:
        direction = direction.normalize()   # sinon on va plus vite en diagonale
    return direction


def difficulte(reglages):
    """Texte décrivant la force de l'adversaire."""
    if reglages["role"] == "survivant":
        return f"Tueur bot : température {reglages['temperature']:.2f}"
    return f"Survivant bot : niveau {reglages['niveau']:.1f}"


def aide_action(etat, qui):
    """Ce que fait Espace maintenant pour `qui` (texte), ou None."""
    if etat["saut"][qui] or (qui == "tueur" and etat["immobile"] > 0):
        return None
    if qui == "survivant" and etat["baisse"] > 0:
        return "Palette baissée : immobile..."
    element = jeu.element_proche(etat, qui)
    if element is None:
        return None
    genre, k = element
    if genre == "fenetre":
        return "ESPACE : sauter la fenêtre"
    if etat["palettes"][k] == "debout":
        return "ESPACE : baisser la palette"
    return "ESPACE : sauter la palette" if qui == "survivant" else "ESPACE : casser la palette"


# =============================================================================
# Dessin (ne modifie rien)
# =============================================================================
def dessiner_carte(ecran, petite_police, etat):
    carte = etat["carte"]
    # Chaque obstacle est un segment épais : on le dessine par ses 4 coins
    for mur in carte["murs"]:
        pygame.draw.polygon(ecran, GRIS_OBSTACLE, mur.coins())
    for muret in carte["murets"]:
        pygame.draw.polygon(ecran, MURET, muret.coins())
        pygame.draw.polygon(ecran, MURET_BORD, muret.coins(), 2)

    for k, fenetre in enumerate(carte["fenetres"]):
        fermee = jeu.fenetre_fermee(etat, k)
        pygame.draw.polygon(ecran, ROUGE_FENETRE if fermee else JAUNE, fenetre.coins())
        # Compteur de sauts, à côté de la fenêtre
        texte = petite_police.render(f"{etat['passages'][k]}/{jeu.PASSAGES_MAX}", True, GRIS)
        ecran.blit(texte, texte.get_rect(center=fenetre.centre + fenetre.n * 18))   # à côté

    couleurs = {"debout": ROUGE_PALETTE, "baissee": VERT_PALETTE, "cassee": PALETTE_CASSEE}
    for k, palette in enumerate(carte["palettes"]):
        e = etat["palettes"][k]
        pygame.draw.polygon(ecran, couleurs[e], palette.coins(), 0 if e != "cassee" else 1)

    # Objectifs : simples repères pour l'instant (pleins quand le survivant les touche)
    for p in carte["objectifs"]:
        touche = p.distance_to(etat["survivant"]) < jeu.RAYON_SURVIVANT + RAYON_OBJECTIF
        pygame.draw.circle(ecran, OBJECTIF, p, RAYON_OBJECTIF, 0 if touche else 2)
        pygame.draw.circle(ecran, OBJECTIF, p, 3)


def dessiner_cone(ecran, etat, portee, angle, couleur, zone_vue=None):
    """Un cône transparent devant le tueur : `portee` px après son bord,
    `angle` degrés de chaque côté de son regard.
    Le cône s'arrête aux murs (on le coupe par la zone que voit le tueur).
    zone_vue : polygone de vision de celui qui regarde ; si donné, on ne dessine
    en plus que la partie du cône qui est dedans."""
    calque = pygame.Surface((jeu.LARGEUR, jeu.HAUTEUR), pygame.SRCALPHA)   # image transparente
    longueur = jeu.RAYON_TUEUR + portee
    points = [etat["tueur"]] + [etat["tueur"] + etat["regard"].rotate(a) * longueur
                                for a in range(-angle, angle + 1, 5)]
    pygame.draw.polygon(calque, couleur, points)
    # Masques : blanc opaque dans une zone, transparent ailleurs. Multiplier le
    # calque par un masque efface le cône hors de la zone.
    for zone in (vision.polygone_vision(etat, "tueur"), zone_vue):
        if zone is not None:
            masque = pygame.Surface((jeu.LARGEUR, jeu.HAUTEUR), pygame.SRCALPHA)
            pygame.draw.polygon(masque, (255, 255, 255, 255), zone)
            calque.blit(masque, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    ecran.blit(calque, (0, 0))


def dessiner_lumiere(ecran, etat, zone_vue=None):
    """Le cône rouge devant le tueur (fait partie de son corps)."""
    dessiner_cone(ecran, etat, jeu.LUMIERE_PORTEE, jeu.LUMIERE_ANGLE, LUMIERE, zone_vue)


def dessiner_attaque(ecran, etat, zone_vue=None):
    """Le cône d'attaque en jaune, seulement pendant DUREE_FLASH_ATTAQUE s après une attaque."""
    if etat["attaque"] and etat["temps"] - etat["attaque"][1] < DUREE_FLASH_ATTAQUE:
        dessiner_cone(ecran, etat, jeu.ATTAQUE_PORTEE, jeu.ATTAQUE_ANGLE, FLASH_ATTAQUE, zone_vue)


def dessiner_traces(ecran, etat, tout_voir=False):
    """Traces de pas que le tueur voit (champ de vision, pas derrière un mur),
    de plus en plus pâles (fondues dans le sol) en vieillissant."""
    for p, instant in etat["traces"]:
        if not (tout_voir or jeu.voit(etat, "tueur", p)):
            continue
        frais = 1 - (etat["temps"] - instant) / jeu.DUREE_TRACE      # 1 = toute fraîche, 0 = effacée
        frais = min(max(frais, 0.0), 1.0)                             # sinon couleur hors de 0..255
        couleur = [round(f + (c - f) * frais) for c, f in zip(TRACE, FOND)]
        pygame.draw.circle(ecran, couleur, p, 2)


def dessiner_brouillard(ecran, etat, qui, zone_vue):
    """Assombrit tout ce que `qui` ne voit pas (zone_vue : polygone de vision.py)."""
    calque = pygame.Surface((jeu.LARGEUR, jeu.HAUTEUR), pygame.SRCALPHA)
    calque.fill(BROUILLARD)
    pygame.draw.polygon(calque, (0, 0, 0, 0), zone_vue)                # on "perce" la zone vue
    if qui == "tueur":
        pygame.draw.circle(calque, (0, 0, 0, 0), etat["tueur"], 3 * jeu.RAYON_TUEUR)   # autour de lui
    ecran.blit(calque, (0, 0))


def dessiner_tueur(ecran, petite_police, etat):
    if etat["immobile"] > 0:
        # Tueur immobile (coup, attaque ratée, casse) : plus sombre, anneau qui se vide
        pygame.draw.circle(ecran, ROUGE_SOMBRE, etat["tueur"], jeu.RAYON_TUEUR)
        part = min(etat["immobile"] / jeu.DUREE_ETOURDI, 1)
        cadre = pygame.Rect(0, 0, 2 * jeu.RAYON_TUEUR + 8, 2 * jeu.RAYON_TUEUR + 8)
        cadre.center = etat["tueur"]
        pygame.draw.arc(ecran, BLANC, cadre, 1.5708, 1.5708 + 6.2832 * part, 2)
    else:
        pygame.draw.circle(ecran, ROUGE, etat["tueur"], jeu.RAYON_TUEUR)
    # Résultat de la dernière attaque, ou palette reçue, pendant 0,8 s
    message = None
    if etat["attaque"] and etat["temps"] - etat["attaque"][1] < 0.8:
        message = "Touché !" if etat["attaque"][0] else "Raté"
    if etat["assomme"] is not None and etat["temps"] - etat["assomme"] < 0.8:
        message = "Assommé !"
    if message:
        texte = petite_police.render(message, True, BLANC)
        ecran.blit(texte, texte.get_rect(midbottom=(etat["tueur"].x, etat["tueur"].y - 18)))


def dessiner_croyance(ecran, petite_police, croyance):
    """Carte de chaleur de la croyance : transparent = improbable, jaune puis
    rouge = de plus en plus probable. Croix blanche : la case la plus probable."""
    carte = croyance.carte()
    v = np.sqrt(carte / carte.max())          # racine : pour voir aussi les faibles probabilités
    ny, nx = v.shape
    image = pygame.Surface((nx, ny), pygame.SRCALPHA)   # 1 pixel par case, agrandie ensuite
    rgb = pygame.surfarray.pixels3d(image)               # tableaux (x, y) : on transpose
    alpha = pygame.surfarray.pixels_alpha(image)
    rgb[..., 0] = 255
    rgb[..., 1] = (220 * (1 - v)).T.astype(np.uint8)     # jaune -> rouge
    rgb[..., 2] = 0
    alpha[:] = (190 * v).T.astype(np.uint8)
    del rgb, alpha                                       # libère l'image pour pouvoir l'afficher
    ecran.blit(pygame.transform.smoothscale(image, (nx * croyance.taille, ny * croyance.taille)), (0, 0))

    x, y = croyance.plus_probable()
    pygame.draw.line(ecran, BLANC, (x - 6, y - 6), (x + 6, y + 6), 2)
    pygame.draw.line(ecran, BLANC, (x - 6, y + 6), (x + 6, y - 6), 2)
    modes = "  ".join(f"{nom} {p:.0%}" for nom, p in zip(("court", "marche", "arrêté"), croyance.modes()))
    texte = petite_police.render(f"Croyance du tueur : entropie {croyance.entropie():.1f} bits   {modes}",
                                 True, BLANC)
    ecran.blit(texte, (jeu.LARGEUR - texte.get_width() - 20, 50))


def dessiner(ecran, police, petite_police, etat, reglages, croyance=None):
    moi = reglages["role"]
    autre = "tueur" if moi == "survivant" else "survivant"
    tout_voir = reglages.get("tout_voir", False)
    ecran.fill(FOND)                                   # on efface tout à chaque image

    # 1. Ce qui est "au sol" : seulement ce qu'on voit
    zone_vue = None if tout_voir else vision.polygone_vision(etat, moi)
    if moi == "tueur" or tout_voir:
        dessiner_traces(ecran, etat, tout_voir)        # seul le tueur voit les traces
    # Le cône fait partie du corps du tueur : le survivant n'en voit que la
    # partie dans son champ de vision (le tueur, lui, voit toujours le sien)
    zone_cone = zone_vue if moi == "survivant" else None
    dessiner_lumiere(ecran, etat, zone_cone)
    dessiner_attaque(ecran, etat, zone_cone)
    if not tout_voir:
        dessiner_brouillard(ecran, etat, moi, zone_vue)

    # 2. La carte (on la connaît, elle est toujours affichée)
    dessiner_carte(ecran, petite_police, etat)

    if croyance is not None and reglages.get("croyance"):
        dessiner_croyance(ecran, petite_police, croyance)

    # 3. Les personnages : toi, et l'autre seulement si tu le vois
    for qui in (moi, autre):
        if qui == autre and not (tout_voir or jeu.voit_adversaire(etat, moi)):
            continue
        if qui == "survivant":
            pygame.draw.circle(ecran, BLEU, etat["survivant"], jeu.RAYON_SURVIVANT)
        else:
            dessiner_tueur(ecran, petite_police, etat)

    # Coups reçus (en haut à gauche) et chrono (en haut à droite)
    for i in range(jeu.COUPS_POUR_GAGNER):
        plein = i < etat["coups"]
        pygame.draw.circle(ecran, ROUGE, (30 + 30 * i, 28), 10, 0 if plein else 2)
    restant = max(0, jeu.DUREE_MANCHE - etat["temps"])
    texte = police.render(f"{restant:4.1f} s", True, BLANC)
    ecran.blit(texte, (jeu.LARGEUR - texte.get_width() - 20, 15))

    # Ce que fait Espace pour toi en ce moment
    aide = aide_action(etat, moi)
    if aide:
        texte = petite_police.render(aide, True, BLANC)
        ecran.blit(texte, texte.get_rect(midbottom=(etat[moi].x, etat[moi].y - 18)))
    if moi == "survivant":
        allure = "Marche (discret)" if etat["marche"] else "Course (traces)"
        if any(jeu.accroupi_contre_muret(etat, m) for m in etat["carte"]["murets"]):
            allure += " - accroupi contre un muret"
        texte = petite_police.render(f"{allure}   Maj : changer", True, BLANC)
        ecran.blit(texte, (130, 20))

    infos = (f"Carte : {etat['carte']['nom']}   Tu joues : {moi}   {difficulte(reglages)}   "
             f"Manches : {reglages['manches']}   (C : croyance, V : tout voir, Tab : menu)")
    ecran.blit(petite_police.render(infos, True, GRIS), (20, jeu.HAUTEUR - 25))

    if etat["fini"]:
        gagne = etat["gagnant"] == reglages["role"]
        message = "Gagné !" if gagne else "Perdu..."
        detail = "le tueur a porté 3 coups" if etat["gagnant"] == "tueur" else "le survivant a tenu"
        texte = police.render(f"{message} ({detail})   R pour rejouer", True, BLANC)
        ecran.blit(texte, texte.get_rect(center=(jeu.LARGEUR / 2, jeu.HAUTEUR / 2)))

    pygame.display.flip()                              # affiche l'image construite


def dessiner_menu(ecran, police, petite_police):
    ecran.fill(FOND)
    lignes = [
        (police, "PAC", BLANC),
        (police, "", BLANC),
        (police, "S : jouer le survivant (bleu)", BLEU),
        (police, "T : jouer le tueur (rouge)", ROUGE),
        (police, "", BLANC),
        (petite_police, f"Le tueur doit toucher {jeu.COUPS_POUR_GAGNER} fois le survivant "
                        f"en {jeu.DUREE_MANCHE} s (attaque : A ou clic).", GRIS),
        (petite_police, "Fenêtres jaunes : Espace pour sauter. Fermées au survivant "
                        f"après {jeu.PASSAGES_MAX} sauts.", GRIS),
        (petite_police, "Palettes rouges : le survivant les baisse (Espace), elles deviennent vertes.", GRIS),
        (petite_police, "Palette verte : le survivant la saute, le tueur la casse (Espace).", GRIS),
        (petite_police, f"On ne voit pas à travers les murs. Survivant : {jeu.VISION_SURVIVANT}°, "
                        f"tueur : {jeu.VISION_TUEUR}°.", GRIS),
        (petite_police, "Survivant : Maj pour marcher (lent, sans traces de pas). V : tout voir (test).", GRIS),
        (petite_police, "Murets (beige) : on voit par-dessus, sauf un survivant qui marche collé derrière.", GRIS),
    ]
    y = 160
    for p, texte, couleur in lignes:
        rendu = p.render(texte, True, couleur)
        ecran.blit(rendu, rendu.get_rect(center=(jeu.LARGEUR / 2, y)))
        y += 40 if p is police else 30
    pygame.display.flip()


# =============================================================================
# Boucle de jeu
# =============================================================================
def nouvelle_manche(reglages):
    """Nouvel état + nouveau bot pour le rôle que le joueur ne joue pas."""
    if reglages["role"] == "survivant":
        bot = TueurBot(reglages["temperature"])
    else:
        bot = SurvivantBot(reglages["niveau"])
    return jeu.start(), bot


def main(max_images=None, role=None):
    pygame.init()
    ecran = pygame.display.set_mode((jeu.LARGEUR, jeu.HAUTEUR))
    pygame.display.set_caption("PAC")
    horloge = pygame.time.Clock()
    police = pygame.font.Font(None, 36)
    petite_police = pygame.font.Font(None, 22)

    reglages = {"role": role, "temperature": TEMPERATURE_INIT, "niveau": NIVEAU_INIT, "manches": 0,
                "marche": False, "tout_voir": False, "croyance": False}
    etat, bot = nouvelle_manche(reglages) if role else (None, None)
    croyance = None                    # croyance du tueur (recréée à chaque manche)
    en_cours = True
    images = 0

    while en_cours:
        dt = horloge.tick(FPS) / 1000
        espace = attaque = False           # pressés à cette image ?

        # a) Événements : fermeture de fenêtre, touches pressées une fois
        for evenement in pygame.event.get():
            if evenement.type == pygame.QUIT:
                en_cours = False
            elif evenement.type == pygame.MOUSEBUTTONDOWN and evenement.button == 1:
                attaque = True
            elif evenement.type != pygame.KEYDOWN:
                continue
            elif evenement.key == pygame.K_ESCAPE:
                en_cours = False
            elif reglages["role"] is None:                        # --- menu
                if evenement.key in (pygame.K_s, pygame.K_t):
                    reglages["role"] = "survivant" if evenement.key == pygame.K_s else "tueur"
                    etat, bot = nouvelle_manche(reglages)
            else:                                                 # --- en jeu
                if evenement.key == pygame.K_TAB:
                    reglages["role"] = None
                elif evenement.key == pygame.K_r:
                    etat, bot = nouvelle_manche(reglages)
                elif evenement.key == pygame.K_SPACE:
                    espace = True
                elif evenement.key == pygame.K_a:
                    attaque = True
                elif evenement.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                    reglages["marche"] = not reglages["marche"]
                elif evenement.key == pygame.K_v:
                    reglages["tout_voir"] = not reglages["tout_voir"]
                elif evenement.key == pygame.K_c:
                    reglages["croyance"] = not reglages["croyance"]
                elif evenement.unicode in ("+", "-"):
                    plus = evenement.unicode == "+"
                    # Adversaire plus fort = tueur moins bruité / survivant de meilleur niveau
                    if reglages["role"] == "survivant":
                        t = reglages["temperature"] + (-0.05 if plus else 0.05)
                        reglages["temperature"] = bot.temperature = round(min(max(t, 0.0), 2.0), 2)
                    else:
                        n = reglages["niveau"] + (0.1 if plus else -0.1)
                        reglages["niveau"] = round(min(max(n, 0.0), 1.0), 1)
                        bot = SurvivantBot(reglages["niveau"])

        if reglages["role"] is None:
            dessiner_menu(ecran, police, petite_police)
        else:
            # b) Commandes : le clavier pour toi, le bot pour l'autre
            clavier = direction_clavier(pygame.key.get_pressed())
            if reglages["role"] == "survivant":
                commandes = ((clavier, espace, reglages["marche"]), bot.decider(etat, dt))
            else:
                commandes = (bot.decider(etat, dt), (clavier, espace, attaque))

            # c) Mise à jour puis dessin
            if jeu.mettre_a_jour(etat, *commandes, dt):
                reglages["manches"] += 1
            # Croyance affichée (touche C) : celle du tueur bot si c'est lui qui joue le
            # tueur (c'est elle qui guide ses décisions) ; sinon une croyance calculée
            # pour toi, avec ce que tu vois
            if getattr(bot, "croyance", None) is not None:
                croyance = bot.croyance
            else:
                if croyance is None or croyance.etat is not etat:     # nouvelle manche
                    croyance = Croyance(etat)
                if not etat["fini"]:
                    croyance.mettre_a_jour(etat, dt)
            dessiner(ecran, police, petite_police, etat, reglages, croyance)

        images += 1
        if max_images is not None and images >= max_images:
            en_cours = False

    pygame.quit()


if __name__ == "__main__":
    main()
