"""
Simulations sans affichage : le tueur bot contre le survivant bot (bots.py).

Pour chaque couple (température du tueur, niveau du survivant), on joue des
manches et on compte les victoires du tueur. Ce sont ces données qui serviront à
calibrer le modèle bayésien : P(victoire) en fonction du niveau et de la difficulté.

Lancer :  python entrainement.py                   (20 manches par couple, ~30 min)
          python entrainement.py --manches 200 --graine 1

Produit :
  simulations.csv          une ligne par manche
  courbe_difficulte.png    taux de victoire du tueur selon le niveau, une courbe par température
"""
import argparse
import random
import time

import matplotlib
matplotlib.use("Agg")                    # pas de fenêtre : on enregistre juste l'image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jeu
from bots import SurvivantBot, TueurBot

DT = 1 / 30                              # pas de temps fixe : 30 mises à jour par seconde de jeu
TEMPERATURES = [0.0, 0.1, 0.2, 0.35, 0.5, 1.0]   # 0 = tueur parfait
NIVEAUX = [0.0, 0.25, 0.5, 0.75, 1.0]


# =============================================================================
# Une manche complète, sans affichage : bot tueur contre bot survivant
# =============================================================================
def jouer_manche(temperature, niveau):
    etat = jeu.start()
    tueur, survivant = TueurBot(temperature), SurvivantBot(niveau)
    while not jeu.mettre_a_jour(etat, survivant.decider(etat, DT), tueur.decider(etat, DT), DT):
        pass
    return {
        "temperature": temperature,
        "niveau": niveau,
        "victoire_tueur": int(etat["gagnant"] == "tueur"),
        "coups": etat["coups"],
        "duree": round(etat["temps"], 2),
        "palettes_baissees": sum(e != "debout" for e in etat["palettes"]),
        "palettes_cassees": sum(e == "cassee" for e in etat["palettes"]),
        "sauts_survivant": sum(etat["passages"]),
        "fenetres_fermees": sum(jeu.fenetre_fermee(etat, k) for k in range(len(etat["passages"]))),
        "carte": etat["carte"]["nom"],
    }


# =============================================================================
# Programme principal
# =============================================================================
def simuler(n_par_couple):
    resultats = []
    debut = time.time()
    print(f"{n_par_couple} manches par couple (température, niveau)\n")
    print("  T \\ niveau " +"".join(f"{n:>8.2f}" for n in NIVEAUX))
    for T in TEMPERATURES:
        ligne = []
        for niveau in NIVEAUX:
            res = [jouer_manche(T, niveau) for _ in range(n_par_couple)]
            resultats += res
            ligne.append(sum(r["victoire_tueur"] for r in res) / n_par_couple)
        print(f"  {T:9.2f}  " + "".join(f"{x:8.0%}" for x in ligne)
              + f"   [{time.time() - debut:4.0f} s]")

    resultats = pd.DataFrame(resultats)
    resultats.index.name = "manche"
    resultats.to_csv("simulations.csv")
    print("\nRésultats enregistrés dans simulations.csv")
    return resultats


def tracer(resultats):
    taux = resultats.groupby(["temperature", "niveau"])["victoire_tueur"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    couleurs = plt.cm.viridis(np.linspace(0, 0.9, len(taux.columns)))
    for T, c in zip(taux.columns, couleurs):
        ax.plot(taux.index, taux[T], "o-", color=c, label=f"T = {T:g}")
    ax.set_xlabel("niveau du survivant bot")
    ax.set_ylabel("taux de victoire du tueur")
    ax.set_ylim(0, 1)
    ax.set_title("Tueur bot : difficulté selon la température")
    ax.legend(title="température du tueur")
    fig.tight_layout()
    fig.savefig("courbe_difficulte.png", dpi=120)
    print("Graphique enregistré dans courbe_difficulte.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simule le tueur bot contre le survivant bot.")
    p.add_argument("--manches", type=int, default=20, help="manches par couple (température, niveau)")
    p.add_argument("--graine", type=int, default=None, help="graine aléatoire (reproductibilité)")
    args = p.parse_args()

    if args.graine is not None:
        random.seed(args.graine)
        np.random.seed(args.graine)

    tracer(simuler(args.manches))
