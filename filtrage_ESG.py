#%%
import pandas as pd
import numpy as np
from normalisation_data import load_data, deduplicate, clean_universe, parse_number, parse_bool, _clean_text, prepare_data, chemin_fichier

universe, controversies = load_data()
df, removed, controversies = prepare_data(universe, controversies)
print(df.head())


#%%
## Modifier la fonction ci dessous pour remplir la colonne EXCLUDED avec des True et False 
## Pour indiquer si l'entreprise est exclue ou non. Pour l'instant, elle est remplie avec des valeurs aléatoires.
def remplir_excluded_aleatoirement(df, seed=42):
    """Remplit la colonne EXCLUDED avec des True et False aléatoires."""
    rng = np.random.default_rng(seed)
    df["EXCLUDED"] = rng.choice([True, False], size=len(df))
    return df


df = remplir_excluded_aleatoirement(df)
print(df[ "EXCLUDED"].head())



## règle binaire sur % CA

## charbon extraction >5%
## électricité charbon >10%
## tabac seuil zéro
## armement controversé seuil zéro

## no new fossil fuel expansion

## GCEL : recense  les entreprises développant de nouvelles projets liés au charbon.
## GOGEL : ecense  les entreprises développant de nouvelles projets liés au pétrole et au gaz.
# %%
