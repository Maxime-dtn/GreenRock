import pandas as pd
import numpy as np

## règle binaire sur % CA

## charbon extraction >5%
## électricité charbon >10%
## tabac seuil zéro
## armement controversé seuil zéro

## no new fossil fuel expansion

## GCEL : recense  les entreprises développant de nouvelles projets liés au charbon.
## GOGEL : ecense  les entreprises développant de nouvelles projets liés au pétrole et au gaz.

def load_data():
    chemin_fichier = "Data/Providence_TD_raw_data.xlsx"
    # Remplacer la lecture du CSV par la lecture du fichier Excel brut
    df = pd.read_excel(chemin_fichier, sheet_name="Univers brut")
    return df


df = load_data()
print(df.head())





def load_gcel():
    return pd.read_excel("Data/GCEL.xlsx")

gcel = load_gcel()


def normalize_company_name(name):
    ...


def flag_controversial_weapons(df):
    mask_weapons = df['controversial_weapons_involvement'] == True
    mask_ungc_oecd = df['ungc_oecd_violation'] == True     
    df.loc[mask_weapons, 'EXCLUDED'] = True
    return df

def flag_controversial_weapons(df):
    mask = df['controversial_weapons_involvement'] == True    
    df.loc[mask, 'EXCLUDED'] = True
    return df