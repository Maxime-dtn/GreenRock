import pandas as pd
import numpy as np

# --- Conventions de nettoyage
LT_FACTOR = 0.5                   # "<1" est lu comme 0.5 (moitié de la borne)
SCOPE3_PROVIDER = "a"             # "a" ou "b" (colonne ghg_scope3_provider_*)


chemin_fichier ="Data/Providence_TD_raw_data.xlsx"
chemin_sortie = "Data/Providence_TD_clean.xlsx"
 
_MISSING_TOKENS = {"", "-", "n/a", "na", "n.a.", "n.d.", "nd", "not disclosed",
                   "none", "nan", "not assessed", "under review"}
_TRUE_TOKENS = {"yes", "y", "true", "1", "oui"}
_FALSE_TOKENS = {"no", "n", "false", "0", "non"}

_EMISSION_TO_TONNES = {"tco2e": 1.0, "ktco2e": 1e3, "mtco2e": 1e6}
_ESTIMATED_SOURCES = {"estimated", "provider estimate", "sector average model"}

#%% 1. IMPORT, DÉDOUBLONNAGE ET NETTOYAGE

def load_data(chemin_fichier):
    """Lit les deux onglets utiles, tout en texte (aucune conversion automatique)."""
    universe = pd.read_excel(chemin_fichier, sheet_name="Univers brut", dtype=str)
    controversies = pd.read_excel(chemin_fichier, sheet_name="Controverses", dtype=str)
    universe.columns = universe.columns.str.strip()
    controversies.columns = controversies.columns.str.strip()
    return universe, controversies


#%% 2. FONCTIONS DE PARSING (texte -> nombre / booléen)
 
def _clean_text(x):
    """Minuscule, sans espaces insécables ni espaces de bord ; '' si vide."""
    if pd.isna(x):
        return ""
    return str(x).replace("\u00a0", " ").replace("\u202f", " ").strip().lower()


def parse_number(x, lt_factor=LT_FACTOR):
    """
    Convertit les nombres stockés en texte : '7 650', '12,4 %', '<1', '~30'.
    'n/a', 'not disclosed', '-', vide => NaN.  '<b' => b * lt_factor.
    """
    raw = _clean_text(x)
    if raw in _MISSING_TOKENS:
        return np.nan
    s = raw.replace(" ", "").replace("%", "").replace("≈", "~").lstrip("~")
    below = s.startswith("<")
    s = s.lstrip("<").replace(",", ".")
    try:
        value = float(s)
    except ValueError:
        return np.nan
    return value * lt_factor if below else value



def parse_bool(x):
    """Yes/Y/TRUE/1/oui => True ; No/N/FALSE/0/non => False ; le reste => <NA>."""
    s = _clean_text(x)
    if s in _TRUE_TOKENS:
        return True
    if s in _FALSE_TOKENS:
        return False
    return pd.NA          # vide, 'Under review', 'Not assessed' : ni oui ni non



#%% Dédoublonnage par ISIN.

def deduplicate(universe):

    df = universe.copy()

    # Nombre de valeurs renseignées par ligne
    df["_completeness"] = df.notna().sum(axis=1)

    # Ordre d'apparition initial
    df["_original_order"] = range(len(df))

    # Tri :
    # 1. ISIN
    # 2. complétude décroissante
    # 3. ordre d'apparition croissant

    df = df.sort_values(
        ["isin", "_completeness", "_original_order"],
        ascending=[True, False, True]
    )

    # Garde une seule ligne par ISIN
    kept = df.drop_duplicates(
        subset="isin",
        keep="first"
    )

    # Récupère les lignes supprimées
    removed = df.loc[
        ~df.index.isin(kept.index),
        ["issuer_name", "isin"]
    ].copy()

    # Nettoyage
    kept = kept.drop(
        columns=["_completeness", "_original_order"]
    )

    return kept.sort_index().reset_index(drop=True), removed.reset_index(drop=True)
    
#%% Normalisation des colonnes et calcul de l'intensité carbone

def clean_universe(df):
    """
    Colonnes typées et normalisées :

      - émissions en tCO2e (facteur selon ghg_unit : t / kt / Mt)
      - chiffre d'affaires et capitalisation en M EUR
        (divisés par fx_rate_to_eur_at_year_end,
        exprimé en devise locale pour 1 EUR :
        USD 1.08, GBP 0.85, JPY 168, ...)
      - carbon_intensity = (scope 1 + scope 2 location-based) / CA,
        en tCO2e par M EUR
    """

    # Colonnes conservées
    # Le dédoublonnage se fait désormais uniquement par ISIN,
    out = df[["isin", "issuer_name", "nace_code", "nace_label"]].copy()

    factor = (
        df["ghg_unit"]
        .map(_clean_text)
        .map(_EMISSION_TO_TONNES)
    )

    if factor.isna().any():
        raise ValueError(
            "Unité d'émissions inconnue : "
            f"{df.loc[factor.isna(), 'ghg_unit'].unique()}"
        )

    # =========================================================
    # TAUX DE CHANGE
    # =========================================================

    fx = df["fx_rate_to_eur_at_year_end"].map(parse_number)

    if fx.isna().any() or (fx <= 0).any():
        raise ValueError("Taux de change manquant ou nul.")

    # =========================================================
    # CHIFFRE D'AFFAIRES ET CAPITALISATION
    # =========================================================

    out["revenue_eur_m"] = (
        df["revenue_reported"].map(parse_number) / fx
    )

    out["market_cap_eur_m"] = (
        df["market_cap"].map(parse_number) / fx
    )

    # =========================================================
    # ÉMISSIONS SCOPE 1 / 2 / 3
    # =========================================================

    scope1 = (
        df["ghg_scope1"].map(parse_number) * factor
    )

    scope2 = (
        df["ghg_scope2_location_based"].map(parse_number) * factor
    )

    scope3 = (
        df[f"ghg_scope3_provider_{SCOPE3_PROVIDER}"]
        .map(parse_number) * factor
    )

    # =========================================================
    # CARBON INTENSITY
    # =========================================================

    out["carbon_intensity"] = (
        scope1 + scope2
    ) / out["revenue_eur_m"]

    out["carbon_intensity_s123"] = (
        scope1 + scope2 + scope3
    ) / out["revenue_eur_m"]

    # =========================================================
    # ANNÉE ET QUALITÉ DES DONNÉES GHG
    # =========================================================

    out["ghg_reporting_year"] = pd.to_numeric(
        df["ghg_reporting_year"],
        errors="coerce"
    )

    out["ghg_estimated"] = (
        df["ghg_data_source"]
        .map(_clean_text)
        .isin(_ESTIMATED_SOURCES)
    )

    # =========================================================
    # ACTIVITÉS CONTROVERSÉES
    # =========================================================

    coal_raw = df["coal_revenue_share"].map(parse_number)

    out["coal_pct"] = coal_raw.clip(upper=100)

    out["tobacco_pct"] = (
        df["tobacco_revenue_share"].map(parse_number)
    )

    out["weapons"] = (
        df["controversial_weapons_involvement"]
        .map(parse_bool)
        .astype("boolean")
    )

    out["ungc_violation"] = (
        df["ungc_oecd_violation"]
        .map(parse_bool)
        .astype("boolean")
    )

    out["new_fossil_expansion"] = (
        df["new_fossil_expansion"]
        .map(parse_bool)
        .astype("boolean")
    )

    # =========================================================
    # VÉRIFICATION DES DONNÉES ESSENTIELLES
    # =========================================================

    bad = out[
        [
            "revenue_eur_m",
            "market_cap_eur_m",
            "carbon_intensity"
        ]
    ].isna().any(axis=1)

    if bad.any():
        raise ValueError(
            "CA, capitalisation ou émissions illisibles pour : "
            f"{out.loc[bad, 'issuer_name'].tolist()}"
        )

    return out

#%% Pipline complète 

def prepare_data(chemin_fichier):
    """
    Importe, dédoublonne et nettoie les données ESG.

    Returns
    -------
    df : pd.DataFrame
        DataFrame final nettoyé.
    removed : pd.DataFrame
        Entreprises supprimées lors du dédoublonnage.
    controversies : pd.DataFrame
        Données de l'onglet Controverses.
    """

    # 1. Import
    universe, controversies = load_data(chemin_fichier)

    # 2. Dédoublonnage
    universe, removed = deduplicate(universe)

    # 3. Nettoyage et normalisation
    df = clean_universe(universe)

    # 4. Initialisation du screening
    df["EXCLUDED"] = False

    return df, removed, controversies


def export_to_excel(df, chemin_sortie, removed=None, controversies=None):
    """
    Exporte les données nettoyées dans un fichier Excel.

    Onglets créés :
    - Données nettoyées
    - Doublons supprimés (si fourni)
    - Controverses (si fourni)
    """

    with pd.ExcelWriter(chemin_sortie, engine="openpyxl") as writer:

        # Données finales
        df.to_excel(
            writer,
            index=False,
            sheet_name="Données nettoyées"
        )

        # Doublons supprimés
        if removed is not None:
            removed.to_excel(
                writer,
                index=False,
                sheet_name="Doublons supprimés"
            )

        # Controverses
        if controversies is not None:
            controversies.to_excel(
                writer,
                index=False,
                sheet_name="Controverses"
            )

    print(f"Fichier Excel créé : {chemin_sortie}")

    df, removed, controversies = prepare_data(chemin_fichier)

df, removed, controversies = prepare_data(chemin_fichier)

# Export Excel
export_to_excel(df, chemin_sortie)

print(df.head())
print(df.shape)

