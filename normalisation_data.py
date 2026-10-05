#%% 1. Import des bibliothèques 

import pandas as pd
import numpy as np
import re
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "Data"
chemin_fichier = DATA_DIR / "data_base.xlsx"
 
_MISSING_TOKENS = {"", "-", "n/a", "na", "n.a.", "n.d.", "nd", "not disclosed",
                   "none", "nan", "not assessed", "under review"}
_TRUE_TOKENS = {"yes", "y", "true", "1", "oui", "1.0"}
_FALSE_TOKENS = {"no", "n", "false", "0", "non", "0.0"}


#%% 1. Import des données

def load_data(path=chemin_fichier):
    """Charge uniquement les onglets Universe et Controversies."""
    with pd.ExcelFile(path, engine="openpyxl") as book:
        universe = pd.read_excel(book, sheet_name="Universe")
        controversies = pd.read_excel(book, sheet_name="Controversies")

    return universe, controversies


#%% 2. FONCTIONS DE PARSING (texte -> nombre / booléen)
 

## Formalise les chaines de carractères : supprime les espaces en début et fin, convertit les chaines vides en <NA>.
def _clean_text(value):
    if pd.isna(value):
        return pd.NA
    value = str(value).strip()
    return value if value else pd.NA



## Formalise les flottants : supprime les espaces, remplace la virgule par un point, convertit les chaines vides en <NA>.
def parse_number(value):
    """Convertit une valeur en float.

    Conventions :
    - virgule ou point comme séparateur décimal ;
    - espaces supprimés ;
    - "12,5 %" -> 12.5 ;
    - "(250)" -> -250 ;
    - "<10" -> 5 ;
    - "10–20" ou "[10;20]" -> 15 ;
    - valeur manquante -> np.nan.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Un booléen ne peut pas être converti en nombre.")

    if isinstance(value, str):
        value = value.strip()

        if value.lower() in {
            "", "na", "n/a", "nan", "none", "null", "nd", "n.d."
        }:
            return np.nan

        value = "".join(value.split())
        value = (
            value.replace("−", "-")
            .replace("–", "-")
            .replace("—", "-")
            .replace(",", ".")
            .replace("%", "")
        )

        # Motif d'un nombre, éventuellement négatif ou scientifique.
        number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

        # Borne supérieure : <x devient x / 2.
        upper_bound = re.fullmatch(rf"<({number})", value)

        # Intervalle : a-b, a;b ou [a;b].
        interval_text = value
        if value.startswith("[") and value.endswith("]"):
            interval_text = value[1:-1]

        interval = re.fullmatch(
            rf"({number})(?:;|-)({number})",
            interval_text,
        )

        if upper_bound:
            bound = float(upper_bound.group(1))

            if bound <= 0:
                raise ValueError(
                    "L'approximation <x -> x/2 exige une borne positive."
                )

            result = bound / 2

        elif interval:
            lower = float(interval.group(1))
            upper = float(interval.group(2))

            if not np.isfinite([lower, upper]).all():
                raise ValueError(f"Bornes non finies : {value!r}")

            if lower > upper:
                raise ValueError(f"Intervalle inversé : {value!r}")

            result = lower / 2 + upper / 2

        else:
            # Parenthèses comptables : (250) devient -250.
            if value.startswith("(") and value.endswith(")"):
                inner = value[1:-1]

                if inner.startswith(("-", "+")):
                    raise ValueError(
                        f"Notation comptable ambiguë : {value!r}"
                    )

                value = "-" + inner

            try:
                result = float(value)
            except ValueError as exc:
                raise ValueError(
                    f"Valeur numérique non reconnue : {value!r}"
                ) from exc

    else:
        result = float(value)

    if not np.isfinite(result):
        raise ValueError(f"Nombre non fini : {value!r}")

    return result


## Formalise les booléens : supprime les espaces, convertit les chaines vides en <NA>, "oui"/"non" en True/False.
def parse_bool(value):

    if pd.isna(value):
        return pd.NA
    token = str(value).strip().lower()
    if token in _MISSING_TOKENS:
        return pd.NA
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise ValueError(f"Booléen non reconnu : {value!r}")

## Formalise les dates : supprime les espaces, convertit les chaines vides en <NA>, convertit les dates en datetime.
def normalize_dates(df, column="event_date", date_format="%Y-%m-%d"):

    if column not in df.columns:
        raise ValueError(f"Colonne absente : {column!r}")

    out = df.copy()

    # Transformer les textes vides en valeurs manquantes.
    values = out[column].replace(r"^\s*$", pd.NA, regex=True)

    out[column] = pd.to_datetime(
        values,
        format=date_format,
        errors="raise"
    )

    return out

#%% Vérifie que toutes les colonnes sont présente 
## (vérifie entre autre que les entreprises présentes dans l'onglet Controverses sont bien dans l'onglet Universe).

def require_columns(df, columns, table):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{table} : colonnes absentes : {sorted(missing)}")

## Normalise les identifiants d'entreprise : supprime les espaces aux extrémités, convertit les chaines vides en <NA>.
def normalize_ids(df, column="id"):
    """Nettoie les identifiants sans modifier leur signification."""
    if column not in df.columns:
        raise ValueError(f"Colonne absente : {column!r}")

    out = df.copy()

    # Supprimer les espaces aux extrémités.
    # Conserver les accents, la casse et les caractères internes.
    out[column] = (
        out[column]
        .map(_clean_text)
        .astype("string")
    )

    # Chaque entreprise doit avoir un identifiant.
    missing = out[column].isna()

    if missing.any():
        raise ValueError(
            f"Identifiant manquant dans {column!r} "
            f"aux index : {out.index[missing].tolist()}"
        )

    return out

## Suppression des doublons
def deduplicate(df, id_column="id"):
    """Garantit une seule ligne par entreprise.

    - Supprime les répétitions entièrement identiques.
    - Refuse les lignes différentes partageant le même identifiant.

    Retourne :
        cleaned : tableau sans doublons
        removed : répétitions identiques supprimées
    """
    out = normalize_ids(df, column=id_column)

    # 1. Retirer uniquement les lignes entièrement identiques.
    exact_duplicates = out.duplicated(keep="first")

    removed = out.loc[exact_duplicates].copy()
    cleaned = out.loc[~exact_duplicates].copy()

    # 2. Vérifier l'unicité de l'identifiant d'entreprise.
    conflicts = cleaned[id_column].duplicated(keep=False)

    if conflicts.any():
        conflicting_ids = (
            cleaned.loc[conflicts, id_column]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Plusieurs lignes différentes pour la même entreprise : "
            f"{conflicting_ids}. "
            "Vérifier les données avant de choisir une ligne."
        )

    return cleaned.reset_index(drop=True), removed.reset_index(drop=True)
    
#%% Nettoyage des des données

def clean_universe(df):
    """Construit un nouveau DataFrame avec uniquement les colonnes normalisées.

    id, name et sector sont nettoyés comme textes pour identifier les entreprises.
    Les autres colonnes sources restent dans le DataFrame brut, sans modification.

    benchmark_weight, return_12m : décimaux (0.05 = 5 %).
    carbon_intensity : tCO2e / M EUR de CA, scope 2 location-based.
    Les flags normalisés portent le suffixe _bool, avec dtype boolean nullable.
    EXCLUDED reste inconnu jusqu'au screening ; aucun titre n'est autorisé ici.
    """

    required = ["benchmark_weight_pct", "return_12m_pct", "carbon_intensity_s12_t_per_meur_revenue", "scope1_mt", "scope2_mt", "revenue_eur_bn"]
    require_columns(df, required, "Universe")
    require_columns(df, ["id", "name", "sector"], "Universe")
    out = pd.DataFrame(index=df.index)
    out["id"] = normalize_ids(df, "id")["id"]
    for column in ["name", "sector"]:
        out[column] = df[column].map(_clean_text).astype("string")
    numeric = {}

    for column in required:
        try:
            numeric[column] = df[column].map(parse_number)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Valeur numérique invalide dans {column}") from exc
        
        if numeric[column].isna().any():
            raise ValueError(f"Donnée indispensable manquante dans {column}")
    
    raw_weights = numeric["benchmark_weight_pct"]
    if (raw_weights < 0).any() or not np.isclose(raw_weights.sum(), 100, atol=0.01, rtol=0):
        raise ValueError(f"Poids du benchmark invalides : somme = {raw_weights.sum()}")
    
    if (numeric["revenue_eur_bn"] <= 0).any():
        raise ValueError("Le chiffre d'affaires doit être strictement positif.")
    
    if any((numeric[c] < 0).any() for c in ["scope1_mt", "scope2_mt", "carbon_intensity_s12_t_per_meur_revenue"]):
        raise ValueError("Émissions ou intensité carbone négatives.")
    
    if (numeric["return_12m_pct"] < -100).any():
        raise ValueError("Rendement inférieur à -100 %.")

    out["benchmark_weight"] = raw_weights / raw_weights.sum()
    out["return_12m"] = numeric["return_12m_pct"] / 100
    out["carbon_intensity"] = numeric["carbon_intensity_s12_t_per_meur_revenue"]
    out["scope1_tco2e"] = numeric["scope1_mt"] * 1_000_000
    out["scope2_tco2e"] = numeric["scope2_mt"] * 1_000_000
    out["revenue_meur"] = numeric["revenue_eur_bn"] * 1_000

    out["carbon_intensity_recalculated"] = (
        out["scope1_tco2e"] + out["scope2_tco2e"]
    ) / out["revenue_meur"]
    # Écart conservé pour audit ; on utilise la valeur source pour reproduire le benchmark.
    out["carbon_intensity_gap"] = out["carbon_intensity_recalculated"] - out["carbon_intensity"]
    
    for column in df.columns:
        if column.endswith("_flag"):
            out[column + "_bool"] = pd.array(df[column].map(parse_bool), dtype="boolean")
    out["EXCLUDED"] = pd.array([pd.NA] * len(out), dtype="boolean")
    out["exclusion_reasons"] = pd.Series("", index=out.index, dtype="string")
    return out




def prepare_data(universe, controversies):
    """Retourne (df normalisé, doublons retirés, controverses préparées).

    Seul le premier DataFrame est destiné à l'export principal.
    Aucun fichier n'est créé par cette fonction.
    """

    # 2. Vérification des colonnes indispensables
    require_columns(
        universe,
        ["id", "name", "sector"],
        "Universe"
    )
    require_columns(
        controversies,
        ["issuer_id", "event_date"],
        "Controversies"
    )

    # 3. Nettoyage et dédoublonnage des entreprises
    # deduplicate appelle déjà normalize_ids.
    universe, removed = deduplicate(universe)

    # 4. Nettoyage des identifiants et dates des controverses
    controversies = normalize_ids(controversies, "issuer_id")
    controversies = normalize_dates(controversies)

    # 5. Vérification des correspondances entre les deux tables
    unknown = set(controversies["issuer_id"]) - set(universe["id"])

    if unknown:
        raise ValueError(
            f"Controversies : identifiants inconnus : {sorted(unknown)}"
        )

    # 6. Normalisation des données de l'univers
    universe = clean_universe(universe)

    return universe, removed, controversies



#%% Export du DataFrame normalisé

def export_data(df, filename="data_normalized.xlsx", data_dir=DATA_DIR):
    """Enregistre le DataFrame dans Data et renvoie le chemin du fichier.

    Un export existant de même nom est remplacé.
    Les pd.NA sont exportés comme cellules vides (EXCLUDED non évalué).
    """
    filename = Path(filename)
    if filename.name != str(filename) or filename.suffix.lower() != ".xlsx":
        raise ValueError("Fournir un nom de fichier .xlsx sans dossier.")

    data_dir = Path(data_dir)
    path = data_dir / filename
    if path.resolve() == (data_dir / "data_base.xlsx").resolve():
        raise ValueError("L'export ne doit pas écraser data_base.xlsx.")

    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_excel(
        path,
        sheet_name="Universe_normalized",
        index=False,
        engine="openpyxl",
    )
    return path


#%% Exécution


universe, controversies = load_data()
df, removed, controversies = prepare_data(universe, controversies)
output_path = export_data(df)

print(f"DataFrame normalisé : {len(df)} entreprises, {len(df.columns)} colonnes")
print(f"Doublons retirés : {len(removed)}")
print(f"Fichier enregistré : {output_path}")

# %%
