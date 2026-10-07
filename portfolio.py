#%% 
"""GreenInvestment — construction climatique, sans classes.

Installation : python -m pip install numpy pandas scipy matplotlib openpyxl

Sources : Classement_362 et onglet Date 1 GreenInvestment.
Les seuils sont modifiables dans PARAMETRES. Aucune certification Article 9
ou conformité réglementaire complète n'est produite par ce programme.
Les données du classeur sont illustratives. Les rendements servent seulement
à une comparaison rétrospective, jamais à la sélection des titres.
"""
from pathlib import Path
import argparse
import json
import warnings
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import milp, Bounds, LinearConstraint



DATA_DIR = Path(__file__).resolve().parent / "Data"
universe_path = DATA_DIR / "data_base_ESG_Classement_complet.xlsx"
date1_path = DATA_DIR / "date1_portfolios.xlsx"

output_dir = Path(__file__).resolve().parent / "Resultats_GreenInvestment"
if __name__ == '__main__':
    output_dir.mkdir(parents=True, exist_ok=True)


PARAMETRES = {
    'fund': 'GreenInvestment',
    'carbon_start_offset': None,  # None : toutes les années simulées ; 0 : Date 2 et futur seulement. 
    'n_positions': 30,
    'min_weight': .02, 
    'max_weight': .1,
    'max_sector': .25,                          # Pas plus de 25 % du portefeuille dans un même secteur
    'max_country': .40,                         # Pas plus de 40 % du portefeuille dans un même pays
    'max_region': .70,                          # Pas plus de 70 % du portefeuille dans une même région
    'min_sectors': 5, 
    'min_regions': 2,
    
    'core_min': .70, 
    'core_max': .80,
    'core_taxonomy_min': .20, 
    'core_capex_min': .30,

    'taxonomy_min': .10, 
    'taxonomy_coverage_min': .80,               # Au moins 80 % du poids du portefeuille doit disposer d’une donnée d’alignement taxonomie.
    'score_coverage_min': .60,                  # Chaque entreprise candidate doit disposer d’au moins 60 % des informations du score climatique
    'minimum_climate_score': .25,               # Chaque entreprise candidate doit obtenir un score climatique d’au moins 0,25 sur 1.
    'sector_carbon_quantile': .90,              # Une entreprise est écartée si son intensité carbone dépasse le 90ᵉ percentile de son secteur
    'high_emitter_quantile': .75,               # Une entreprise dépassant le 75ᵉ percentile d’intensité carbone de l’univers complet
    'initial_carbon_ratio': .50,                # La WACI du portefeuille doit être au maximum égale à 50 % de celle du benchmark,
    'annual_reduction': .07,                    # Trajectoire de réduction annuelle de 7 % par rapport à la base.
    'years_since_base': 0, 
    'base_waci': None, 
    'previous_waci': None,
    'exclude_oil_gas_services': True,

    # Proposition prudente : toute controverse en cours >=4. 
    # La seconde règle dépasse le seul périmètre environnemental.


    # objective = 'climate', ou 'esg'

    'severe_controversy_threshold': 4,
    'accepted_sbti': ['validated 1.5', 'validated wb2'],        # Liste les statuts SBTi acceptés comme preuve de trajectoire validée : « validated 1.5 » pour une cible alignée sur 1,5 °C et « validated WB2 »
    'objective': 'esg',                                         # Maximise le score climatique moyen pondéré du portefeuille.
    'aum_eur_m': 500.,                                          # Montant détenu par le fonds, en millions d’euros, pour calculer les contributions et les transactions.
    'time_limit': 120.,                                         # Accorde au solveur 120 secondes maximum pour l’optimisation.
    'mip_rel_gap': 1e-6,                                        # Fixe la tolérance relative d’optimalité
}


CI = 'carbon_intensity_s12_t_per_meur_revenue'                      # Désigne la colonne d’intensité carbone
FLAGS = ['pab_exclusion_flag', 'ungc_oecd_violation_flag',          # Regroupe les colonnes signalant un motif d’exclusion
         'controversial_weapons_flag', 'new_fossil_exploration_flag',
         'coal_expansion_flag', 'gogel_expansion_listed_flag']


PROJECTION = {
    'reference_year': None,  # None : années relatives à Date 2 ; sinon année confirmée du fichier.
    'past_years': 5,
    'future_years': 5,
    'past_revenue_growth': 0.02,  # Hypothèse, pas une croissance historique observée.
    'scenarios': {
        'Tendance prolongée': {'emissions_shift': 0.0, 'revenue_growth': 0.03},
        'Décarbonation ralentie': {'emissions_shift': 0.02, 'revenue_growth': 0.03},
        'Décarbonation accélérée': {'emissions_shift': -0.02, 'revenue_growth': 0.03},
    },
}


#%%

# Cette fonction vérifie que les paramètres choisis sont cohérents avant de lancer l’optimisation.
def validate_parameters(p):

    if not isinstance(p['n_positions'], int) or not 20 <= p['n_positions'] <= 40:
        raise ValueError('Le nombre de positions doit être entier, entre 20 et 40.')

    
    for key in ['min_weight', 'max_weight', 'max_sector', 'max_country',
                'max_region', 'core_min', 'core_max', 'core_taxonomy_min',
                'core_capex_min', 'taxonomy_min', 'taxonomy_coverage_min',
                'score_coverage_min', 'minimum_climate_score',
                'sector_carbon_quantile', 'high_emitter_quantile',
                'initial_carbon_ratio']:
        if not np.isfinite(p[key]) or not 0 <= p[key] <= 1:
            raise ValueError(f'{key} doit être entre 0 et 1.')

        
    n = p['n_positions']
    if not 0 < p['min_weight'] <= p['max_weight'] or not n*p['min_weight'] <= 1 <= n*p['max_weight']:
        raise ValueError('Bornes de poids incompatibles avec le nombre de positions.')

    
    if p['core_min'] > p['core_max']:
        raise ValueError('Bornes cœur climatique invalides.')

    
    if not 0 <= p['annual_reduction'] < 1 or not isinstance(p['years_since_base'], int) or p['years_since_base'] < 0:
        raise ValueError('Trajectoire invalide.')

    
    for key in ['base_waci', 'previous_waci']:
        if p[key] is not None and (not np.isfinite(p[key]) or p[key] < 0):
            raise ValueError(f'{key} doit être positif ou nul.')


    if p['years_since_base'] and p['base_waci'] is None:
        raise ValueError('Renseigner base_waci pour une année postérieure à Date 2.')

    
    for key in ['min_sectors', 'min_regions']:
        if not isinstance(p[key], int) or not 1 <= p[key] <= n:
            raise ValueError(f'{key} invalide.')

        
    if not np.isfinite([p['aum_eur_m'], p['time_limit'], p['mip_rel_gap']]).all() or p['time_limit'] <= 0 or p['mip_rel_gap'] < 0:
        raise ValueError('Encours et paramètres solveur invalides.')
    if p['aum_eur_m'] <= 0:
        raise ValueError('Encours invalide.')


if __name__ == '__main__':
    p = PARAMETRES.copy()
if __name__ == '__main__':
    validate_parameters(p)


#%%

# On applique la fonction parse_bool pour compléter les colonnes de flags avec des valeurs booléennes True/False ou NaN. 
def parse_bool(series):

    mapping = {'true': True, '1': True, '1.0': True, 'yes': True, 'oui': True,
               'false': False, '0': False, '0.0': False, 'no': False, 'non': False}
    return series.astype('string').str.strip().str.lower().map(mapping).astype('boolean')


# Vérifie les identifiants de l’univers et de la date 1 : pas de valeur manquante, pas de chaîne vide, pas de doublon.
def check_ids(df, label):

    if df['id'].isna().any():
        raise ValueError(f'{label} : identifiant manquant.')
    
    df['id'] = df['id'].astype(str).str.strip()
    
    if df['id'].eq('').any() or df['id'].duplicated().any():
        raise ValueError(f'{label} : identifiant vide ou doublonné.')


# Cette fonction sert à calculer des poids normalisés à partir de pourcentages. Elle vérifie que les valeurs sont numériques, non négatives et que leur somme est proche de 100 %. 
# Si la somme n’est pas exactement 100 %, elle normalise les poids pour qu’ils totalisent 1.0.
def normalize_weights(values, label):

    v = pd.to_numeric(values, errors='raise')
    if v.isna().any() or not np.isfinite(v).all() or (v < 0).any():
        raise ValueError(f'{label} : poids invalides.')

    total = v.sum()

    if abs(total - 100) > .05:
        raise ValueError(f'{label} : somme {total}, attendue proche de 100 %.')

    if not np.isclose(total, 100, atol=1e-9, rtol=0):
        warnings.warn(f'{label} : somme {total:.6f} %, normalisation des arrondis.')

    return v / total

#%%

def load_data(universe_path, date1_path, p):

    date1_path = Path(date1_path)
    if not date1_path.exists() and date1_path.name == 'date1_portfolios.xlsx':
        alternative = date1_path.with_name('date1_portfolios(1).xlsx')
        if alternative.exists():
            date1_path = alternative
    df = pd.read_excel(universe_path, sheet_name='Classement_362')

    # Vérifie que toutes les colonnes attendues sont présentes dans l’univers. Si ce n’est pas le cas, on lève une exception avec la liste des colonnes manquantes.
    required = ['id', 'name', 'sector', 'country', 'region', 'sub_industry', CI,
                'benchmark_weight_pct', 'return_12m_pct', 'esg_score_95',
                'eligible_same_gate', 'tobacco_rev_pct', 'coal_rev_pct',
                'oil_rev_pct', 'gas_rev_pct', 'water_controversy_0_5',
                'ongoing_controversies', 'max_ongoing_severity',
                'taxonomy_aligned_rev_pct', 'green_capex_pct',
                'emissions_change_5y_pct', 'transition_plan_score_0_2',
                'sbti_status', 'scope1_mt', 'scope2_mt', 'revenue_eur_bn'] + FLAGS
    
    missing = sorted(set(required) - set(df))

    if missing:
        raise ValueError(f'Colonnes absentes : {missing}')

    # numérise les colonnes attendues et lève une exception si une valeur non numérique est trouvée.
    check_ids(df, 'Univers')
    numeric = [CI, 'benchmark_weight_pct', 'return_12m_pct', 'esg_score_95',
               'tobacco_rev_pct', 'coal_rev_pct', 'oil_rev_pct', 'gas_rev_pct',
               'water_controversy_0_5', 'ongoing_controversies',
               'max_ongoing_severity', 'taxonomy_aligned_rev_pct',
               'green_capex_pct', 'emissions_change_5y_pct', 'transition_plan_score_0_2']
    
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors='raise')

    for col in ['scope1_mt', 'scope2_mt', 'revenue_eur_bn']:
        df[col] = pd.to_numeric(df[col], errors='raise')
    if not np.isfinite(df[['scope1_mt', 'scope2_mt', 'revenue_eur_bn']]).all().all() or (df[['scope1_mt', 'scope2_mt']] < 0).any().any() or df.revenue_eur_bn.le(0).any():
        raise ValueError('Scopes ou CA invalides.')
    df['ci_provided_source'] = df[CI]
    df[CI] = (df.scope1_mt + df.scope2_mt) * 1000 / df.revenue_eur_bn
    for col in ['taxonomy_aligned_rev_pct', 'green_capex_pct']:
        if not df[col].dropna().between(0, 100).all():
            raise ValueError(f'{col} doit être entre 0 et 100.')
    if not df.esg_score_95.dropna().between(0, 95).all():
        raise ValueError('esg_score_95 doit être entre 0 et 95.')

    # Normalise les poids du benchmark pour qu’ils totalisent 1.0 et les ajoute à l’univers.
    df['benchmark_weight'] = normalize_weights(df['benchmark_weight_pct'], 'Benchmark')

    # Récupère les données du fichier date1_portfolio, onglet p['fund'] (GreenInvestement dans notre cas)
    raw = pd.read_excel(date1_path, sheet_name=p['fund'], header=None)
    headers = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq('id')]
    
    if len(headers) != 1:
        raise ValueError('En-tête Date 1 introuvable ou ambigu.')
    initial = pd.read_excel(date1_path, sheet_name=p['fund'], header=int(headers[0]))
    initial = initial[initial['id'].notna() & ~initial['id'].astype(str).str.strip().str.lower().eq('total')].copy()
    
    check_ids(initial, 'Date 1')


    # Ici on vérifie que tous les titres de Date 1 sont bien présents dans l’univers. Si ce n’est pas le cas, on lève une exception avec la liste des titres manquants.
    unknown = set(initial.id) - set(df.id)
    
    if unknown:
        raise ValueError(f'Titres Date 1 absents de l’univers : {sorted(unknown)}')

    # Calcul les poids normalisés de Date 1 et les fusionne avec l’univers. Les titres non présents dans Date 1 reçoivent un poids nul.
    initial['weight_date1'] = normalize_weights(initial['weight_pct'], 'Date 1')
    df = df.merge(initial[['id', 'weight_date1']], on='id', how='left', validate='one_to_one')
    df['weight_date1'] = df['weight_date1'].fillna(0.)

    return df


if __name__ == '__main__':
    universe = load_data(universe_path, date1_path, p)
if __name__ == '__main__':
    print(f'Univers chargé : {len(universe)} entreprises, {universe.benchmark_weight.sum():.2%} du benchmark.')
if __name__ == '__main__':
    print(f'Date 1 : {len(universe.loc[universe.weight_date1>0])} positions, {universe.weight_date1.sum():.2%} du portefeuille.')
if __name__ == '__main__':
    print(universe.head(10).to_string(index=False, formatters={'benchmark_weight': '{:.2%}'.format, 'weight_date1': '{:.2%}'.format}))

#%%

# Ajoute à chaque entreprise les informations nécessaires à sa sélection : 
# scores, niveau d’intensité carbone, trajectoire vérifiée et appartenance au cœur climatique
def prepare_candidates(universe, p):

    df = universe.copy()
    df['taxonomy'] = df['taxonomy_aligned_rev_pct'] / 100
    df['green_capex'] = df['green_capex_pct'] / 100

    # Ici on regroupe les entreprises par secteur
    # On calcule le percentile d’intensité carbone pour chaque secteur et on l’ajoute à chaque entreprise. 
    # On inscrit ce score dans une nouvelle colonne 'sector_carbon_limit'.
    df['sector_carbon_limit'] = df.groupby('sector')[CI].transform(lambda x: x.quantile(p['sector_carbon_quantile']))

    # Maintenant, on repère les entreprises fortement émettrices à l’échelle de l’univers
    # Dans une nouvelle colonne 'high_emitter', on indique True pour les entreprises dont l’intensité carbone dépasse le 75ᵉ percentile de l’univers, et False sinon.
    df['high_emitter'] = df[CI] > df[CI].quantile(p['high_emitter_quantile'])
    

    # Classement croissant : le meilleur rendement a le rang le plus élevé.
    rank = df['return_12m_pct'].rank(method='average')
    n = df['return_12m_pct'].notna().sum()

    # Sous-score de rendement compris entre 0 et 5.
    df['return_score_5'] = 5 * (rank - 1) / (n - 1) if n > 1 else rank.where(rank.isna(), 2.5)

    # Score total sur 100.
    # Si l'une des deux données manque, le score final reste manquant.
    df['final_score_100'] = df['esg_score_95'] + df['return_score_5']

    # Score que l'optimisation doit maximiser.
    df['objective_score'] = df['final_score_100'] / 100 

    # Ici, on détermine si chaque entreprise possède une trajectoire climatique considérée comme vérifiée (en comparant le statut SBTi à une liste de statuts acceptés).
    df['verified_trajectory'] = df['sbti_status'].astype('string').str.strip().str.lower().isin(p['accepted_sbti'])
    # Si un vérification particulière est fournie dans la colonne 'trajectory_verified_flag', on l'utilise pour compléter l'information sur la trajectoire vérifiée.
    if 'trajectory_verified_flag' in df:
        df['verified_trajectory'] |= parse_bool(df['trajectory_verified_flag']).fillna(False)

    # Puis, on détermine l’appartenance au cœur climatique
    # Une entreprise appartient au cœur climatique si elle satisfait au moins l’une de ces deux voies :
    # - Au moins 20 % du chiffre d’affaires aligné taxonomie.
    # - Au moins 30 % de CapEx verts, ET un plan de transition noté 2/2, ET une trajectoire vérifiée.
    df['core'] = (df['taxonomy'].ge(p['core_taxonomy_min']) |
                 (df['green_capex'].ge(p['core_capex_min']) &
                  df['transition_plan_score_0_2'].eq(2) & df['verified_trajectory']))
    reasons = [[] for _ in range(len(df))]

    # La fonction reject ajoute un motif de rejet aux entreprises qui remplissent une condition peu souhaitable donnée.
    # Cette fonction prend en entrée une liste de booléens indiquant quelles entreprises sont concernées et une raison de rejet.
    def reject(mask, reason):

        for i in np.flatnonzero(np.asarray(mask.fillna(True), dtype=bool)):
            reasons[i].append(reason)

    # On applique ensuite cette fonction pour rejeter les entreprises qui ne sont pas éligibles selon les critères de screening des collègues. 
    gate = parse_bool(df['eligible_same_gate'])
    reject(~gate, 'Screening collègues refusé ou incomplet')


    # Maintenant, on rejette les entreprises si leur intensité carbone dépasse le 90ᵉ percentile de son secteur
    reject(df[CI] > df['sector_carbon_limit'], 'Intensité au-dessus du percentile sectoriel')
    # On rejette également les entreprises fortement émettrices, tout secteurs confondus
    reject(df['high_emitter'] & ~df['verified_trajectory'], 'Fort émetteur sans trajectoire vérifiée')
    # On rejette les entreprises qui n’atteignent pas le score climatique minimum requis
    reject(df['objective_score'] < p['minimum_climate_score'], 'Score climatique insuffisant')
    # On rejette les entreprises dont le secteur, le pays ou la région est : manquant (NaN) ou vide. (nécessaires pour contrôler la diversification du portefeuille)
    reject(df[['sector', 'country', 'region']].isna().any(axis=1) |
           df[['sector', 'country', 'region']].astype(str).apply(lambda s: s.str.strip().eq('')).any(axis=1), 'Classification incomplète')
    reject(~np.isfinite(df['objective_score']), 'Score objectif absent ou invalide')
    if 'EXCLUDED' in df:
        reject(parse_bool(df['EXCLUDED']), 'EXCLUDED vrai ou inconnu')
    df['admission_reasons'] = ['; '.join(r) for r in reasons]
    df['admissible'] = df['admission_reasons'].eq('')
    return df



if __name__ == '__main__':
    candidates = prepare_candidates(universe, p)
if __name__ == '__main__':
    print(f"Entreprises admissibles : {candidates['admissible'].sum()}")
if __name__ == '__main__':
    print(
        "Entreprises admissibles du cœur climatique :",
        (candidates["admissible"] & candidates["core"]).sum()
    )

#%%

# On reconstitue ici les valeurs de S1, S2 et CA
def extrapolate_company_data(d):

    q = PROJECTION
    if d.empty:
        raise ValueError('Univers vide.')
    if not np.isfinite(q['past_revenue_growth']) or q['past_revenue_growth'] <= -1:
        raise ValueError('past_revenue_growth doit être fini et supérieur à -1.')

    # Vérification des horizons
    for key in ('past_years', 'future_years'):
        if (
            isinstance(q[key], bool)
            or not isinstance(q[key], int)
            or q[key] < 0
        ):
            raise ValueError(
                f'{key} doit être un entier positif ou nul.'
            )

    # Vérification de l'année de référence
    year = q['reference_year']

    if year is not None:
        if isinstance(year, bool) or not isinstance(year, int):
            raise ValueError(
                'reference_year doit être un entier ou None.'
            )

    # Vérification des scénarios
    if not q['scenarios']:
        raise ValueError(
            'Le dictionnaire scenarios ne peut pas être vide.'
        )

    required = [
        'id',
        'name',
        'scope1_mt',
        'scope2_mt',
        'revenue_eur_bn',
        'emissions_change_5y_pct'
    ]

    missing = sorted(set(required) - set(d.columns))

    if missing:
        raise ValueError(
            f'Colonnes absentes : {missing}'
        )

    # Copie de travail
    base = d.copy().reset_index(drop=True)

    # Vérification des identifiants
    check_ids(base, 'Projection')

    # Conversion des colonnes numériques
    numeric_cols = [
        'scope1_mt',
        'scope2_mt',
        'revenue_eur_bn',
        'emissions_change_5y_pct'
    ]

    for col in numeric_cols:
        base[col] = pd.to_numeric(
            base[col],
            errors='raise'
        )

    # Vérification des valeurs
    valid = np.isfinite(
        base[numeric_cols]
    ).all(axis=1)

    valid &= base['scope1_mt'].ge(0)
    valid &= base['scope2_mt'].ge(0)
    valid &= base['revenue_eur_bn'].gt(0)
    valid &= base['emissions_change_5y_pct'].gt(-100)

    if not valid.all():
        invalid_ids = base.loc[~valid, 'id'].astype(str)

        raise ValueError(
            'Projection impossible pour : '
            + ', '.join(invalid_ids)
        )

    # Taux annuel composé des émissions
    g = np.expm1(
        np.log1p(
            base['emissions_change_5y_pct'].to_numpy() / 100
        ) / 5
    )

    # Conversion des unités
    # MtCO2e -> tCO2e
    s1 = base['scope1_mt'].to_numpy() * 1e6
    s2 = base['scope2_mt'].to_numpy() * 1e6

    # Md€ -> M€
    ca = base['revenue_eur_bn'].to_numpy() * 1e3

    # Intensité carbone fournie dans le fichier initial

    ci_provided = pd.to_numeric(
        base['ci_provided_source'] if 'ci_provided_source' in base else base.get(CI, pd.Series(np.nan, index=base.index)),
        errors='raise'
    ).to_numpy()

    # Liste qui contiendra toutes les années et tous les scénarios
    parts = []

    # Parcours des scénarios
    for scenario, rates in q['scenarios'].items():

        try:
            shift = float(rates['emissions_shift'])
            ca_growth = float(rates['revenue_growth'])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f'Scénario invalide : {scenario}'
            ) from error

        # Vérification des taux
        if not np.isfinite(shift):
            raise ValueError(
                f'emissions_shift invalide dans {scenario}.'
            )

        if not np.isfinite(ca_growth) or ca_growth <= -1 or np.any(g + shift <= -1):
            raise ValueError(
                f'revenue_growth invalide dans {scenario}.'
            )

        # Parcours des années passées et futures
        for offset in range(
            -q['past_years'],
            q['future_years'] + 1
        ):

            # Le scénario modifie uniquement le futur
            if offset <= 0:
                emissions_growth = g
                revenue_growth = q['past_revenue_growth']
            else:
                emissions_growth = g + shift
                revenue_growth = ca_growth

            with np.errstate(
                over='raise',
                invalid='raise',
                divide='raise',
                under='raise'
            ):

                emission_factor = np.power(
                    1 + emissions_growth,
                    offset
                )

                revenue = ca * np.power(
                    1 + revenue_growth,
                    offset
                )

                scope1 = s1 * emission_factor
                scope2 = s2 * emission_factor

                intensity = (
                    (scope1 + scope2) / revenue
                )

            # DataFrame de l'année et du scénario courant
            frame = base[
                ['id', 'name']
            ].copy()

            frame['scenario'] = scenario
            frame['year_offset'] = offset

            if year is None:
                frame['year'] = pd.NA
            else:
                frame['year'] = year + offset

            if offset < 0:
                frame['data_type'] = (
                    'Reconstitution estimée'
                )
            elif offset == 0:
                frame['data_type'] = (
                    'Donnée du fichier illustratif'
                )
            else:
                frame['data_type'] = (
                    'Projection estimée'
                )

            # Valeurs calculées
            frame['scope1_tco2e'] = scope1
            frame['scope2_tco2e'] = scope2
            frame['revenue_eur_m'] = revenue
            frame[CI] = intensity

            # Hypothèses propres à chaque entreprise et scénario
            frame['emissions_growth_assumed'] = (
                emissions_growth
            )

            frame['revenue_growth_assumed'] = (
                revenue_growth
            )

            frame['emissions_change_5y_pct'] = (
                base['emissions_change_5y_pct'].to_numpy()
            )

            frame['emissions_cagr_assumed'] = g

            # Valeurs initiales conservées pour l'audit
            frame['scope1_base_tco2e'] = s1
            frame['scope2_base_tco2e'] = s2
            frame['revenue_base_eur_m'] = ca
            frame['ci_provided_base'] = ci_provided

            frame['scope_assumption'] = (
                'Variation sur 5 ans appliquée identiquement '
                'aux scopes 1 et 2'
            )

            parts.append(frame)

    # Assemblage final
    result = pd.concat(
        parts,
        ignore_index=True
    )

    # Organisation des colonnes
    columns = [
        'id',
        'name',
        'scenario',
        'year',
        'year_offset',
        'data_type',
        'scope1_tco2e',
        'scope2_tco2e',
        'revenue_eur_m',
        CI,
        'emissions_growth_assumed',
        'revenue_growth_assumed',
        'emissions_change_5y_pct',
        'emissions_cagr_assumed',
        'scope1_base_tco2e',
        'scope2_base_tco2e',
        'revenue_base_eur_m',
        'ci_provided_base',
        'scope_assumption'
    ]

    return result[columns]


if __name__ == '__main__':
    panel = extrapolate_company_data(universe)
if __name__ == '__main__':
    print(panel.head())




# Cette fonction calcule la WACI maximale autorisée pour le portefeuille.
# On compare trois limites : la moitié de la WACI du benchmark, la trajectoire de réduction annuelle par rapport à une base fixe, et la WACI de l’année précédente.
# Puis on contraint notre portefeuille à respecter la limite la plus exigeante

def prepare_carbon_panel(d, p, projection_panel=None, scenario='Tendance prolongée'):
    """Valide les données annuelles et la couverture complète du benchmark."""
    if d.empty or d.id.isna().any() or d.id.duplicated().any():
        raise ValueError('Univers vide ou identifiants manquants/dupliqués.')
    b = pd.to_numeric(d['benchmark_weight'], errors='raise')
    if not np.isfinite(b).all() or b.lt(0).any() or not np.isclose(b.sum(), 1, atol=1e-7, rtol=0):
        raise ValueError('Le benchmark doit être complet et ses poids totaliser 1.')
    if projection_panel is None:
        panel = d[['id', CI]].copy()
        panel['scenario'], panel['year_offset'], panel['year'] = scenario, 0, pd.NA
    else:
        missing = {'id', 'scenario', 'year_offset', 'year', CI} - set(projection_panel)
        if missing:
            raise ValueError(f'Colonnes carbone absentes : {sorted(missing)}')
        panel = projection_panel.loc[projection_panel.scenario.eq(scenario)].copy()
    if panel.empty or panel.id.isna().any() or panel.duplicated(['id', 'year_offset']).any():
        raise ValueError('Scénario absent ou doublons/identifiants manquants dans les projections.')
    offset = pd.to_numeric(panel.year_offset, errors='raise')
    if not np.isfinite(offset).all() or not offset.eq(np.floor(offset)).all():
        raise ValueError('year_offset doit contenir des entiers finis.')
    panel['year_offset'] = offset.astype(int)
    panel[CI] = pd.to_numeric(panel[CI], errors='raise')
    if not np.isfinite(panel[CI]).all() or panel[CI].lt(0).any():
        raise ValueError('Intensités annuelles absentes, négatives ou infinies.')
    if panel.year.notna().any():
        years = pd.to_numeric(panel.year, errors='raise')
        if not np.isfinite(years).all() or not years.eq(np.floor(years)).all() or (years-panel.year_offset).nunique() != 1:
            raise ValueError('Années calendaires incomplètes ou incohérentes avec year_offset.')
        panel['year'] = years.astype(int)
    for _, block in panel.groupby('year_offset'):
        if set(block.id) != set(d.id):
            raise ValueError('Chaque année doit couvrir exactement tout le benchmark, exclusions comprises.')
    start = p.get('carbon_start_offset')
    if start is not None:
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError('carbon_start_offset doit être un entier ou None.')
        panel = panel.loc[panel.year_offset.ge(start)].copy()
    if panel.empty or 0 not in set(panel.year_offset):
        raise ValueError('La période de contraintes doit inclure Date 2 (year_offset=0).')
    offsets = sorted(panel.year_offset.unique())
    if any(y-x != 1 for x, y in zip(offsets, offsets[1:])):
        raise ValueError('Les années du panel doivent être consécutives.')
    return panel


def carbon_limit(
    d,
    p,
    projection_panel=None,
    scenario='Tendance prolongée'
):
    """
    Calcule le WACI maximal autorisé pour chaque année.

    Le plafond annuel est le minimum entre :

    1. initial_carbon_ratio * WACI du benchmark de l'année ;
    2. 93 % du plafond autorisé de l'année précédente.

    Parameters
    ----------
    d : DataFrame
        Univers contenant au minimum :
        id, benchmark_weight.

    p : dict
        Paramètres de l'optimisation.

    projection_panel : DataFrame
        Résultat de extrapolate_company_data().
        Il doit contenir :
        id, scenario, year, year_offset et CI.

    scenario : str
        Scénario utilisé pour calculer les intensités futures.
    """

    projection_panel = prepare_carbon_panel(d, p, projection_panel, scenario)
    validate_parameters(p)

    required_d = {'id', 'benchmark_weight'}
    missing_d = required_d - set(d.columns)

    if missing_d:
        raise ValueError(
            f'Colonnes absentes de d : {sorted(missing_d)}'
        )

    required_panel = {
        'id',
        'scenario',
        'year',
        'year_offset',
        CI
    }

    missing_panel = required_panel - set(projection_panel.columns)

    if missing_panel:
        raise ValueError(
            'Colonnes absentes de projection_panel : '
            f'{sorted(missing_panel)}'
        )

    # Sélection du scénario
    panel = projection_panel.loc[
        projection_panel['scenario'].eq(scenario)
    ].copy()

    if panel.empty:
        raise ValueError(
            f'Le scénario {scenario!r} est introuvable.'
        )

    # Vérifie qu'il n'existe qu'une valeur par entreprise et par année
    if panel.duplicated(['id', 'year_offset']).any():
        raise ValueError(
            'Plusieurs projections existent pour une même '
            'entreprise et une même année.'
        )

    # On ne conserve que les entreprises présentes dans l'univers
    panel = panel.loc[
        panel['id'].isin(d['id'])
    ].copy()

    # Fusion avec les poids du benchmark
    panel = panel.merge(
        d[['id', 'benchmark_weight']],
        on='id',
        how='left',
        validate='many_to_one'
    )

    if panel['benchmark_weight'].isna().any():
        missing_ids = panel.loc[
            panel['benchmark_weight'].isna(),
            'id'
        ].unique()

        raise ValueError(
            'Poids benchmark absents pour : '
            + ', '.join(map(str, missing_ids))
        )

    if panel[CI].isna().any():
        missing_ids = panel.loc[
            panel[CI].isna(),
            'id'
        ].unique()

        raise ValueError(
            'Intensités carbone absentes pour : '
            + ', '.join(map(str, missing_ids))
        )

    # Ordonne les années chronologiquement
    panel = panel.sort_values('year_offset')

    # Initialisation éventuelle avec une WACI précédente ou une base
    previous_limit = None

    limits_by_year = {}
    first_year = True

    for year_offset, block in panel.groupby(
        'year_offset',
        sort=True
    ):

        # WACI du benchmark pour l'année considérée
        benchmark_waci = float(
            block['benchmark_weight'] @ block[CI]
        )

        benchmark_limit = (
            p['initial_carbon_ratio']
            * benchmark_waci
        )

        limits = {
            'benchmark_50pct': benchmark_limit
        }

        if previous_limit is not None:

            # Pour la première année, on utilise directement
            # la base ou la WACI précédente.
            if first_year:
                previous_year_limit = previous_limit
            else:
                previous_year_limit = (
                    previous_limit
                    * (1 - p['annual_reduction'])
                )

            limits['previous_year_trajectory'] = (
                previous_year_limit
            )

        elapsed = p['years_since_base'] + int(year_offset)
        if p.get('base_waci') is not None and elapsed >= 0:
            limits['base_trajectory'] = float(p['base_waci']) * (1-p['annual_reduction'])**elapsed
        if p.get('previous_waci') is not None and year_offset >= 0:
            limits['previous_observed_trajectory'] = float(p['previous_waci']) * (1-p['annual_reduction'])**(int(year_offset)+1)

        # Le plafond annuel est le plus contraignant
        max_waci = min(limits.values())

        # Si la colonne year est renseignée, on l'utilise comme clé.
        # Sinon, on utilise year_offset.
        year_values = block['year'].dropna()

        if year_values.empty:
            year_key = int(year_offset)
        else:
            year_key = int(year_values.iloc[0])

        limits_by_year[year_key] = float(max_waci)

        # Le plafond de cette année devient la référence
        # pour l'année suivante.
        previous_limit = max_waci
        first_year = False

    return limits_by_year


if __name__ == '__main__':
    carbon_limits = carbon_limit(
        candidates,
        p,
        projection_panel=panel,
        scenario='Tendance prolongée'
    )

if __name__ == '__main__':
    print(carbon_limits)


#%% 

# Maintenant, on cherche la meilleur combinaison possible d'entreprise 
# compatible avec la WACI, la diversification, la taxonomie et le cœur climatique.
# Il s'agit d'une optimisation linéaire mixte en nombres entiers (MILP) : 
# On possèdes des variables entières binaires (pour la sélection des entreprises) et des variables continues (pour les poids des entreprises).
# On maximise la somme des scores pondérés par les poids des entreprises, tout en respectant les contraintes de poids minimum et maximum, ainsi que les contraintes de diversification imposées par les variables binaires et continues.

def optimize_portfolio(d, p, projection_panel=None, scenario='Tendance prolongée'):

    validate_parameters(p)
    if projection_panel is not None:
        projection_panel = prepare_carbon_panel(d, p, projection_panel, scenario)
    # On conserve uniquement les entreprises admissibles
    a = d.loc[d.admissible].copy().reset_index(drop=True)
    n = len(a)

    # On vérifie que le nombre de candidats est suffisant pour le nombre de positions souhaité. Si ce n’est pas le cas, on lève une exception.
    if n < p['n_positions']:
        raise ValueError(f'Seulement {n} candidats pour {p["n_positions"]} positions.')

    # Recense les secteurs et régions disponibles
    groups = [(col, value) for col in ['sector', 'region'] for value in sorted(a[col].unique())]
    # Il s'agit de la taille du problème d'optimisation : 2*n variables pour les poids et les sélections binaires, 
    # plus une variable binaire pour chaque groupe de diversification.
    # On construit donc un vecteur x contenant les poids w, les sélections binaires z et les présences de groupes y.
    size = 2*n + len(groups)
    # Impose les bornes inférieures et supérieures pour les variables d'optimisation :
    # Toutes les variables ont une borne inférieure de zéro.
    # Les n premières, correspondant aux poids, ont pour maximum max_weight
    lower = np.zeros(size); upper = np.ones(size)
    upper[:n] = p['max_weight']
    # Ici on indique au solveur quelles sont les variables qui doivent être entières ou continues. 
    integrality = np.r_[np.zeros(n), np.ones(n+len(groups))]
    rows, lbs, ubs = [], [], []

    # Dorénavant, on définit une fonction add qui permet d'ajouter des contraintes linéaires au problème d'optimisation.
    # Cette fonction prend en entrée les indices des variables concernées, les coefficients de la contrainte, ainsi que les bornes inférieure et supérieure de la contrainte.
    def add(indices, values, lo=-np.inf, hi=np.inf):
        row = np.zeros(size); row[np.asarray(indices, dtype=int)] = values
        rows.append(row); lbs.append(lo); ubs.append(hi)
    
    wi = np.arange(n)
    zi = n + wi

    # On impose que la somme des poids soit égale à 1
    add(wi, 1, 1, 1)

    # On impose que le nombre de positions sélectionnées soit exactement n_positions.
    add(zi, 1, p['n_positions'], p['n_positions'])

    # Si une entreprise n'est pas sélectionnée (zi = 0), on impose à son poids d'être nul. 
    # Inversement, si une entreprise est sélectionnée (zi = 1), son poids doit être compris entre min_weight et max_weight.
    for i in wi:
        add([i, n+i], [1, -p['max_weight']], hi=0)
        add([i, n+i], [1, -p['min_weight']], lo=0)

    # On impose la contrainte de WACI maximale pour le portefeuille
    
    # Contraintes carbone
    if projection_panel is None:
        # Fonctionnement classique avec une seule WACI
        limits = carbon_limit(d, p)
        cap = limits[0]

        add(
            wi,
            a[CI].to_numpy(),
            hi=cap
        )

    else:
        # Calcul d'un plafond pour chaque année
        limits = carbon_limit(
            d,
            p,
            projection_panel=projection_panel,
            scenario=scenario
        )

        projected = projection_panel.loc[
            projection_panel['scenario'].eq(scenario)
        ].copy()

        if projected.empty:
            raise ValueError(
                f'Scénario introuvable : {scenario}'
            )

        # Vérification de l'unicité entreprise/année
        if projected.duplicated(
            ['id', 'year_offset']
        ).any():
            raise ValueError(
                'Doublons dans les projections annuelles.'
            )

        # On ajoute une contrainte WACI pour chaque année
        for year_offset, block in projected.groupby(
            'year_offset',
            sort=True
        ):

            # Clé utilisée dans le dictionnaire des plafonds
            year_values = block['year'].dropna()

            if year_values.empty:
                year_key = int(year_offset)
            else:
                possible_year = int(year_values.iloc[0])

                if possible_year in limits:
                    year_key = possible_year
                else:
                    year_key = int(year_offset)

            if year_key not in limits:
                raise ValueError(
                    f'Plafond WACI absent pour {year_key}.'
                )

            # Association : id de l'entreprise -> intensité de l'année
            ci_by_id = block.set_index('id')[CI]

            # Réordonne les intensités dans le même ordre que les candidats
            ci_year = a['id'].map(ci_by_id)

            if ci_year.isna().any():
                missing_ids = a.loc[
                    ci_year.isna(),
                    'id'
                ].tolist()

                raise ValueError(
                    f'Intensité manquante pour {year_key} : '
                    + ', '.join(map(str, missing_ids))
                )

            ci_year = pd.to_numeric(
                ci_year,
                errors='raise'
            )

            if not np.isfinite(ci_year).all() or ci_year.lt(0).any():
                raise ValueError(
                    f"Intensités invalides pour l'année {year_key}.")

            # Contrainte :
            # somme_i (poids_i * intensité_i_année)
            # <= plafond WACI de l'année
            add(
                wi,
                ci_year.to_numpy(),
                hi=float(limits[year_key])
            )

    # On impose la contrainte de taxonomie minimale pour le portefeuille
    add(wi, a['taxonomy'].fillna(0), lo=p['taxonomy_min'])
    add(wi, a['taxonomy'].notna().astype(float), lo=p['taxonomy_coverage_min'])

    # On impose la contrainte sur le coeur climatique pour le portefeuille : 
    # le poids total des entreprises appartenant au coeur climatique doit être compris entre core_min et core_max.
    add(wi, a['core'].astype(float), p['core_min'], p['core_max'])

    # Pour chaque dimension de diversification (secteur, pays, région), on impose que le poids total des entreprises appartenant à un même groupe 
    # ne dépasse pas la limite maximale définie dans les paramètres.
    for col in ['sector', 'country', 'region']:
        for _, idx in a.groupby(col).groups.items():
            add(list(idx), 1, hi=p['max_'+col])

    # On fait ici le lien entre les variables binaires de sélection d'entreprise et les variables binaires de présence de groupe.
    # Si une entreprise est sélectionnée (zi = 1), alors le groupe auquel elle appartient doit être présent (y = 1).
    # Inversement, si une entreprise n'est pas sélectionnée (zi = 0), le groupe peut être présent ou non.
    for j, (col, value) in enumerate(groups):
        idx = np.flatnonzero(a[col].eq(value)); y = 2*n+j
        add(np.r_[n+idx, y], np.r_[np.ones(len(idx)), -len(idx)], hi=0)
        add(np.r_[n+idx, y], np.r_[np.ones(len(idx)), -1], lo=0)

    # On impose un nombre minimum de secteurs et de régions dans le portefeuille.
    for col, minimum in [('sector', p['min_sectors']), ('region', p['min_regions'])]:
        add([2*n+j for j, (c, _) in enumerate(groups) if c == col], 1, lo=minimum)


    # On définit maintenant le score à maximiser
    # Le solveur milp minimise une fonction. On lui donne donc les scores avec un signe négatif pour qu’il maximise le score total du portefeuille.
    objective = np.zeros(size); objective[:n] = -a['objective_score']
    res = milp(objective, integrality=integrality, bounds=Bounds(lower, upper),
               constraints=LinearConstraint(np.array(rows), lbs, ubs),
               options={'time_limit': p['time_limit'], 'mip_rel_gap': p['mip_rel_gap']})

    
    if res.status != 0:
        raise RuntimeError(f'Optimisation non certifiée (statut {res.status}) : {res.message}. '
                           'Aucun portefeuille validé. Consulter diagnostic_candidats.csv ; '
                           'ne modifier les seuils qu’après justification.')

    
    weights = pd.Series(res.x[:n], index=a.id)
    result = d.copy(); result['weight_date2'] = result.id.map(weights).fillna(0.)
    result['weight_date2'] = result['weight_date2'].mask(result['weight_date2'].abs() < 1e-9, 0.)
    return result, {'message': str(res.message), 'mip_gap': float(res.mip_gap),
                    'objective_value': float(-res.fun), 'carbon_limits': limits, 'scenario': scenario}



# On vérifie ensuite que le portefeuille final respecte bien toutes les contraintes imposées.
    
# Fonction permettant de vérier qu'une contrainte est respectée. 
# Elle prend en entrée le nom de la contrainte, la valeur calculée, ainsi que les bornes inférieure et supérieure.

def audit_constraints(
    d,
    p,
    projection_panel=None,
    scenario='Tendance prolongée'
):
    """
    Vérifie les contraintes du portefeuille, notamment
    les contraintes WACI pour chaque année simulée.
    """

    if projection_panel is not None:
        projection_panel = prepare_carbon_panel(d, p, projection_panel, scenario)
    w = pd.to_numeric(d['weight_date2'], errors='raise')
    if not np.isfinite(w).all():
        raise ValueError('Poids du portefeuille absents ou infinis.')
    held = w > 1e-7
    rows = []

    def check(
        name,
        value,
        lo=-np.inf,
        hi=np.inf
    ):
        """
        Ajoute une contrainte au tableau de contrôle.
        """

        value = float(value)

        rows.append({
            'contrainte': name,
            'valeur': value,
            'minimum': lo,
            'maximum': hi,
            'respectee': bool(
                np.isfinite(value) and
                lo - 1e-7
                <= value
                <= hi + 1e-7
            )
        })

    # Contraintes générales
    check(
        'Somme poids',
        w.sum(),
        1,
        1
    )

    check(
        'Nombre positions',
        held.sum(),
        p['n_positions'],
        p['n_positions']
    )

    if held.any():
        check(
            'Poids minimum détenu',
            w.loc[held].min(),
            p['min_weight']
        )
    else:
        check(
            'Poids minimum détenu',
            0,
            p['min_weight']
        )

    check(
        'Poids maximum',
        w.max(),
        hi=p['max_weight']
    )

    check(
        'Long only',
        w.min(),
        0
    )

    check(
        'Poids non admissibles',
        w.loc[~d['admissible']].sum(),
        0,
        0
    )

    if projection_panel is None:

        # Contrôle de compatibilité avec l'ancien fonctionnement
        benchmark_waci = float(
            d['benchmark_weight'] @ d[CI]
        )

        current_limit = carbon_limit(d, p)[0]

        check(
            'WACI S1–2 actuelle',
            w @ d[CI],
            hi=current_limit
        )

    else:

        required = {
            'id',
            'scenario',
            'year_offset',
            'year',
            CI
        }

        missing = required - set(projection_panel.columns)

        if missing:
            raise ValueError(
                'Colonnes absentes de projection_panel : '
                f'{sorted(missing)}'
            )

        projected = projection_panel.loc[
            projection_panel['scenario'].eq(scenario)
        ].copy()

        if projected.empty:
            raise ValueError(
                f'Scénario introuvable : {scenario}'
            )

        if projected.duplicated(
            ['id', 'year_offset']
        ).any():
            raise ValueError(
                'Doublons dans projection_panel pour '
                'une entreprise et une année.'
            )

        # On recalcule les plafonds annuels
        carbon_limits = carbon_limit(
            d,
            p,
            projection_panel=projection_panel,
            scenario=scenario
        )

        # Une contrainte WACI est contrôlée par année
        for year_offset, block in projected.groupby(
            'year_offset',
            sort=True
        ):

            # Identification de la clé du dictionnaire carbon_limits
            year_values = block['year'].dropna()

            if year_values.empty:
                year_key = int(year_offset)
            else:
                possible_year = int(
                    year_values.iloc[0]
                )

                if possible_year in carbon_limits:
                    year_key = possible_year
                else:
                    year_key = int(year_offset)

            if year_key not in carbon_limits:
                raise ValueError(
                    f'Plafond WACI absent pour {year_key}.'
                )

            # Intensité simulée de chaque entreprise
            ci_by_id = block.set_index('id')[CI]

            # Réorganisation selon l'ordre des lignes de d
            ci_year = d['id'].map(ci_by_id)

            if ci_year.isna().any():
                missing_ids = d.loc[
                    ci_year.isna(),
                    'id'
                ].tolist()

                raise ValueError(
                    f'Intensités manquantes pour {year_key} : '
                    + ', '.join(map(str, missing_ids))
                )

            ci_year = pd.to_numeric(
                ci_year,
                errors='raise'
            )

            # WACI du portefeuille pour l'année considérée
            portfolio_waci = float(
                w @ ci_year
            )

            # Plafond annuel calculé précédemment
            annual_limit = float(
                carbon_limits[year_key]
            )

            check(
                f'WACI S1–2 année {year_key}',
                portfolio_waci,
                hi=annual_limit
            )

    # Autres contraintes environnementales
    check(
        'Taxonomie borne basse',
        w @ d['taxonomy'].fillna(0),
        p['taxonomy_min']
    )

    check(
        'Couverture taxonomie',
        w @ d['taxonomy'].notna(),
        p['taxonomy_coverage_min']
    )

    check(
        'Cœur climatique',
        w @ d['core'],
        p['core_min'],
        p['core_max']
    )

    # Contraintes de diversification
    for col in ['sector', 'country', 'region']:

        for group, exposure in (
            d.groupby(col)['weight_date2'].sum().items()
        ):
            check(
                f'{col}: {group}',
                exposure,
                hi=p[f'max_{col}']
            )

    check(
        'Nombre secteurs',
        d.loc[held, 'sector'].nunique(),
        p['min_sectors']
    )

    check(
        'Nombre régions',
        d.loc[held, 'region'].nunique(),
        p['min_regions']
    )

    checks = pd.DataFrame(rows)

    if not checks['respectee'].all():

        failed = checks.loc[
            ~checks['respectee'],
            'contrainte'
        ].tolist()

        raise RuntimeError(
            'Échec contrôle indépendant : '
            + ', '.join(failed)
        )

    return checks




# %% 7. Optimiser les poids

if __name__ == '__main__':
    candidates.to_csv(output_dir / 'diagnostic_candidats.csv', index=False)
if __name__ == '__main__':
    result, solver_info = optimize_portfolio(candidates, p, projection_panel=panel, scenario='Tendance prolongée')

if __name__ == '__main__':
    print("Message du solveur :", solver_info["message"])
if __name__ == '__main__':
    print(f"Écart d'optimalité : {solver_info['mip_gap']:.8f}")
if __name__ == '__main__':
    print(
        "Score final pondéré sur 100 :",
        round(100 * solver_info["objective_value"], 2)
    )

if __name__ == '__main__':
    checks = audit_constraints(result, p, projection_panel=panel, scenario='Tendance prolongée')

if __name__ == '__main__':
    print(checks.to_string(index=False))



#%% 


if __name__ == '__main__':
    portfolio_date2 = (
        result.loc[result["weight_date2"] > 1e-7]
        .sort_values("weight_date2", ascending=False)
        .copy()
    )

if __name__ == '__main__':
    portfolio_date2["weight_pct"] = portfolio_date2["weight_date2"] * 100

if __name__ == '__main__':
    portfolio_date2["amount_eur_m"] = (
        portfolio_date2["weight_date2"] * p["aum_eur_m"]
    )

if __name__ == '__main__':
    columns = [
        "id", "name", "sector", "country", "region",
        "weight_pct", "amount_eur_m",
        "esg_score_95", "return_score_5", "final_score_100", "core"
    ]

if __name__ == '__main__':
    print(portfolio_date2[columns].to_string(index=False))



#%%

# Cette fonction calcule les indicateurs du benchmark, du portefeuille initial (Date 1) et du portefeuille optimisé (Date 2), 
# puis les regroupe dans un tableau pour les comparer.

def portfolio_metrics(d):
    rows = []
    for label, col in [('Benchmark', 'benchmark_weight'), ('Date 1', 'weight_date1'), ('Date 2', 'weight_date2')]:
        w = d[col]; known = d.return_12m_pct.notna(); coverage = float(w @ known)
        rows.append({'allocation': label, 
                     'positions': int((w > 1e-7).sum()),
                     'waci_s12': w @ d[CI], 
                     'taxonomy_lower_bound': w @ d.taxonomy.fillna(0),
                     'core_weight': w @ d.core,
                     'esg_score95_lower_bound': w @ d.esg_score_95.fillna(0),
                     'final_score100_lower_bound': w @ d.final_score_100.fillna(0),
                     'esg_coverage': w @ d.esg_score_95.notna(),
                     'return_coverage': coverage,
                     'historical_return_pct': float(w @ d.return_12m_pct.fillna(0)) if coverage >= 1-1e-9 else np.nan})

    result = pd.DataFrame(rows).set_index('allocation')
    result['historical_active_return_pp'] = result.historical_return_pct - result.loc['Benchmark', 'historical_return_pct']
    return result


if __name__ == '__main__':
    metrics = portfolio_metrics(result)

if __name__ == '__main__':
    print(metrics.to_string())


#%% 

# Cette fonction compare les allocations de Date 1 et Date 2, en calculant les contributions en points de pourcentage, 
# les décisions d'achat ou de vente, ainsi que les raisons de sélection ou de rejet des entreprises.
def compare_allocations(d, metrics, p):

    c = pd.DataFrame(index=d.index)

    # Identifiants pour reconnaître chaque entreprise
    for col in ('id', 'isin', 'name'):
        if col in d.columns:
            c[col] = d[col]

    c['delta_weight'] = d.weight_date2 - d.weight_date1
    c['trade_eur_m'] = c.delta_weight * p['aum_eur_m']
    c['amount_date2_eur_m'] = d.weight_date2 * p['aum_eur_m']
    c['delta_benchmark'] = d.weight_date2 - d.benchmark_weight

    c['contribution_ecart_rendement_benchmark'] = (
        c.delta_benchmark * d.return_12m_pct
    )
    c['contribution_ecart_rendement_date1'] = (
        c.delta_weight * d.return_12m_pct
    )

    return c


if __name__ == '__main__':
    comparison = compare_allocations(result, metrics, p)
if __name__ == '__main__':
    print(comparison.to_string())

#%% 


def _save_plot(fig, output, filename, show=False):
    """Enregistre la figure et retourne son chemin ; affichage optionnel."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    if not getattr(fig, '_layout_prepared', False):
        fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_weights(c, output, show=False):
    """Poids Date 2, colorés selon la poche, en pourcentage."""
    held = c.loc[c.weight_date2 > 1e-7].sort_values('weight_date2')
    fig, ax = plt.subplots(figsize=(11, max(5, len(held)*.30)))
    bars = ax.barh(held['name'], held.weight_date2*100,
                   color=np.where(held.core, '#16836b', '#d99c30'))
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
    ax.set(xlabel='Poids (%)', title='GreenInvestment — Date 2\nVert : cœur climatique ; or : complémentaire')
    ax.margins(x=.16)
    return _save_plot(fig, output, 'poids_date2.png', show)


def plot_exposures(c, output, dimension='sector', show=False):
    """Compare les poids du benchmark, de Date 1 et de Date 2."""
    if dimension not in ['sector', 'country', 'region']:
        raise ValueError('Dimension attendue : sector, country ou region.')
    exp = c.groupby(dimension)[['benchmark_weight', 'weight_date1', 'weight_date2']].sum()
    exp = exp.sort_values('weight_date2')*100
    exp.columns = ['Benchmark', 'Date 1', 'Date 2']
    ax = exp.plot.barh(figsize=(11, max(5, len(exp)*.5)),
                      color=['#a8b2bd', '#426ba2', '#16836b'])
    ax.set(xlabel='Poids (%)', ylabel='', title='GreenInvestment — '+dimension)
    return _save_plot(ax.figure, output, 'exposition_'+dimension+'.png', show)


def plot_metrics(metrics, output, show=False):
    """Score final /100 et indicateurs disponibles dans portfolio_metrics."""
    specs = [
        ('final_score100_lower_bound', 'Score final pondéré /100 — borne basse', 1),
        ('waci_s12', 'WACI S1–2 (tCO₂e/M€ de CA)', 1),
        ('taxonomy_lower_bound', 'Taxonomie — borne basse (%)', 100),
        ('core_weight', 'Cœur climatique (%)', 100),
        ('historical_return_pct', 'Rendement 12 mois rétrospectif (%)', 1),
        ('historical_active_return_pp', 'Écart de rendement au benchmark (points)', 1)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (col, label, factor) in zip(axes.flat, specs):
        values = metrics[col]*factor
        bars = ax.bar(metrics.index, values, color=['#a8b2bd', '#426ba2', '#16836b'])
        ax.bar_label(bars, labels=[f'{v:.2f}' if np.isfinite(v) else 'N/D' for v in values], padding=4)
        ax.axhline(0, color='#555555', linewidth=.6)
        ax.set_title(label, fontsize=11)
        finite = values[np.isfinite(values)]
        low = min(0., float(finite.min())) if len(finite) else 0.
        high = max(0., float(finite.max())) if len(finite) else 1.
        pad = max(high-low, 1.)*.18
        ax.set_ylim(low-pad if low < 0 else 0., high+pad)
    fig.suptitle('GreenInvestment — comparaison des allocations', fontsize=15)
    return _save_plot(fig, output, 'comparaison_indicateurs.png', show)

def plot_extrapolated_evolutions(
    panel,
    output,
    scenario='Tendance prolongée',
    ids=None,
    show=False
):

    required = {
        'id',
        'scenario',
        'year_offset',
        'scope1_tco2e',
        'scope2_tco2e',
        'revenue_eur_m'
    }

    missing = required - set(panel.columns)

    if missing:
        raise ValueError(
            f'Colonnes absentes du panel : {sorted(missing)}'
        )

    data = panel.loc[
        panel['scenario'].eq(scenario)
    ].copy()

    if data.empty:
        raise ValueError(
            f'Le scénario {scenario!r} est introuvable.'
        )

    if ids is not None:
        if isinstance(ids, str):
            ids = [ids]
        absent = set(ids) - set(data.id)
        if absent:
            raise ValueError(f'Identifiants inconnus : {sorted(absent)}')
        data = data.loc[data['id'].isin(ids)]

        if data.empty:
            raise ValueError(
                'Aucune entreprise correspondant aux identifiants fournis.'
            )

    # Si aucune entreprise n'est spécifiée,
    # on trace l'évolution moyenne de l'univers.
    aggregate = ids is None

    if aggregate:
        plot_data = (
            data.groupby('year_offset', as_index=False)[
                [
                    'scope1_tco2e',
                    'scope2_tco2e',
                    'revenue_eur_m'
                ]
            ]
            .mean()
            .sort_values('year_offset')
        )

        x = plot_data['year_offset']

    else:
        plot_data = data.sort_values(
            ['id', 'year_offset']
        )

        x = None

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, 6),
        sharex=True
    )

    variables = [
        (
            'revenue_eur_m',
            'Chiffre d’affaires',
            'CA (millions €)',
            '#426ba2'
        ),
        (
            'scope1_tco2e',
            'Scope 1',
            'Scope 1 (tCO₂e)',
            '#c44e52'
        ),
        (
            'scope2_tco2e',
            'Scope 2',
            'Scope 2 (tCO₂e)',
            '#55a868'
        )
    ]

    for ax, (column, title, ylabel, color) in zip(
        axes,
        variables
    ):

        if aggregate:
            ax.plot(
                x,
                plot_data[column],
                color=color,
                linewidth=2.5,
                marker='o',
                markersize=4,
                label='Moyenne de l’univers'
            )

        else:
            for company_id, group in plot_data.groupby('id'):
                ax.plot(
                    group['year_offset'],
                    group[column],
                    linewidth=1.5,
                    alpha=.8,
                    label=str(company_id)
                )

        # Ligne verticale correspondant à Date 2
        ax.axvline(
            0,
            color='black',
            linestyle='--',
            linewidth=.9,
            label='Date 2'
        )

        ax.set_title(
            f'{title}\nScénario : {scenario}',
            fontsize=11
        )

        ax.set_xlabel(
            'Années par rapport à Date 2'
        )

        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)

        # Différenciation graphique du passé et du futur
        ax.axvspan(
            x.min() if aggregate else plot_data['year_offset'].min(),
            0,
            color='#eeeeee',
            alpha=.5
        )

        if aggregate or len(ids) <= 10:
            ax.legend(
                fontsize=8,
                loc='best'
            )

    fig.suptitle(
        'GreenInvestment — évolution simulée des données carbone et du CA',
        fontsize=15
    )

    fig.text(
        0.5,
        0.01,
        'Zone grisée : passé reconstitué | Zone blanche : futur projeté',
        ha='center',
        fontsize=9
    )

    fig.tight_layout(
        rect=[0, 0.04, 1, 0.92]
    )

    fig._layout_prepared = True
    suffix = 'univers' if ids is None else '_'.join(map(str, ids))
    suffix = re.sub(r'[^\w-]+', '_', scenario + '_' + suffix)
    return _save_plot(
        fig,
        output,
        'evolution_scope_ca_' + suffix + '.png',
        show
    )

from pathlib import Path
import pandas as pd


def export_results_excel(results, file_path, index=False):
    """Exporte les résultats vers un fichier Excel."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_excel(file_path, index=index, sheet_name="Resultats")
    print(f"Fichier créé : {file_path}")


if __name__ == '__main__':
    metrics_results = metrics
    allocations_results = comparison

    export_results_excel(
        metrics_results,
        "Results/portfolio_metrics.xlsx",
        index=True
    )

    export_results_excel(
        allocations_results,
        "Results/compare_allocations.xlsx"
    )

if __name__ == '__main__':
    plot_extrapolated_evolutions(
        panel=panel,
        output=output_dir,
        scenario='Tendance prolongée',
        ids=['edf', 'apple', 'orange']
    )

if __name__ == '__main__':
    plot_extrapolated_evolutions(
        panel=panel,
        output=output_dir,
        scenario='Tendance prolongée'
    )





REVIEW_NOTES = [
    "Le score final inclut 95 points ESG et 5 points de rendement passé. Le commentaire initial sur l'absence de rendement dans la sélection est obsolète, conservé à la demande de l'auteur.",
    "L'intensité de référence est recalculée avec S1+S2 et CA pour être identique aux projections ; ci_provided_source conserve la valeur initiale arrondie.",
    "Les poids Date 2 sont constants sur toutes les années contraintes. Le passé est un test rétrospectif simulé du portefeuille choisi aujourd'hui, pas son historique réel.",
    "carbon_start_offset=None impose toutes les années disponibles ; 0 limite les contraintes à Date 2 et au futur. Ce choix change la date de départ du plafond récursif.",
    "La règle porte sur le plafond précédent : elle ne garantit pas une baisse de 7 % du WACI effectivement atteint si celui-ci était déjà bien inférieur au plafond.",
    "previous_waci désigne la WACI réalisée en année -1. base_waci se rapporte à l'année -years_since_base. Les deux limites sont prises en compte conjointement.",
    "Le périmètre de la variation sur cinq ans est supposé comparable aux scopes 1 et 2. Les deux scopes suivent le même taux ; les CA passé et futur sont des hypothèses.",
    "Le filtrage repose sur eligible_same_gate, EXCLUDED si présent, et les contrôles de prepare_candidates. FLAGS et les paramètres de controverses/pétrole-gaz ne déclenchent pas ici de filtres supplémentaires ; le screening est délégué aux collègues.",
    "score_coverage_min et objective sont des paramètres hérités sans effet : le score disponible est esg_score_95, complété par le classement des rendements ; sa couverture par composante n'est pas disponible.",
    "Aucune hypothèse ni limite n'est assouplie automatiquement si l'optimisation est impossible."
]


def annual_waci_table(d, p, panel, scenario='Tendance prolongée'):
    """Rapport annuel à poids constants et plafond du scénario choisi."""
    selected = prepare_carbon_panel(d, p, panel, scenario)
    limits = carbon_limit(d, p, selected, scenario)
    rows = []
    for offset, block in selected.groupby('year_offset', sort=True):
        key = int(block.year.iloc[0]) if block.year.notna().all() else int(offset)
        intensity = d.id.map(block.set_index('id')[CI]).to_numpy(float)
        row = {'year_offset': int(offset), 'annee': key, 'scenario': scenario, 'waci_max': limits[key]}
        for name, col in [('benchmark', 'benchmark_weight'), ('date1', 'weight_date1'), ('date2', 'weight_date2')]:
            row['waci_'+name] = float(d[col].to_numpy(float) @ intensity)
        row['respectee'] = row['waci_date2'] <= limits[key]+1e-7
        rows.append(row)
    return pd.DataFrame(rows)


def plot_annual_waci(table, output):
    """Trace le WACI simulé des allocations et les plafonds réellement imposés."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for col, label, color in [('waci_benchmark', 'Benchmark', '#88939c'), ('waci_date1', 'Date 1', '#426ba2'), ('waci_date2', 'Date 2', '#16836b'), ('waci_max', 'Plafond imposé', '#111111')]:
        ax.plot(table.annee, table[col], '--' if col=='waci_max' else '-', label=label, color=color, marker='o', markersize=3)
    ax.set(xlabel='Année (relative à Date 2 si non datée)', ylabel='WACI (tCO2e / M€ de CA)', title='GreenInvestment — WACI simulé à poids constants')
    ax.grid(alpha=.2)
    ax.legend()
    return _save_plot(fig, output, 'waci_annuel.png')

# %% Créer le dossier de résultats

if __name__ == '__main__':
    print("Dossier de résultats :", output_dir.resolve())


# Générer et enregistrer les graphiques

if __name__ == '__main__':
    plot_weights(comparison, output_dir)

if __name__ == '__main__':
    for dimension in ['sector', 'country', 'region']:
        plot_exposures(comparison, output_dir, dimension)

if __name__ == '__main__':
    plot_metrics(metrics, output_dir)
if __name__ == '__main__':
    waci_annual = annual_waci_table(result, p, panel)
if __name__ == '__main__':
    plot_annual_waci(waci_annual, output_dir)


# Exporter les tableaux dans Excel

if __name__ == '__main__':
    excel_path = output_dir / "GreenInvestment_resultats.xlsx"

if __name__ == '__main__':
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        panel.to_excel(writer, sheet_name='Projections', index=False)
        pd.DataFrame(list(solver_info['carbon_limits'].items()), columns=['annee', 'waci_max']).to_excel(writer, sheet_name='Plafonds_WACI', index=False)
        waci_annual.to_excel(writer, sheet_name='WACI_annuel', index=False)
        pd.DataFrame([{'parametre': k, 'valeur': json.dumps(v, ensure_ascii=False)} for k, v in {**p, 'projection': PROJECTION}.items()]).to_excel(writer, sheet_name='Parametres', index=False)
        pd.DataFrame({'note': REVIEW_NOTES}).to_excel(writer, sheet_name='Notes_methodologiques', index=False)
        portfolio_date2.to_excel(
            writer, sheet_name="Portefeuille_Date2", index=False
        )
        comparison.to_excel(
            writer, sheet_name="Comparaison", index=False
        )
        metrics.to_excel(
            writer, sheet_name="Indicateurs"
        )
        checks.to_excel(
            writer, sheet_name="Controles", index=False
        )
        candidates.to_excel(
            writer, sheet_name="Audit_candidats", index=False
        )

if __name__ == '__main__':
    print("Export terminé :", excel_path.resolve())
