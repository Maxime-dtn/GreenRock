#%%
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from normalisation_data import load_data, deduplicate, clean_universe, parse_number, parse_bool, _clean_text, prepare_data, chemin_fichier
from filtrage_ESG import remplir_excluded_aleatoirement

#%%

## IMPORT DES DONNEES 
universe, controversies = load_data()
df, removed, controversies = prepare_data(universe, controversies)
print(df.head())

df = remplir_excluded_aleatoirement(df)

# %%

## Paramètres de diversification : 
n_holdings = 30              # PROPOSITION ; brief : 20 à 40
min_weight = .02             # PROPOSITION
max_weight = .05             # PROPOSITION
max_sector = .25             # PROPOSITION
max_country = .40            # PROPOSITION
max_region = .70             # PROPOSITION
min_sectors = 5              # PROPOSITION
min_regions = 2              # PROPOSITION

## Contraintes sur le portefeuille : 
core_min = .70
core_max = .80
initial_carbon_ratio = .50
annual_reduction = .07
taxonomy_min = .10            # PROPOSITION, % pondéré / 100
taxonomy_coverage_min = .80   # PROPOSITION, poids couverts

years_since_base = 0
base_waci = None
previous_waci = None


## Paramètres pour faire parti du coeur : 
score_coverage_min = .60      # PROPOSITION, (disposer de suffisamment de données pour calculer le score ESG)
minimum_score = .25           # PROPOSITION, (score ESG minimum pour être éligible)
core_taxonomy_pct = 20.       # PROPOSITION, (Une entreprise est classée dans le cœur climatique si au moins 20 % de son chiffre d’affaires est aligné sur la taxonomie ou si ...)
core_green_capex_pct = 30.    # PROPOSITION, (Au moins 30 % de ses dépenses d’investissement sont vertes ; son plan de transition est noté 2/2 ; et sa trajectoire est vérifiée : statut SBTi validé accepté par le code ou indicateur explicite de vérification par un tiers.)
sector_carbon_quantile = .90  # PROPOSITION, (Ecarte les entreprises les plus intensives de chaque secteur)
high_emitter_quantile = .75   # PROPOSITION, (Écarte les entreprises les plus intensives de l’univers investissable)
time_limit = 90.              # PROPOSITION, (limite le temps du solveur)

# %%

def validate():

    if not 20 <= n_holdings <= 40:
        raise ValueError('Le brief impose 20 à 40 titres.')
    
    if not (0 < min_weight <= max_weight <= 1):
        raise ValueError('Bornes de poids invalides.')

    ## Cette partie permet de vérifier qu'il est possible de construire un portefeuille avec les contraintes de poids et le nombre d'entreprises sélectionnées.
    if not n_holdings*max_weight <= 1 <= n_holdings*min_weight:
        raise ValueError('Nombre de lignes incompatible avec les bornes de poids.')
    
    for key in ('max_sector','max_country','max_region','taxonomy_min',
                'taxonomy_coverage_min','score_coverage_min','minimum_score',
                'sector_carbon_quantile','high_emitter_quantile','initial_carbon_ratio'):
        
        if not 0 <= getattr(key) <= 1:
            raise ValueError(f'{key} doit être entre 0 et 1.')
        
    if not 0 <= core_min <= core_max <= 1:
        raise ValueError('Bornes cœur climatique invalides.')
    
    if not 0 <= annual_reduction < 1:
        raise ValueError('Trajectoire invalide.')
    
    if years_since_base and base_waci is None:
        raise ValueError('Renseigner base_waci réalisé à la date de base.')
    
    for v in (base_waci,previous_waci):
        if v is not None and (not np.isfinite(v) or v < 0):
            raise ValueError('WACI historique invalide.')
        
    if not (1 <= min_sectors <= n_holdings and 1 <= min_regions <= n_holdings):
        raise ValueError('Nombre minimal de secteurs/régions invalide.')









def get_investable_universe(df):
    """
    Retourne uniquement les entreprises qui passent
    tous les filtres d'exclusion.
    """

    investable = df[df["EXCLUDED"] == 0].copy()

    if len(investable) == 0:
        raise ValueError(
            "Aucune entreprise ne reste dans l'univers investissable."
        )
    return investable


df_investable = get_investable_universe(df)
carbon_intensity = df['carbon_intensity'].tolist()
esperance = df['esperance'].tolist()
cov = df[['covariance_1', 'covariance_2', 'covariance_3']].values
poids = [0]*len(carbon_intensity)
lambda_risk = 5

#%% 

def calculate_waci(weights, carbon_intensity): 
    return np.dot( weights, carbon_intensity )

def calculate_benchmark_waci(carbon): 
    benchmark_weights = carbon["BenchmarkWeight"].values 
    carbon_intensity = carbon["CarbonIntensity"].values 
    waci = calculate_waci( benchmark_weights, carbon_intensity ) 
    return waci

## Il est nécessaire d'imposer WACI =< 1/2 * WACI du benchmark

## Problème d'optimisation sous contraintes : minimiser WACI sous les contraintes de filtrage ESG et de poids des actifs.
## Dans ce cas, on peut utiliser des bibliothèques d'optimisation comme scipy.optimize ou cvxpy pour résoudre le problème.
## Les contraintes sont linéaires et admettent un pour pour lesquels le problème est réalisable.
## De plus la fonction d'optimisation est concave et on recherche un maximum.
## Finalement, la solution optimale est unique

## Les contraintes : 

def create_carbon_constraint( carbon_intensity, carbon_limit ): 
    def constraint(weights): 
        portfolio_waci = calculate_waci( weights, carbon_intensity ) 
        return carbon_limit - portfolio_waci 
    return constraint

def create_constraints( carbon_intensity, carbon_limit ): 
    constraints = [ # Les poids doivent sommer à 100 % 
        { "type": "eq", "fun": lambda w: np.sum(w) - 1 }, 
        # WACI <= limite# 
        { "type": "ineq", "fun": create_carbon_constraint( carbon_intensity, carbon_limit ) } 
        ] 
    return constraints

## Markovitz 

def markowitz_objective( weights, mu, covariance, risk_aversion ): 
    portfolio_return = ( weights @ mu ) 
    portfolio_variance = ( weights @ covariance @ weights ) 
    utility = ( portfolio_return - risk_aversion / 2 * portfolio_variance ) 
    return -utility

def optimize_markowitz( mu, covariance, carbon_intensity, carbon_limit, initial_weights, risk_aversion ): 
    constraints = create_constraints( carbon_intensity, carbon_limit ) 
    bounds = [ (MIN_WEIGHT, MAX_WEIGHT) for _ in range(len(mu)) ] 
    result = minimize( markowitz_objective, initial_weights, args=( mu, covariance, risk_aversion ), method="SLSQP", bounds=bounds, constraints=constraints, options={ "maxiter": 2000, "ftol": 1e-10 } ) 
    return result

## INDICATEURS DU PORTEFEUILLE

def portfolio_metrics( weights, mu, covariance, carbon_intensity, benchmark_weights ): 
    portfolio_return = ( weights @ mu ) 
    portfolio_variance = ( weights @ covariance @ weights ) 
    portfolio_volatility = np.sqrt( portfolio_variance )
    portfolio_waci = ( weights @ carbon_intensity ) 
    benchmark_waci = ( benchmark_weights @ carbon_intensity ) 
    carbon_reduction = ( 1 - portfolio_waci / benchmark_waci )
    return portfolio_return, portfolio_variance, portfolio_volatility, portfolio_waci, benchmark_waci, carbon_reduction

def carbon_trajectory( initial_waci, years, annual_reduction=0.07 ): 
    trajectory = []
    for year in range(years + 1):
        target = ( initial_waci * (1 - annual_reduction) ** year )
        trajectory.append({ "Year": year, "Maximum WACI": target })
    return pd.DataFrame(trajectory)

def optimize_for_year( year, initial_waci, mu, covariance, carbon_intensity, benchmark_weights ):
    carbon_limit = ( initial_waci * (1 - ANNUAL_CARBON_REDUCTION) ** year )
    initial_weights = benchmark_weights.copy()
    result = optimize_markowitz( mu, covariance, carbon_intensity, carbon_limit, initial_weights, RISK_AVERSION )
    if not result.success: 
        raise RuntimeError( f"Optimisation échouée : " f"{result.message}" ) 
    return result, carbon_limit


