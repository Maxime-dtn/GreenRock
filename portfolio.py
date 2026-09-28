import pandas as pd
import numpy as np
from scipy.optimize import minimize

#%%

## IMPORT DES DONNEES 

def load_data():
    return pd.read_csv("Data/données.csv")

df = load_data()
print(df.head())
# %%

## SCREENING DES DONNES (FILTRES)

CARBON_REDUCTION_INITIAL = 0.50 # WACI <= 50 % du benchmark 
ANNUAL_CARBON_REDUCTION = 0.07 # -7 % par an
RISK_AVERSION = 5.0 
TRADING_DAYS = 252 
MIN_WEIGHT = 0.0 
MAX_WEIGHT = 1.0

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

def optimize_for_year( year, initial_waci, mu, covariance, carbon_intensity, benchmark_weights, method="markowitz" ):
    carbon_limit = ( initial_waci * (1 - ANNUAL_CARBON_REDUCTION) ** year )
    initial_weights = benchmark_weights.copy()
    result = optimize_markowitz( mu, covariance, carbon_intensity, carbon_limit, initial_weights, RISK_AVERSION )
    if not result.success: 
        raise RuntimeError( f"Optimisation échouée : " f"{result.message}" ) 
    return result, carbon_limit


