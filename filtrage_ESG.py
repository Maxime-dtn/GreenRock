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

## ... 



def load_gcel():
    return pd.read_excel("Data/GCEL.xlsx")

gcel = load_gcel()


def normalize_company_name(name):
    ...
    