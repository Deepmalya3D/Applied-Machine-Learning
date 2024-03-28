import pandas as pd

def score(text:str, model, threshold:float=0.5) -> (bool, float): # type: ignore
    propensity = model.predict_proba(pd.Series(text))[:, 1]
    if propensity >= threshold:
        prediction = 1
    else:
        prediction = 0
    
    return prediction, propensity