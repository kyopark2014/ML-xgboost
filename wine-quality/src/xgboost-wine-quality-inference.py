import numpy as np
import pandas as pd
import time

from xgboost import XGBRegressor

def load_samples():
    data = pd.DataFrame(pd.read_json('../data/samples.json'))

    print('samples:', data)

    return data

def main():
    start = time.time()

    # load model
    
    model = XGBRegressor()

    model_name = "../output/xgboost_wine_quality.json"    
    model.load_model(model_name)
    
    # load samples
    data = load_samples()
    
    # inference
    results = model.predict(data)

    print(results)
    print('Elapsed time: %0.2fs' % (time.time()-start))   

if __name__ == '__main__':
    main()


