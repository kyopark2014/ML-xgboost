import numpy as np
import pandas as pd
import time
from xgboost import XGBRegressor

# load model    
model = XGBRegressor()
model_name = "../output/xgboost_wine_quality.json"    
model.load_model(model_name)
    
def handler(data): 
    start = time.time()

    # inference
    results = model.predict(data)
    print('result:', results)

    return {
        'statusCode': 200,
        'body': results.tolist()
    }

def load_samples():
    data = pd.DataFrame(pd.read_json('../data/samples.json'))
    print('samples:', data)
    return data

def main():
    start = time.time()
    
    # load samples
    event = load_samples()
    
    # Inference
    results = handler(event)  

    # results
    print(results['statusCode'])
    print(results['body'])

    print('Elapsed time: %0.2fs' % (time.time()-start))   

if __name__ == '__main__':
    main()



