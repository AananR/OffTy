import pandas as pd
from PreProcessing import jsonFunctions as js
import json

def main():

    jsonData=json.loads("new.json")

    df = pd.DataFrame(jsonData)
    df.to_csv(r'new.csv', index=None)


main()