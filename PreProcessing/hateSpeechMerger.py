from PreProcessing import jsonFunctions as js
import csv



def creatDict(path):
    newDict = {}
    reader = csv.DictReader(open(path))

    for raw in reader:
        newDict[raw["tweet_id"]] = raw["label"]
    return newDict

def merge(dOne, dTwo):
    newTwoDict = {}
    newName = 'Cleaned_new_Two.csv'
    with open(newName, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['text', 'label'])
        for key in dOne:
            if(dTwo[key] != None):
                newTwoDict[dOne[key]] = dTwo[key]
                writer.writerow([dOne[key], dTwo[key]])






def main(one, two):
    dTwo = creatDict(two)

    dOne = js.getJSON(one)

    merge( dOne, dTwo)





if __name__ == '__main__':
        main("PreProcessing/new_hate_Two.json", "PreProcessing/hatespeech_labels.csv")
