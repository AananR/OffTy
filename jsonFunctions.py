import json

#The following function was taken from: Weaver, K (2019) save-file.py. https://gist.github.com/keithweaver/ae3c96086d1c439a49896094b5a59ed0
def writeToJSONFile(path, fileName, data):
    filePathNameWExt = path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)



#The following function was taken from: Weaver, K (2019) get-json.py. https://gist.github.com/keithweaver/ae3c96086d1c439a49896094b5a59ed0
def getJSON(filePathAndName):
    with open(filePathAndName, 'r') as fp:
        return json.load(fp)