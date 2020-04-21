import csv

def cleanup(name):
    reader = csv.DictReader(open(name))
    newName = 'Cleaned_'+name
    with open(newName, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['id', 'created_at', 'text'])
        for raw in reader:
            tmp = raw['text'].encode().decode()



            writer.writerow([raw['id'], raw['created_at'], tmp])
            print(tmp)


if __name__ == '__main__':
    # pass in the username of the account you want to download
    cleanup("JustinTrudeau_tweets.csv")