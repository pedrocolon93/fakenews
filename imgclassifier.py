import random

import GPy
import json
import pickle

import math
import numpy as np

def processfeatures(article):
    row = []
    if article is None:
        for i in range(0,45):
            row.append(0)
        return row
    if len(article['faces_anger'])==0:
        for i in range(0,10):
            row.append(0)
    else:
        row.append(np.mean(article['faces_anger']))
        row.append(np.mean(article['faces_headwear']))
        row.append(np.mean(article['faces_joy']))
        row.append(np.mean(article['faces_pan']))
        row.append(np.mean(article['faces_propertiesblurred']))
        row.append(np.mean(article['faces_propertiesunderexposed']))
        row.append(np.mean(article['faces_roll']))
        row.append(np.mean(article['faces_sorrow']))
        row.append(np.mean(article['faces_surprise']))
        row.append(np.mean(article['faces_tilt']))
    row.append(len(article['labels_labelslist']))
    row.append(len(article['landmarks_landmarkslist']))
    row.append(len(article['logos_logolist']))
    avgalpha = 0
    avgblue = 0
    avggreen = 0
    avgred = 0
    for colorindex in range (0,5):
        row.append(article["properties_color" + str(colorindex) + "alpha"])
        avgalpha+=row[-1]
        row.append(article["properties_color" + str(colorindex) + "blue"])
        avgblue+=row[-1]
        row.append(article["properties_color" + str(colorindex) + "green"])
        avggreen+=row[-1]
        row.append(article["properties_color" + str(colorindex) + "red"])
        avgred+=row[-1]
    row.append(avgalpha/5)
    row.append(avgblue/5)
    row.append(avggreen/5)
    row.append(avgred/5)
    row.append(article['safesearch_adult'])
    row.append(article['safesearch_medical'])
    row.append(article['safesearch_spoof'])
    row.append(article['safesearch_violence'])
    row.append(len(article['web_full_matching_images']))
    row.append(len(article['web_pages_with_matching_images']))
    row.append(len(article['web_partial_matching_images']))
    row.append(len(article['web_web_entities']))
    # for indx, x in enumerate(row):
    #     if math.isnan(x):
    #         row[indx] = np.array([0])
    #     if type(x) is list:
    #         row[indx] = np.array(x)
    #     else:
    #         row[indx] = np.array([x])
    # print len(row)
    return row


def main():
    print 'Starting feature processing'
    articles = json.load(open('buzzfeed_sandbox/buzzfeed_sandbox.json'))
    datamap = {}
    for article in articles:
        content = articles[article]
        datamap[article] = ['', '']
        if content['labels']['Rating'] == 'mostly true':
            datamap[article][1] = 1
        else:
            datamap[article][1] = 0
    raw_features = pickle.load(open('imagefeatures.p'))
    for articleindex in raw_features:
        article = raw_features[articleindex]
        featurerow = processfeatures(article)
        datamap[articleindex][0]= np.array(featurerow)

    dummy = ''
    for key in datamap:
        dummy = key
        break

    print 'Starting classification'
    keys = datamap.keys()
    random.shuffle(keys)
    train = keys[0:1308]
    test = keys[1308:]
    X = []
    Y = []
    for artindex in train:
        if type(datamap[artindex][0]) is str:
            continue
        X.append(datamap[artindex][0])
        Y.append(datamap[artindex][1])
    for indx, label in enumerate(Y):
        Y[indx] = np.array([label])

    X = np.array(X)
    # print X.ndim
    Y = np.array(Y)
    print 'Featamount',len(X[0])
    k = GPy.kern.RBF(len(X[0]))
    m = GPy.models.GPClassification(X,Y,k)
    m.save('imgclass.model')
    m.optimize()
    preds = m.predict(test)
    match = 0
    nomatch = 0
    for index,item in enumerate(test):
        if item == preds[0][index]:
            match+=1
        else:
            nomatch+=1
    print match,nomatch,match+nomatch,match/(match+nomatch)

if __name__ == '__main__':
    main()