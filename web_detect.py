#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demonstrates web detection using the Google Cloud Vision API.

Example usage:
  python web_detect.py https://goo.gl/X4qcB6
  python web_detect.py ../detect/resources/landmark.jpg
  python web_detect.py gs://your-bucket/image.png
"""
# [START full_tutorial]
# [START imports]
import json

import numpy as np
import io

import time
from google.cloud import vision
from google.cloud.vision.feature import Feature
from google.cloud.vision.feature import FeatureTypes
from google.cloud.vision.image import Image

# [END imports]

#CONSTANTS
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

project_name = 'fakenewsll-169502'
limit = 10
vision_client = None
    # [END get_annotations]


def report(annotations):
    """Prints detected features in the provided web annotations."""
    # [START print_annotations]
    if annotations.pages_with_matching_images:
        print('\n{} Pages with matching images retrieved')

        for page in annotations.pages_with_matching_images:
            print('Score : {}'.format(page.score))
            print('Url   : {}'.format(page.url))

    if annotations.full_matching_images:
        print ('\n{} Full Matches found: '.format(
               len(annotations.full_matching_images)))

        for image in annotations.full_matching_images:
            print('Score:  {}'.format(image.score))
            print('Url  : {}'.format(image.url))

    if annotations.partial_matching_images:
        print ('\n{} Partial Matches found: '.format(
               len(annotations.partial_matching_images)))

        for image in annotations.partial_matching_images:
            print('Score: {}'.format(image.score))
            print('Url  : {}'.format(image.url))

    if annotations.web_entities:
        print ('\n{} Web entities found: '.format(
            len(annotations.web_entities)))

        for entity in annotations.web_entities:
            print('Score      : {}'.format(entity.score))
            print('Description: {}'.format(entity.description))
    # [END print_annotations]


def convert_likelihood_to_int(likelihood):
    from google.cloud.vision.likelihood import Likelihood
    if likelihood == Likelihood.VERY_UNLIKELY:
        return 1
    elif likelihood == Likelihood.UNLIKELY:
        return 2
    elif likelihood == Likelihood.POSSIBLE:
        return 3
    elif likelihood == Likelihood.LIKELY:
        return 4
    elif likelihood == Likelihood.VERY_LIKELY:
        return 5
    else:
        return 0
    pass


def extract_features(annotations):
    featurevector = []
    feature_dictionary = {}
    # Web feats
    feature_dictionary["web_full_matching_images"] = [fmi.url for fmi in annotations.web.full_matching_images]
    feature_dictionary["web_pages_with_matching_images"] = [ pmi.url for pmi in annotations.web.pages_with_matching_images]
    feature_dictionary["web_partial_matching_images"] = [x.url for x in annotations.web.partial_matching_images]
    feature_dictionary["web_web_entities"] = [x.description for x in annotations.web.web_entities]

    # Safesearch
    feature_dictionary["safesearch_adult"] = convert_likelihood_to_int(annotations.safe_searches.adult)
    feature_dictionary["safesearch_medical"] = convert_likelihood_to_int(annotations.safe_searches.medical)
    feature_dictionary["safesearch_spoof"] = convert_likelihood_to_int(annotations.safe_searches.spoof)
    feature_dictionary["safesearch_violence"] = convert_likelihood_to_int(annotations.safe_searches.violence)

    # properties
    for colorindex in range(0,5):
        color = annotations.properties.colors[colorindex].color
        feature_dictionary["properties_color" + str(colorindex) + "alpha"] = color.alpha
        feature_dictionary["properties_color" + str(colorindex) + "blue"] = color.blue
        feature_dictionary["properties_color" + str(colorindex) + "green"] = color.green
        feature_dictionary["properties_color" + str(colorindex) + "red"] = color.red

    # logos
    feature_dictionary["logos_logolist"] = [x.description for x in annotations.logos]
    # landmark
    feature_dictionary["landmarks_landmarkslist"] = [x.description for x in annotations.landmarks]
    # labels
    feature_dictionary["labels_labelslist"] = [x.description for x in annotations.labels]
    # faces
    feature_dictionary["faces_anger"] = [convert_likelihood_to_int(x.anger) for x in annotations.faces]
    feature_dictionary["faces_joy"] = [convert_likelihood_to_int(x.joy) for x in annotations.faces]
    feature_dictionary["faces_sorrow"] = [convert_likelihood_to_int(x.sorrow) for x in annotations.faces]
    feature_dictionary["faces_surprise"] = [convert_likelihood_to_int(x.surprise) for x in annotations.faces]
    feature_dictionary["faces_pan"] = [x.angles.pan for x in annotations.faces]
    feature_dictionary["faces_roll"] = [x.angles.roll for x in annotations.faces]
    feature_dictionary["faces_tilt"] = [x.angles.tilt for x in annotations.faces]
    feature_dictionary["faces_headwear"] = [x.angles.tilt for x in annotations.faces]
    feature_dictionary["faces_propertiesblurred"] = [x.image_properties.blurred for x in annotations.faces]
    feature_dictionary["faces_propertiesunderexposed"] = [x.image_properties.underexposed for x in annotations.faces]

    return feature_dictionary



if __name__ == '__main__':
    # [START run_web]
    #Initialize web environment


    # Get a list of images/paths to images

    articles = json.load(open('buzzfeed_sandbox/buzzfeed_sandbox.json'))
    path_to_articles = "buzzfeed_sandbox"
    images = []
    for article in articles:
        try:
            image = articles[article]['image']['location']
            images.append((article, path_to_articles + '/' + image))
        except:
            images.append((article, None))

    # Feature list
    imgfeats = {}

    # Iterate through images and get the annotations and the features
    print 'Starting feature extraction'
    count = 0

    client = vision.Client()
    vision_client = vision.Client().batch()
    features = []

    features.append(Feature(feature_type=FeatureTypes.WEB_DETECTION, max_results=100000))
    features.append(Feature(feature_type=FeatureTypes.FACE_DETECTION, max_results=100000))
    features.append(Feature(feature_type=FeatureTypes.LANDMARK_DETECTION, max_results=limit))
    features.append(Feature(feature_type=FeatureTypes.IMAGE_PROPERTIES, max_results=limit))
    features.append(Feature(feature_type=FeatureTypes.TEXT_DETECTION, max_results=limit))
    features.append(Feature(feature_type=FeatureTypes.LABEL_DETECTION, max_results=limit))
    features.append(Feature(feature_type=FeatureTypes.SAFE_SEARCH_DETECTION, max_results=limit))
    features.append(Feature(feature_type=FeatureTypes.LOGO_DETECTION, max_results=limit))
    # cnt = 0
    for image in images:

        path = image[1]
        if path is None:
            continue
        # if cnt == 10:
        #     break
        # else:
        #     cnt+=1
        if 'http' in path or path.startswith('gs:'):
            image = Image(source_uri=path,client=client)
            vision_client.add_image(image=image,features=features)
        else:
            with io.open(path, 'rb') as image_file:
                content = image_file.read()

            image = Image(content=content,client=client)
            vision_client.add_image(image=image,features=features)


    results = vision_client.detect()
    for image in images:
        if image is None:
            imgfeats[image[0]] = None
            continue
        print 'Extracting for',count+1
        annotations = results[count]
        feature_dict = extract_features(annotations)
        imgfeats[image[0]] = feature_dict

        count+=1
    import pickle
    pickle.dump(imgfeats, open("imagefeatures.p", "wb"))
    # json.dump(imgfeats,open("imagefeatures.json",'wb'))
    # Classify whether true or not
    # imgfeats = imgfeats.transpose()
    # X = imgfeats[0:1]
    # Y = [1, 0, 0]
    # classif = OneVsRestClassifier(SVC(kernel='linear'))
    # classif.fit(X, Y[0:1])
    # print classif.predict(imgfeats[2])


    # [END run_web]
# [END full_tutorial]
