##################################################################
#
# Functionalize
# Arch DL Starsystem, 5 Data-environments, bidirectional internode links, monodirectional intranode links, low-dimensional data (sampled by stratification)
# Computable implementation of sub-minimal scale
# Extent of inter-/ or intralink-complexity only due NN Specification.
# For considering effectivity of system over node-counts and stratified data, fix NN-link to dense
##################################################################

 
################################################################################
# INIATIALIZE MINIMAL INTERLINK-MERGE OVER FULL COMBINATORIC SPECTRUM
################################################################################

# USUAL STARLINKS
# D1 -> D5
# D2 -> D5
# D3 -> D5
# D4 -> D5

# COMBINATORIC INTEGRATION
# D1, D2 -> D5
# D2, D3 -> D5
# D2, D4 -> D5
# D4, D3 -> D5
#... (12 TOTAL COMBINATIONS)
# D1, D2, D4 -> D5
# D3, D2, D4 -> D5
# D1, D2, D4 -> D5
# D1, D2, D3 -> D5
# D3, D1, D4 -> D5
#... (4 TOTAL COMBINATIONS)
# D1, D2, D3, D4 -> D5

# DEVELOP VARIANTS OF COMBINATORIC INTEGRATION
# VARIANT 1 (OUTPUT: LINK BETWEEN LINKS <=> LINK BETWEEN LINK AND LINKS) 
#           (INTERESTING; D1 - D5 ARE ENVIRONMENTS THAT ARE WORLD'S AWAY FROM ONE ANOTHER)
# CHECK CODE FOR BUGS IN COMPUTATION; D1 -> D5 - D4 -> D5 PRECISION VARIATION EXPECTED
# TWEAK LINK-PARAMETERS

##################################################################
# NODE-ENVIRONMENT PROCEDURE OF DOWNLOAD (License: Keras Developer)
##################################################################

##################################################################
# THE BELOW DICTATES CONTINUOUS DATA-INJECTIVITY AND COPUTABILITY IN DL-STARSYSTEMS. HERE
# FOLLOWS ONE IMPLEMENTATION OF 

# An AI for everyone, by everyone, the usefulness of which becomes clear 
# in everyday wonder at connections in anything. It's not about where the connections 
# lie, but the connection itself and the extensions of knowledge that can be derived 
# from it. So that all forms of enlightenment can be extracted from the things we 
# observe, and that enlightenment is stored in an external entity that will be as 
# close to objectivity as possible, at least as true as functionally necessary. 
# It will serve as an Encyclopedia for Humanity, so that people can turn to it 
# for guidance in knowledge, science, law, and, ultimately, life orientation. 
# So that someone in Japan knows how many fish live in a local river based on 
# the good or bad experience of a fellow human being in a territory flagged to 
# Canada. So that learning truly happens, not just in our heads but in an external 
# entity, most accessible through and derived from reality, whatever that may be. 
# So that major conflicts and wars can be avoided as there is a right answer, 
# so that we know what to aim for, so that we know how to navigate the universe
# in the desired event that interstellar travel becomes something grand.

# WE BUT NEED TO INTEGRATE ONLINE ACTIVITY. HOW DOES ONE PROCEED THEREIN?
# IDEA: INTEGRATE DL-STARSYSTEM BACKEND TO A WWW INTERFACE FOR MASSIVE ONLINE DATA-INJECTION.
##################################################################

from zipfile import ZipFile

import keras
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
import math
from helpfile import Ytractorfunctional
from zipfile import ZipFile

global raw_data

def UniversalDeepLearner(DESIRED_COMPUTATION_COUNT):
    
    uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
    zip_file = ZipFile(zip_path)
    zip_file.extractall()
    csv_path = "jena_climate_2009_2016.csv"
    fname = os.path.join(csv_path)
    with open(fname) as f:
        data = f.read()

    lines = data.split("\n")
    header = lines[0].split(",")
    lines = lines[1:]
    
    raw_data2 = Ytractorfunctional(None)
    raw_data4 = np.zeros(4, len(header) - 1) # TWEAK raw_data4 HOWEVER MUCH YOU DESIRE
    x = input("Enter raw_data4 np.zeros(4, len(header) - 1):")
    raw_data2 = raw_data2 + raw_data4

    while counter < DESIRED_COMPUTATION_COUNT:
        raw_data2 = Ytractorfunctional(raw_data2)
        raw_data4 = np.zeros(4, len(header) - 1) # TWEAK raw_data4 HOWEVER MUCH YOU DESIRE
        # prompt for raw_data4
        x = input("Enter raw_data4 np.zeros(4, len(header) - 1):")
        raw_data2 = raw_data2 + raw_data4
        # NOTICE THE BOOTSTRAP
        counter = counter + 1
    
    return raw_data2

# THIS DEFINES CONTINUOUS DATA-INJECTIVITY, WITHOUT LOSS OF DATA DURING INJECTION OR DURING COMPUTATION.
# WE NEED OUTPUT FROM Ytractorfunction.py TO TOO 
# SERVE AS INPUT TO YTRACTORFUNCTIONAL.PY
# HOW TO PROCEED?