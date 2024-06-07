import re

def charFilter(myString):
    return re.sub('[^A-Z]+', '', myString, 0, re.I)