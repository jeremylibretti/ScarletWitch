# https://stackoverflow.com/questions/11310248/find-number-of-columns-in-csv-file

import csv

datafilename = 'model/dynamic_classifier/dynamic_data.csv'
with open(datafilename, 'r') as csv:
     first_line = csv.readline()
     your_data = csv.readlines()

ncol = first_line.count(',') + 1

print(ncol)