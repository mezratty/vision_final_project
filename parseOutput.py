import os
import json
from pprint import pprint
import scipy.io as io

keypoints = {}
 
path = '.'
allFiles = os.listdir(path)
i = 0
for file in allFiles:
	if file == '.DS_Store' or file == 'parseOutput.py' or file == 'maiaT1.mat':
		continue
	i = i + 1
	with open(file, "r") as f:
  		data = json.load(f)
  		keypoints[i] = data["people"][0]["pose_keypoints"]
	
	# print "current file is: " + file
	# data = json.load(open(file))

print (keypoints)

io.savemat('maiaT1.mat', mdict = keypoints)

