""" Convert json files for every movement into a mat file"""
import glob
import json
import scipy.io as sio
import numpy as np

# Number of features is points detected * 2 because of x and y coordinates
NUM_FEATURES = 18 * 2
# Number of frames is the number of frames per movement returned to us
NUM_FRAMES = 80
NUM_CLASSES = 6

def get_movement_data(folder_name, label_val):
    # Get all directories in a folder
    subdirectories = glob.glob(folder_name + "/*/")
    # Iterate through each subdirectory (the various examples of falls and turns)
    # print(subdirectories)
    for directory in subdirectories:
        path = directory + '*.json'
        list_files = sorted(glob.glob(path)) # Sorted in order to preserve time sensitive info
        matrix = np.zeros([NUM_FEATURES, NUM_FRAMES])

        # For each instance of a fall and turn, get 80 frames and save to a mat file
        for i in range(NUM_FRAMES):
            # Later will want to edit to choose center 80 frames
            fname = list_files[i]
            with open(fname) as json_file:
                json_data = json.load(json_file)
                keypoints = json_data["people"][0]["pose_keypoints"] # 0 is for first person
                del keypoints[2::3] # Check how to make this robust later
                matrix[:, i] = keypoints
        
        mat_fname = 'mat/' + directory[10:-1] + '.mat'
        label = np.zeros(NUM_CLASSES)
        label[label_val - 1] = 1
        sio.savemat(mat_fname, mdict = {'keypoints': matrix, 'label': label})

# Label code
# Turn: 1 and 4
# Jump: 2 and 5
# Fall: 3 and 6
def get_maia_turn():
    get_movement_data('maia_turn', 1)

def get_maia_jump():
    get_movement_data('maia_jump', 2)

def get_maia_fall():
    get_movement_data('maia_fall', 3)

def get_sliu_turn():
    get_movement_data('sliu_turn', 4)

def get_sliu_jump():
    get_movement_data('sliu_jump', 5)

def get_sliu_fall():
    get_movement_data('sliu_fall', 6)

def main():
    get_maia_jump()
    get_maia_turn()
    get_maia_fall()
    get_sliu_turn()
    get_sliu_fall()
    get_sliu_jump()

if __name__ == "__main__":
    main()
