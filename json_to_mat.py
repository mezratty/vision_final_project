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

        # Folder name for tests is formatted slightly differently, hence the if-else statement
        mat_fname = 'mat/' + directory[10:-1] + '.mat'
        label = np.zeros(NUM_CLASSES)
        label[label_val - 1] = 1
        sio.savemat(mat_fname, mdict = {'keypoints': matrix, 'label': label})

# The way that the data is formatted means that there are no subdirectories for the test data
# As a result, we get the data in a slightly different way
def get_test_movement_data(folder_name, label_val):
    mat_fname = 'mat_test/' + directory[15:-1] + '.mat'


# Label code
# Turn: 1 (turn right) and 4 (turn left)
# Jump: 2 (jump to stage right) and 5 (jump to stage left)
# Fall: 3 (to stage right) and 6 (fall to stage left)
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

################### TESTING DATA ####################
def test_1():
    get_movement_data('test_keypoints/test1_maia_turn', 1)

def test_2():
    get_movement_data('test_keypoints/test2_maia_jump', 5)

def test_3():
    get_movement_data('test_keypoints/test3_maia_jump', 5)

def test_4():
    get_movement_data('test_keypoints/test4_maia_turn', 1)

def test_5():
    get_movement_data('test_keypoints/test5_maia_fall', 3)

def test_6():
    get_movement_data('test_keypoints/test6_maia_jump', 2)

def test_7():
    get_movement_data('test_keypoints/test7_maia_fall', 5)

def test_8():
    get_movement_data('test_keypoints/test8_maia_double', 1)

def test_9():
    get_movement_data('test_keypoints/test9_both_turn', 1)

def test_10():
    get_movement_data('test_keypoints/test10_both_turn', 1)

def test_11():
    get_movement_data('test_keypoints/test11_both_jump', 2)

def test_12():
    get_movement_data('test_keypoints/test12_maia_up', 6)

def test_13():
    get_movement_data('test_keypoints/test13_maia_occ', 2)

def test_14():
    get_movement_data('test_keypoints/test14_maia_jump_close', 2)

def test_15():
    get_movement_data('test_keypoints/test15_maia_fall_close', 3)

def test_16():
    get_movement_data('test_keypoints/test16_maia_turn_close', 1)

def main():
    # get_maia_jump()
    # get_maia_turn()
    # get_maia_fall()
    # get_sliu_turn()
    # get_sliu_fall()
    # get_sliu_jump()

    ###### TESTING DATA ########
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_6()
    test_7()
    test_8()
    test_9()
    test_10()
    test_11()

if __name__ == "__main__":
    main()
