""" Convert json files to mat files that record distance between frames of movement """
import glob
import json
import scipy.io as sio
import numpy as np

# Number of features is points detected * 2 because of x and y coordinates
NUM_FEATURES = 36
# Number of frames is the number of frames per movement returned to us
NUM_FRAMES = 80
NUM_CLASSES = 3
NUM_NOSE_INDEX = 1
NUM_NECK_INDEX = 3

def get_train_movement_data(folder_name, label_val):
    # Get all directories in a folder
    subdirectories = glob.glob(folder_name + "/*/")
    # Iterate through each subdirectory (the various examples of falls and turns)
    for directory in subdirectories:
        path = directory + '*.json'
        list_files = sorted(glob.glob(path)) # Sorted in order to preserve time sensitive info
        matrix = np.zeros([NUM_FEATURES, NUM_FRAMES])

        # For each instance of a fall and turn, get 80 frames and save to a mat file
        x = (len(list_files) - NUM_FRAMES) // 2
        prev_keypoints = np.zeros(NUM_FEATURES)
        prev_dist = 1
        for i in range(x, NUM_FRAMES + x):
            index = i - x
            fname = list_files[i]
            with open(fname) as json_file:
                json_data = json.load(json_file)
                curr_keypoints = json_data["people"][0]["pose_keypoints"] # 0 is for first person
                del curr_keypoints[2::3]
                # y distance between nose and neck
                dist = curr_keypoints[NUM_NECK_INDEX] - curr_keypoints[NUM_NOSE_INDEX]
                print(dist)
                if index == 0:
                    matrix[:, index] = prev_keypoints # always start with all 0 for motion vectors
                else:
                    if dist != 0:
                        vector = np.divide(np.subtract(curr_keypoints, prev_keypoints), dist)
                        dist = prev_dist
                    else:
                        vector = np.divide(np.subtract(curr_keypoints, prev_keypoints), prev_dist)
                    matrix[:, index] = vector
                prev_keypoints = curr_keypoints

        mat_fname = 'mat_motion_vector_train/' + directory[10:-1] + '.mat'
        label = np.zeros(NUM_CLASSES)
        label[label_val - 1] = 1
        sio.savemat(mat_fname, mdict = {'keypoints': matrix, 'label': label})

# The way that the data is formatted means that there are no subdirectories for the test data
# As a result, we get the data in a slightly different way for tests
def get_test_movement_data(folder_name, label_val):
    list_files = sorted(glob.glob(folder_name + '/*.json'))
    matrix = np.zeros([NUM_FEATURES, NUM_FRAMES])
    x = (len(list_files) - NUM_FRAMES) // 2
    prev_keypoints = np.zeros(NUM_FEATURES)
    prev_dist = 1
    for i in range(x, NUM_FRAMES + x):
        # Later will want to edit to choose center 80 frames
        fname = list_files[i]
        with open(fname) as json_file:
            index = i - x
            json_data = json.load(json_file)
            # If a person is not detected, we assume motion has not changed i.e. distance change is all zero
            if len(json_data["people"]) == 0:
                matrix[:, index] = np.zeros(NUM_FEATURES)
            else:
                curr_keypoints = json_data["people"][0]["pose_keypoints"] # 0 is for first person
                del curr_keypoints[2::3]
                # y distance between nose and neck
                dist = curr_keypoints[NUM_NECK_INDEX] - curr_keypoints[NUM_NOSE_INDEX]
                if index == 0:
                    matrix[:, index] = prev_keypoints # always start with all 0 for motion vector
                else:
                    if dist != 0:
                        vector = np.divide(np.subtract(curr_keypoints, prev_keypoints), dist)
                        dist = prev_dist
                    else:
                        vector = np.divide(prev_keypoints, prev_dist)
                    matrix[:, index] = vector

                prev_keypoints = curr_keypoints

    mat_fname = 'mat_motion_vector_test/' + folder_name[15:] + '.mat'
    label = np.zeros(NUM_CLASSES)
    label[label_val - 1] = 1
    sio.savemat(mat_fname, mdict = {'keypoints': matrix, 'label': label})


# Label code
# Turn: 1 (turn right) and 4 (turn left) 
# Jump: 2 (jump to stage right) and 5 (jump to stage left)
# Fall: 3 (to stage right) and 6 (fall to stage left)

# Turn: 1
# Jump: 2
# Fall: 3

###### 3 classes

def get_maia_turn():
    get_train_movement_data('maia_turn', 1)

def get_maia_jump():
    get_train_movement_data('maia_jump', 2)

def get_maia_fall():
    get_train_movement_data('maia_fall', 3)

def get_sliu_turn():
    get_train_movement_data('sliu_turn', 1)

def get_sliu_jump():
    get_train_movement_data('sliu_jump', 2)

def get_sliu_fall():
    get_train_movement_data('sliu_fall', 3)

###### 6 classes

# def get_maia_turn():
#     get_train_movement_data('maia_turn', 1)

# def get_maia_jump():
#     get_train_movement_data('maia_jump', 2)

# def get_maia_fall():
#     get_train_movement_data('maia_fall', 3)

# def get_sliu_turn():
#     get_train_movement_data('sliu_turn', 4)

# def get_sliu_jump():
#     get_train_movement_data('sliu_jump', 5)

# def get_sliu_fall():
#     get_train_movement_data('sliu_fall', 6)

################### TESTING DATA ####################


###### 3 classes

# def test_10():
#     get_test_movement_data('test_keypoints/test10_both_turn', 1)

# def test_11():
#     get_test_movement_data('test_keypoints/test11_both_jump', 2)

def test_12():
    get_test_movement_data('test_keypoints/test12_maia_up', 3)

def test_13():
    get_test_movement_data('test_keypoints/test13_maia_occ', 2)

def test_14():
    get_test_movement_data('test_keypoints/test14_maia_jump_close', 2)

def test_15():
    get_test_movement_data('test_keypoints/test15_maia_fall_close', 3)

def test_16():
    get_test_movement_data('test_keypoints/test16_maia_turn_close', 1)

def test_17():
    get_test_movement_data('test_keypoints/test17_maia_fall_180', 3)

def test_18():
    get_test_movement_data('test_keypoints/test18_maia_fall_90', 3)

def test_19():
    get_test_movement_data('test_keypoints/test19_maia_jump_180', 2)
# Test number 10
def test_1():
    get_test_movement_data('test_keypoints/test1_maia_turn', 1)
# 11
def test_20():
    get_test_movement_data('test_keypoints/test20_maia_jump_90', 2)
# 12
def test_21():
    get_test_movement_data('test_keypoints/test21_maia_turn_180', 1)
# 13
def test_22():
    get_test_movement_data('test_keypoints/test22_maia_turn_45', 1)
# 14
def test_23():
    get_test_movement_data('test_keypoints/test23_maia_turn_90', 1)
# 15
def test_24():
    get_test_movement_data('test_keypoints/test24_maia_turn_out', 1)

def test_2():
    get_test_movement_data('test_keypoints/test2_maia_jump', 2)

def test_3():
    get_test_movement_data('test_keypoints/test3_maia_jump', 2)

def test_4():
    get_test_movement_data('test_keypoints/test4_maia_turn', 1)

def test_5():
    get_test_movement_data('test_keypoints/test5_maia_fall', 3)

def test_6():
    get_test_movement_data('test_keypoints/test6_maia_jump', 2)

def test_7():
    get_test_movement_data('test_keypoints/test7_maia_fall', 3)

def test_8():
    get_test_movement_data('test_keypoints/test8_maia_double', 1)

# def test_9():
#     get_test_movement_data('test_keypoints/test9_both_turn', 1)


##### 6 classes 

# def test_1():
#     get_test_movement_data('test_keypoints/test1_maia_turn', 1)

# def test_2():
#     get_test_movement_data('test_keypoints/test2_maia_jump', 5)

# def test_3():
#     get_test_movement_data('test_keypoints/test3_maia_jump', 5)

# def test_4():
#     get_test_movement_data('test_keypoints/test4_maia_turn', 1)

# def test_5():
#     get_test_movement_data('test_keypoints/test5_maia_fall', 3)

# def test_6():
#     get_test_movement_data('test_keypoints/test6_maia_jump', 2)

# def test_7():
#     get_test_movement_data('test_keypoints/test7_maia_fall', 6)

# def test_8():
#     get_test_movement_data('test_keypoints/test8_maia_double', 1)

# def test_9():
#     get_test_movement_data('test_keypoints/test9_both_turn', 1)

# def test_10():
#     get_test_movement_data('test_keypoints/test10_both_turn', 1)

# def test_11():
#     get_test_movement_data('test_keypoints/test11_both_jump', 2)

# def test_12():
#     get_test_movement_data('test_keypoints/test12_maia_up', 6)

# def test_13():
#     get_test_movement_data('test_keypoints/test13_maia_occ', 2)

# def test_14():
#     get_test_movement_data('test_keypoints/test14_maia_jump_close', 2)

# def test_15():
#     get_test_movement_data('test_keypoints/test15_maia_fall_close', 3)

# def test_16():
#     get_test_movement_data('test_keypoints/test16_maia_turn_close', 1)

def main():
    # 6 sets of training videos, 3 categories
    get_maia_jump()
    get_maia_turn()
    get_maia_fall()
    get_sliu_turn()
    get_sliu_fall()
    get_sliu_jump()

    ###### TESTING DATA ########
    test_1()
    test_2()
    test_3()
    test_4()
    test_5()
    test_6()
    test_7()
    test_8()
    # These tests are with two people--data is not clean
    # test_9()
    # test_10()
    # test_11()
    test_12()
    test_13()
    test_14()
    test_15()
    test_16()
    test_17()
    test_18()
    test_19()
    test_20()
    test_21()
    test_22()
    test_23()
    test_24()

if __name__ == "__main__":
    main()
