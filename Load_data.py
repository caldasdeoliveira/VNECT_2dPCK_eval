import pandas as pd
import numpy as np
import re
import os

def get_image_groups_by_dist(gt):
    image_names = gt.index.values.tolist()
    dict = {k: [] for k in range(3,19)}
    for filename in image_names:
        aux = int(re.search('%s(.*)%s' % ('_', 'm'), filename).group(1))
        dict[aux].append(filename)
    return dict

def evaluate_pck( gt, estimation_results, threshold_type = "bbox"):
    '''
    inputs: gt - loaded groundtruth data
            estimation_results - path to file with all results or to directory
                                 which constains such files
            estimation_skeleton_format - can be "coco" or "mpii"
            threshold_type - can be "bbox", for 50% of bounding box height, or
                            "h", for 50% of head segment length
    '''
    ## mine:theirs
    correspondence = {1:16, 6:5, 7:2, 8:6, 9:3, 10:7, 11:4, 12:11, 13:8, 14:12,\
                      15:9,16:13,17:10,18:14,19:15,20:1,21:0}
    keypoints_order= { 1: "nose", 6: "left_shoulder", 7: "right_shoulder",
                 8: "left_elbow", 9: "right_elbow", 10: "left_wrist",
                 11: "right_wrist", 12: "left_hip", 13: "right_hip",
                 14: "left_knee", 15: "right_knee", 16: "left_ankle",
                 17: "right_ankle", 18: "pelvis", 19: "thorax",
                 20: "upper neck", 21: "head top"}

    # TODO:get all images in dataset
    images_by_dist = get_image_groups_by_dist(gt)

    score_by_dist = dict.fromkeys(list(images_by_dist.keys()))

    for dist, images in images_by_dist.items():
        total_keypoints = 0
        detected_keypoints = 0

        for img in images:
            #get keypoints
            gt_keypoints = dict(zip(list(correspondence.keys()),
                                list(gt.loc[ img ,
                                list(correspondence.keys())])))
            est_keypoints = dict(zip(
                            list(correspondence.keys()),
                            list(estimation_results.loc[ img ,
                            list(correspondence.values())])))

            # TODO: calculate target height (divide by 2 here)
            if threshold_type == "bbox":
                l = dict(zip([int(i) for i in gt.columns],list(gt.loc[ img , :])))
                #top = max([i[1] for i in l.values() if i != None])
                top = max([i[1] for i in l.values()])
                #bottom = min([i[1] for i in l.values() if i != None])
                bottom = min([i[1] for i in l.values()])
                #print("top: " + str(top) + " bottom: " + str(bottom))
                thres = (top-bottom)/5
            elif threshold_type == "h":
                l = list(gt.loc[img,[20,21]])
                thres_x = l[0][0]-l[1][0]
                thres_y = l[0][1]-l[1][1]
                thres = (thres_x**2+thres_y**2)/2
            else:
                print("ERROR: threshold_type not recognized")

            for k in keypoints_order.keys():
                if gt_keypoints[k] is not None:
                    total_keypoints += 1
                    if est_keypoints[k] is not None:
                        delta_x = gt_keypoints[k][0] - est_keypoints[k][0]
                        delta_y = gt_keypoints[k][1] - est_keypoints[k][1]
                        est_dist = np.sqrt(delta_x**2 + delta_y**2)

                        if est_dist <= thres:
                            detected_keypoints += 1
        if total_keypoints != 0:
            score = detected_keypoints/total_keypoints
        else:
            score = None
        score_by_dist[dist]=score
    return score_by_dist



def load_gt(path_to_json):

    gt = pd.read_json(path_to_json)
    gt = gt.transpose()
    gt = gt.drop(['size'], axis=1)

    aux = pd.DataFrame(pd.DataFrame(gt['file_attributes'].tolist())['discrete pose'].tolist())
    aux.reset_index(drop=True, inplace=True)
    gt.reset_index(drop=True, inplace=True)
    gt = pd.concat([gt.drop(['file_attributes'],axis=1),aux],axis=1)
    gt = gt.set_index(['filename'])

    temp = pd.DataFrame(columns=['filename']+list(range(1,22)))
    for i in gt['regions'].iteritems():
        temp_dict = dict.fromkeys(['filename'] + list(range(1,22)))
        temp_dict['filename'] = i[0]
        for n,j in enumerate(i[1]):
            try:
                a=j['region_attributes']['Visibility']['Occluded']==True
            except:
                a=False
            temp_dict[int(j['region_attributes']['Keypoints'])]=(j['shape_attributes']['cx'],j['shape_attributes']['cy'],a)
        temp=temp.append(temp_dict , ignore_index=True)

    temp = temp.set_index(['filename'])

    #t = gt.drop(['regions'], axis=1)
    #gt_final = pd.concat([t, temp], axis=1)

    return temp
