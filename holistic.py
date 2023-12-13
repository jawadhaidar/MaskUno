class Holistic:
    def __init__(self):
        self.selected_boxes=[]
        self.selected_scores=[]
        self.selected_classes=[]

    def first_stage(self):
        '''
        description:
        ensures that the uno predictions can all be found in the
        detctionall else remove extrac boxes

        setps:
        -calculate IOU for each uno box with each detection box
        -remove uno box with maxIOU<thr
        
        Arguments:
            -prediction_boxes uno_boxes
        Return:
            -filtered uno boxes 

        '''
        pass
    def second_stage(self,boxes,scores,classes):
        '''
        description:
            opt1:choose the higest score for repeated predictions due to different unos
            opt2:apply non max sup

        steps:
            -from the pool of boxes choose one box
            -calculate iou of this box with remaining boxes in pool
            -group boxes having iou >thr and remove them from pool
            -From each group choose the one with highest score
            -repeat until pool does not have boxes with IOU>thr

        Arguments:
            -filtered boxes
            -scores
            -classes
        return:
            -selected boxes
        '''
        while(len(boxes)>1):
            #remove box
            box=boxes.pop()
            score=scores.pop()
            class_box=classes.pop()
            #calculate IOU with the rest
            IOU_list=IOU(box,boxes)
            #get ids
            candidate_boxes_remove=IOU_list>=0.95
            candidate_boxes_keep=IOU_list<0.95
            #case there exist ious>0.95
            if True in candidate_boxes_remove:
                #choose pool
                boxes_test=boxes[candidate_boxes_remove].append(box)
                scores_test=scores[candidate_boxes_remove].append(score)
                classes_test=classes[candidate_boxes_remove].append(class_box)
                #check the highest score
                max_index,max_value=max(scores_test)
                #add to selected list
                self.selected_boxes.append(boxes_test[max_index])
                self.selected_scores.append(scores_test[max_index])
                self.selected_classes.append(classes_test[max_index])
            else:#no IOUS>0.95 
                #take only the box
                self.selected_boxes.append(box)
                self.selected_scores.append(score)
                self.selected_classes.append(class_box)
            #filter out tested boxes
            boxes=boxes[candidate_boxes_keep]
            scores=scores[candidate_boxes_keep]
            classes=classes[candidate_boxes_keep]
        #add the last one
        self.selected_boxes.append(boxes[0])
        self.selected_scores.append(scores[0])
        self.selected_classes.append(classes[0])

                   


        pass

    def IOU(box,boxes):
        pass