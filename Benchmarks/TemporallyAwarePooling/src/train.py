import logging
import os
import zipfile
import sys
import json
import time
from tqdm import tqdm
import torch
import numpy as np

import sklearn
import sklearn.metrics
from sklearn.metrics import average_precision_score
from SoccerNet.Evaluation.utils import LoadJsonFromZip, AverageMeter, EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2, EVENT_DICTIONARY_V1, INVERSE_EVENT_DICTIONARY_V1
from SoccerNet.Evaluation.ActionSpotting import average_mAP
from dataset import INVERSE_EVENT_DICTIONARY_V3, EVENT_DICTIONARY_V3, getListGames
import glob




def trainer(train_loader,
            val_loader,
            val_metric_loader,
            model,
            optimizer,
            scheduler,
            criterion,
            model_name,
            max_epochs=1000,
            evaluation_frequency=20):

    logging.info("start training")

    best_loss = 9e99

    for epoch in range(max_epochs):
        best_model_path = os.path.join("models", model_name, "model.pth.tar")

        # train for one epoch
        loss_training = train(train_loader, model, criterion,
                              optimizer, epoch + 1, train=True)

        # evaluate on validation set
        loss_validation = train(
            val_loader, model, criterion, optimizer, epoch + 1, train=False)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join("models", model_name), exist_ok=True)

        # remember best prec@1 and save checkpoint
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        # Save the best model based on loss only if the evaluation frequency too long
        if is_better:
            torch.save(state, best_model_path)

        # Test the model on the validation set
        if epoch % evaluation_frequency == 0 and epoch != 0:
            performance_validation = test(
                val_metric_loader,
                model,
                model_name)

            logging.info("Validation performance at epoch " +
                         str(epoch+1) + " -> " + str(performance_validation))

        # Reduce LR on Plateau after patience reached
        prevLR = optimizer.param_groups[0]['lr']
        scheduler.step(loss_validation)
        currLR = optimizer.param_groups[0]['lr']
        if (currLR is not prevLR and scheduler.num_bad_epochs == 0):
            logging.info("Plateau Reached!")

        if (prevLR < 2 * scheduler.eps and
                scheduler.num_bad_epochs >= scheduler.patience):
            logging.info(
                "Plateau Reached and no more reduction -> Exiting Loop")
            break

    return


def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          train=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (feats, labels) in t:
            # measure data loading time
            data_time.update(time.time() - end)
            feats = feats.cuda()
            labels = labels.cuda()
            # compute output
            output = model(feats)

            # hand written NLL criterion
            loss = criterion(labels, output)

            # measure accuracy and record loss
            losses.update(loss.item(), feats.size(0))

            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if train:
                desc = f'Train {epoch}: '
            else:
                desc = f'Evaluate {epoch}: '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            desc += f'Loss {losses.avg:.4e} '
            t.set_description(desc)

    return losses.avg


def test(dataloader, model, model_name):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.eval()

    end = time.time()
    all_labels = []
    all_outputs = []
    with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
        for i, (feats, labels) in t:
            # measure data loading time
            data_time.update(time.time() - end)
            feats = feats.cuda()
            # labels = labels.cuda()

            # print(feats.shape)
            # feats=feats.unsqueeze(0)
            # print(feats.shape)

            # compute output
            output = model(feats)

            all_labels.append(labels.detach().numpy())
            all_outputs.append(output.cpu().detach().numpy())

            batch_time.update(time.time() - end)
            end = time.time()

            desc = f'Test (cls): '
            desc += f'Time {batch_time.avg:.3f}s '
            desc += f'(it:{batch_time.val:.3f}s) '
            desc += f'Data:{data_time.avg:.3f}s '
            desc += f'(it:{data_time.val:.3f}s) '
            t.set_description(desc)

    AP = []
    for i in range(1, dataloader.dataset.num_classes+1):
        AP.append(average_precision_score(np.concatenate(all_labels)
                                          [:, i], np.concatenate(all_outputs)[:, i]))

    # t.set_description()
    # print(AP)
    mAP = np.mean(AP)
    print(mAP, AP)

    return mAP

def testSpotting(dataloader, model, model_name, overwrite=True, NMS_window=30, NMS_threshold=0.5):
    
    split = '_'.join(dataloader.dataset.split)
    # print(split)
    output_results = os.path.join("models", model_name, f"results_spotting_{split}.zip")
    output_folder = f"outputs_{split}"


    if not os.path.exists(output_results) or overwrite:
        batch_time = AverageMeter()
        data_time = AverageMeter()

        spotting_grountruth = list()
        spotting_grountruth_visibility = list()
        spotting_predictions = list()

        model.eval()

        count_visible = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)
        count_unshown = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)
        count_all = torch.FloatTensor([0.0]*dataloader.dataset.num_classes)

        end = time.time()
        with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
            for i, (game_ID, feat_half1, feat_half2, label_half1, label_half2) in t:
                data_time.update(time.time() - end)

                # Batch size of 1
                game_ID = game_ID[0]
                feat_half1 = feat_half1.squeeze(0)
                label_half1 = label_half1.float().squeeze(0)
                feat_half2 = feat_half2.squeeze(0)
                label_half2 = label_half2.float().squeeze(0)

                # Compute the output for batches of frames
                BS = 256
                timestamp_long_half_1 = []
                for b in range(int(np.ceil(len(feat_half1)/BS))):
                    start_frame = BS*b
                    end_frame = BS*(b+1) if BS * \
                        (b+1) < len(feat_half1) else len(feat_half1)
                    feat = feat_half1[start_frame:end_frame].cuda()
                    output = model(feat).cpu().detach().numpy()
                    timestamp_long_half_1.append(output)
                timestamp_long_half_1 = np.concatenate(timestamp_long_half_1)

                timestamp_long_half_2 = []
                for b in range(int(np.ceil(len(feat_half2)/BS))):
                    start_frame = BS*b
                    end_frame = BS*(b+1) if BS * \
                        (b+1) < len(feat_half2) else len(feat_half2)
                    feat = feat_half2[start_frame:end_frame].cuda()
                    output = model(feat).cpu().detach().numpy()
                    timestamp_long_half_2.append(output)
                timestamp_long_half_2 = np.concatenate(timestamp_long_half_2)


                timestamp_long_half_1 = timestamp_long_half_1[:, 1:]
                timestamp_long_half_2 = timestamp_long_half_2[:, 1:]

                spotting_grountruth.append(torch.abs(label_half1))
                spotting_grountruth.append(torch.abs(label_half2))
                spotting_grountruth_visibility.append(label_half1)
                spotting_grountruth_visibility.append(label_half2)
                spotting_predictions.append(timestamp_long_half_1)
                spotting_predictions.append(timestamp_long_half_2)

                batch_time.update(time.time() - end)
                end = time.time()

                desc = f'Test (spot.): '
                desc += f'Time {batch_time.avg:.3f}s '
                desc += f'(it:{batch_time.val:.3f}s) '
                desc += f'Data:{data_time.avg:.3f}s '
                desc += f'(it:{data_time.val:.3f}s) '
                t.set_description(desc)



                def get_spot_from_NMS(Input, window=60, thresh=0.0):

                    detections_tmp = np.copy(Input)
                    indexes = []
                    MaxValues = []
                    while(np.max(detections_tmp) >= thresh):

                        # Get the max remaining index and value
                        max_value = np.max(detections_tmp)
                        max_index = np.argmax(detections_tmp)
                        MaxValues.append(max_value)
                        indexes.append(max_index)
                        # detections_NMS[max_index,i] = max_value

                        nms_from = int(np.maximum(-(window/2)+max_index,0))
                        nms_to = int(np.minimum(max_index+int(window/2), len(detections_tmp)))
                        detections_tmp[nms_from:nms_to] = -1

                    return np.transpose([indexes, MaxValues])

                framerate = dataloader.dataset.framerate
                get_spot = get_spot_from_NMS

                json_data = dict()
                json_data["UrlLocal"] = game_ID
                json_data["predictions"] = list()

                for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
                    for l in range(dataloader.dataset.num_classes):
                        spots = get_spot(
                            timestamp[:, l], window=NMS_window*framerate, thresh=NMS_threshold)
                        for spot in spots:
                            # print("spot", int(spot[0]), spot[1], spot)
                            frame_index = int(spot[0])
                            confidence = spot[1]
                            # confidence = predictions_half_1[frame_index, l]

                            seconds = int((frame_index//framerate)%60)
                            minutes = int((frame_index//framerate)//60)

                            prediction_data = dict()
                            prediction_data["gameTime"] = str(half+1) + " - " + str(minutes) + ":" + str(seconds)
                            if dataloader.dataset.version == 2:
                                prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[l]
                            elif dataloader.dataset.version == 1:
                                prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V1[l]
                            elif dataloader.dataset.version == 3:
                                prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V3[l]

                            prediction_data["position"] = str(int((frame_index/framerate)*1000))
                            prediction_data["half"] = str(half+1)
                            prediction_data["confidence"] = str(confidence)
                            json_data["predictions"].append(prediction_data)
                
                os.makedirs(os.path.join("models", model_name, output_folder, game_ID), exist_ok=True)
                with open(os.path.join("models", model_name, output_folder, game_ID, "results_spotting.json"), 'w') as output_file:
                    json.dump(json_data, output_file, indent=4)


        def zipResults(zip_path, target_dir, filename="results_spotting.json"):            
            zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
            rootlen = len(target_dir) + 1
            for base, dirs, files in os.walk(target_dir):
                for file in files:
                    if file == filename:
                        fn = os.path.join(base, file)
                        zipobj.write(fn, fn[rootlen:])

        # zip folder
        zipResults(zip_path = output_results,
                target_dir = os.path.join("models", model_name, output_folder),
                filename="results_spotting.json")

    if split == "challenge": 
        print("Visit eval.ai to evalaute performances on Challenge set")
        return None

    results =  evaluate(SoccerNet_path=dataloader.dataset.path, 
                 Predictions_path=output_results,
                 split="test",
                 prediction_file="results_spotting.json", 
                 version=dataloader.dataset.version, 
                 framerate=dataloader.dataset.framerate)

    return results


def evaluate(SoccerNet_path, Predictions_path, prediction_file="results_spotting.json", split="test", version=2, framerate=2, metric="tight"):
    # evaluate the prediction with respect to some ground truth
    # Params:
    #   - SoccerNet_path: path for labels (folder or zipped file)
    #   - Predictions_path: path for predictions (folder or zipped file)
    #   - prediction_file: name of the predicted files - if set to None, try to infer it
    #   - split: split to evaluate from ["test", "challenge"]
    #   - frame_rate: frame rate to evalaute from [2]
    # Return:
    #   - details mAP
    list_games = getListGames(split=split)
    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()

    for game in tqdm(list_games):

        # Load labels
        if version==2:
            label_files = "Labels-v2.json"
            num_classes = 17
        if version==1:
            label_files = "Labels.json"
            num_classes = 3
        if version==3:
            label_files = "Labels-v3.json"
            num_classes = 1
        if zipfile.is_zipfile(SoccerNet_path):
            labels = LoadJsonFromZip(SoccerNet_path, os.path.join(game, label_files))
        else:
            labels = json.load(open(os.path.join(SoccerNet_path, game, label_files)))
        # convert labels to vector
        label_half_1, label_half_2 = label2vector(labels, num_classes=num_classes, version=version, framerate=framerate)
        # print(version)
        # print(label_half_1)
        # print(label_half_2)



        # infer name of the prediction_file
        if prediction_file == None:
            if zipfile.is_zipfile(Predictions_path):
                with zipfile.ZipFile(Predictions_path, "r") as z:
                    for filename in z.namelist():
                        #       print(filename)
                        if filename.endswith(".json"):
                            prediction_file = os.path.basename(filename)
                            break
            else:
                for filename in glob.glob(os.path.join(Predictions_path,"*/*/*/*.json")):
                    prediction_file = os.path.basename(filename)
                    # print(prediction_file)
                    break

        # Load predictions
        if zipfile.is_zipfile(Predictions_path):
            predictions = LoadJsonFromZip(Predictions_path, os.path.join(game, prediction_file))
        else:
            predictions = json.load(open(os.path.join(Predictions_path, game, prediction_file)))
        # convert predictions to vector
        predictions_half_1, predictions_half_2 = predictions2vector(predictions, num_classes=num_classes, version=version, framerate=framerate)

        targets_numpy.append(label_half_1)
        targets_numpy.append(label_half_2)
        detections_numpy.append(predictions_half_1)
        detections_numpy.append(predictions_half_2)

        closest_numpy = np.zeros(label_half_1.shape)-1
        #Get the closest action index
        for c in np.arange(label_half_1.shape[-1]):
            indexes = np.where(label_half_1[:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = label_half_1[indexes[i],c]
        closests_numpy.append(closest_numpy)

        closest_numpy = np.zeros(label_half_2.shape)-1
        for c in np.arange(label_half_2.shape[-1]):
            indexes = np.where(label_half_2[:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = label_half_2[indexes[i],c]
        closests_numpy.append(closest_numpy)


    if metric == "loose":
        deltas=np.arange(12)*5 + 5
    elif metric == "tight":
        deltas=np.arange(5)*1 + 1
    # Compute the performances
    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = average_mAP(targets_numpy, detections_numpy, closests_numpy, framerate, deltas=deltas)
    
    results = {
        "a_mAP": a_mAP,
        "a_mAP_per_class": a_mAP_per_class,
        "a_mAP_visible": a_mAP_visible if version==2 else None,
        "a_mAP_per_class_visible": a_mAP_per_class_visible if version==2 else None,
        "a_mAP_unshown": a_mAP_unshown if version==2 else None,
        "a_mAP_per_class_unshown": a_mAP_per_class_unshown if version==2 else None,
    }
    return results




def label2vector(labels, num_classes=17, framerate=2, version=2):


    vector_size = 90*60*framerate

    label_half1 = np.zeros((vector_size, num_classes))
    label_half2 = np.zeros((vector_size, num_classes))

    for annotation in labels["annotations"]:

        time = annotation["gameTime"]
        event = annotation["label"]

        half = int(time[0])

        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        frame = framerate * ( seconds + 60 * minutes ) 

        if version == 2:
            if event not in EVENT_DICTIONARY_V2:
                continue
            label = EVENT_DICTIONARY_V2[event]
        elif version == 1:
            # print(event)
            # label = EVENT_DICTIONARY_V1[event]
            if "card" in event: label = 0
            elif "subs" in event: label = 1
            elif "soccer" in event: label = 2
            else: 
                # print(event)
                continue
        # print(event, label, half)
        elif version == 3:
            if event not in EVENT_DICTIONARY_V3:
                continue
            label = EVENT_DICTIONARY_V3[event]

        value = 1
        if "visibility" in annotation.keys():
            if annotation["visibility"] == "not shown":
                value = -1

        if half == 1:
            frame = min(frame, vector_size-1)
            label_half1[frame][label] = value

        if half == 2:
            frame = min(frame, vector_size-1)
            label_half2[frame][label] = value

    return label_half1, label_half2





def predictions2vector(predictions, num_classes=17, version=2, framerate=2):


    vector_size = 90*60*framerate

    prediction_half1 = np.zeros((vector_size, num_classes))-1
    prediction_half2 = np.zeros((vector_size, num_classes))-1

    for annotation in predictions["predictions"]:

        time = int(annotation["position"])
        event = annotation["label"]

        half = int(annotation["half"])

        frame = int(framerate * ( time/1000 ))

        if version == 2:
            if event not in EVENT_DICTIONARY_V2:
                continue
            label = EVENT_DICTIONARY_V2[event]
        elif version == 1:
            label = EVENT_DICTIONARY_V1[event]
            # print(label)
            # EVENT_DICTIONARY_V1[l]
            # if "card" in event: label=0
            # elif "subs" in event: label=1
            # elif "soccer" in event: label=2
            # else: continue
        elif version == 3:
            label = EVENT_DICTIONARY_V3[event]

        value = annotation["confidence"]

        if half == 1:
            frame = min(frame, vector_size-1)
            prediction_half1[frame][label] = value

        if half == 2:
            frame = min(frame, vector_size-1)
            prediction_half2[frame][label] = value

    return prediction_half1, prediction_half2