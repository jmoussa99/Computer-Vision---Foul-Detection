import logging
import os
import time
import torch
import gc
import csv
from config.classes import INVERSE_EVENT_DICTIONARY
import json
try:
    from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
except ModuleNotFoundError:
    evaluate = None
from tqdm import tqdm


def _require_evaluate():
    if evaluate is None:
        raise ModuleNotFoundError("SoccerNet is required for VARS evaluation. Install it with `pip install SoccerNet`.")
    return evaluate


def _append_loss_row(loss_path, epoch, split, loss_action, loss_offence_severity):
    file_exists = os.path.exists(loss_path)
    total_loss = loss_action + loss_offence_severity
    with open(loss_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["epoch", "split", "loss_action", "loss_offence_severity", "loss_total"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "epoch": epoch,
            "split": split,
            "loss_action": loss_action,
            "loss_offence_severity": loss_offence_severity,
            "loss_total": total_loss,
        })
    logging.info(
        "%s epoch %s loss_action=%.6f loss_offence_severity=%.6f loss_total=%.6f",
        split.upper(),
        epoch,
        loss_action,
        loss_offence_severity,
        total_loss,
    )

def trainer(train_loader,
            val_loader2,
            test_loader2,
            model,
            optimizer,
            scheduler,
            criterion,
            best_model_path,
            epoch_start,
            model_name,
            path_dataset,
            max_epochs=1000
            ):
    

    logging.info("start training")
    counter = 0
    loss_path = os.path.join(best_model_path, "losses.csv")

    for epoch in range(epoch_start, max_epochs):
        
        print(f"Epoch {epoch+1}/{max_epochs}")
    
        # Create a progress bar
        pbar = tqdm(total=len(train_loader), desc="Training", position=0, leave=True)

        ###################### TRAINING ###################
        prediction_file, loss_action, loss_offence_severity = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=True,
            set_name="train",
            pbar=pbar,
        )
        _append_loss_row(loss_path, epoch + 1, "train", loss_action, loss_offence_severity)

        results = _require_evaluate()(os.path.join(path_dataset, "Train", "annotations.json"), prediction_file)
        print("TRAINING")
        print(results)

        ###################### VALIDATION ###################
        prediction_file, loss_action, loss_offence_severity = train(
            val_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train = False,
            set_name="valid"
        )
        _append_loss_row(loss_path, epoch + 1, "valid", loss_action, loss_offence_severity)

        results = _require_evaluate()(os.path.join(path_dataset, "Valid", "annotations.json"), prediction_file)
        print("VALIDATION")
        print(results)


        ###################### TEST ###################
        prediction_file, loss_action, loss_offence_severity = train(
                test_loader2,
                model,
                criterion,
                optimizer,
                epoch + 1,
                model_name,
                train=False,
                set_name="test",
            )
        _append_loss_row(loss_path, epoch + 1, "test", loss_action, loss_offence_severity)

        results = _require_evaluate()(os.path.join(path_dataset, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)
        

        scheduler.step()

        counter += 1

        if counter > 3:
            state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
            }
            path_aux = os.path.join(best_model_path, str(epoch+1) + "_model.pth.tar")
            torch.save(state, path_aux)
        
    pbar.close()    
    return

def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          model_name,
          train=False,
          set_name="train",
          pbar=None,
        ):
    

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    loss_total_action = 0
    loss_total_offence_severity = 0
    total_loss = 0

    if not os.path.isdir(model_name):
        os.mkdir(model_name) 

    # path where we will save the results
    prediction_file = "predicitions_" + set_name + "_epoch_" + str(epoch) + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}

    if True:
        for batch in dataloader:

            targets_offence_severity, targets_action, mvclips, cv_features, action = _unpack_batch(batch)

            targets_offence_severity = targets_offence_severity.cuda()
            targets_action = targets_action.cuda()
            mvclips = mvclips.cuda().float()
            if cv_features is not None:
                cv_features = cv_features.cuda().float()
            
            if pbar is not None:
                pbar.update()

            # compute output
            outputs_offence_severity, outputs_action, _ = model(mvclips, cv_features)
            
            if len(action) == 1:
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), dim=-1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), dim=-1)

                values = {}
                values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                if preds_sev.item() == 0:
                    values["Offence"] = "No offence"
                    values["Severity"] = ""
                elif preds_sev.item() == 1:
                    values["Offence"] = "Offence"
                    values["Severity"] = "1.0"
                elif preds_sev.item() == 2:
                    values["Offence"] = "Offence"
                    values["Severity"] = "3.0"
                elif preds_sev.item() == 3:
                    values["Offence"] = "Offence"
                    values["Severity"] = "5.0"
                actions[action[0]] = values       
            else:
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

                for i in range(len(action)):
                    values = {}
                    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                    if preds_sev[i].item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev[i].item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev[i].item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev[i].item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[action[i]] = values       

            
            if len(outputs_offence_severity.size()) == 1:
                outputs_offence_severity = outputs_offence_severity.unsqueeze(0)   
            if len(outputs_action.size()) == 1:
                outputs_action = outputs_action.unsqueeze(0)  
   
            #compute the loss
            loss_offence_severity = criterion[0](outputs_offence_severity, targets_offence_severity)
            loss_action = criterion[1](outputs_action, targets_action)

            loss = loss_offence_severity + loss_action

            if train:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_total_action += float(loss_action)
            loss_total_offence_severity += float(loss_offence_severity)
            total_loss += 1
          
        gc.collect()
        torch.cuda.empty_cache()
    
    data["Actions"] = actions
    with open(os.path.join(model_name, prediction_file), "w") as outfile: 
        json.dump(data, outfile)  
    return os.path.join(model_name, prediction_file), loss_total_action / total_loss, loss_total_offence_severity / total_loss




# Evaluation function to evaluate the test or the chall set
def evaluation(dataloader,
          model,
          set_name="test",
        ):
    

    model.eval()

    prediction_file = "predicitions_" + set_name + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}
           
    if True:
        for batch in dataloader:

            _, _, mvclips, cv_features, action = _unpack_batch(batch)
            mvclips = mvclips.cuda().float()
            if cv_features is not None:
                cv_features = cv_features.cuda().float()
            #mvclips = mvclips.float()
            outputs_offence_severity, outputs_action, _ = model(mvclips, cv_features)

            if len(action) == 1:
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), dim=-1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), dim=-1)

                values = {}
                values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                if preds_sev.item() == 0:
                    values["Offence"] = "No offence"
                    values["Severity"] = ""
                elif preds_sev.item() == 1:
                    values["Offence"] = "Offence"
                    values["Severity"] = "1.0"
                elif preds_sev.item() == 2:
                    values["Offence"] = "Offence"
                    values["Severity"] = "3.0"
                elif preds_sev.item() == 3:
                    values["Offence"] = "Offence"
                    values["Severity"] = "5.0"
                actions[action[0]] = values       
            else:
                preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
                preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

                for i in range(len(action)):
                    values = {}
                    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                    if preds_sev[i].item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev[i].item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev[i].item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev[i].item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[action[i]] = values                    


        gc.collect()
        torch.cuda.empty_cache()
    
    data["Actions"] = actions
    with open(prediction_file, "w") as outfile: 
        json.dump(data, outfile)  
    return prediction_file


def _unpack_batch(batch):
    if len(batch) == 5:
        targets_offence_severity, targets_action, mvclips, cv_features, action = batch
        return targets_offence_severity, targets_action, mvclips, cv_features, action
    targets_offence_severity, targets_action, mvclips, action = batch
    return targets_offence_severity, targets_action, mvclips, None, action
