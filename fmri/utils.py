import os
import numpy as np
import torch
import pydicom
import nibabel as nib

def load_nii_to_array(nii_path):
    return nib.load(nii_path).get_data()

def min_max_scale(x):
    return (x - x.min()) / (x.max() - x.min())

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)
        
def save_res(res, path):
    ensure_dir(path)
    with open(path, "w") as f:
        f.write(str(res))
        
def load_res(path):
    with open(path) as f:
        res = f.read()
    return eval(res)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    state - dict containing:
    "model" : model.state_dict(),
    "optimizer" : optimizer.state_dict(),
    (optionally) loss, epoch, etc.
    """
    ensure_dir(filename)
    torch.save(state, filename)
    
def load_checkpoint(filename):
    """
    """
    state = torch.load(filename)
    return state
    
# def load_checkpoint(filename):
#     """
#     """
# #     model = TheModelClass(*args, **kwargs)
# #     optimizer = TheOptimizerClass(*args, **kwargs)

#     checkpoint = torch.load(filename)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     opt.load_state_dict(checkpoint['optimizer_state_dict'])
# #     epoch = checkpoint['epoch']
# #     loss = checkpoint['loss']
    
    
def load_results(problem_name, problem):
    train_loss_l = load_res("models/" + problem_name + "/train_loss_" + problem.replace("/", "_"))
    val_loss_l = load_res("models/" + problem_name + "/val_loss_" + problem.replace("/", "_"))
    train_metric_l = load_res("models/" + problem_name + "/train_auc_" + problem.replace("/", "_"))
    val_metric_l = load_res("models/" + problem_name + "/val_auc_" + problem.replace("/", "_"))
    val_last_preds_l = load_res("models/" + problem_name + "/val_last_probs_" + problem.replace("/", "_"))
    return train_loss_l, val_loss_l, train_metric_l, val_metric_l, val_last_preds_l
    
def save_results(problem_name, problem, train_loss_l, val_loss_l, train_metric_l, val_metric_l, val_last_preds_l):
    save_res(train_loss_l, "models/" + problem_name + "/" + "train_loss_" + problem.replace("/", "_"))
    save_res(val_loss_l, "models/" + problem_name + "/" + "val_loss_" + problem.replace("/", "_"))
    save_res(train_metric_l, "models/" + problem_name + "/" + "train_auc_" + problem.replace("/", "_"))
    save_res(val_metric_l, "models/" + problem_name + "/" + "val_auc_" + problem.replace("/", "_"))
    save_res(val_last_preds_l, "models/" + problem_name + "/" + "val_last_probs_" + problem.replace("/", "_"))
    print("saved.")