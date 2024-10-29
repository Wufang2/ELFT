import torch, json

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

#%% md

# 0. Initialize and Load Pre-trained Models

#%%

model_config_path = " "  # change the path of the model config file
model_checkpoint_path = " " # change the path of the model checkpoint

#%%

args = SLConfig.fromfile(model_config_path)
args.device = 'cuda'
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()

#%%

# load coco names
with open('util/coco_id2name.json') as f:
    id2name = json.load(f)
    id2name = {int(k):v for k,v in id2name.items()}

# %% md

# 1. Visualize images from a dataloader
## 1.1 Load Datasets

#%%

args.dataset_file = 'coco'
args.coco_path = "./COCO2017/" # the path of coco
args.fix_size = False

dataset_val = build_dataset(image_set='val', args=args)
print(args.coco_path)
#%% md

## 1.2 Get an Example and Visualize it

#%%

image, targets = dataset_val[184]

#%%

# build gt_dict for vis
box_label = [id2name[int(item)] for item in targets['labels']]
gt_dict = {
    'boxes': targets['boxes'],
    'image_id': targets['image_id'],
    'size': targets['size'],
    'box_label': box_label,
}
vslzr = COCOVisualizer()
vslzr.visualize(image, gt_dict, savedir=None)

#%% md

## 1.3 Visualize Model Predictions

#%%

output = model.cuda()(image[None].cuda())
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

#%%

thershold = 0.3 # set a thershold

scores = output['scores']
labels = output['labels']
boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
select_mask = scores > thershold

#%%

box_label = [id2name[int(item)] for item in labels[select_mask]]
pred_dict = {
    'boxes': boxes[select_mask],
    'size': targets['size'],
    'box_label': box_label
}
vslzr.visualize(image, pred_dict, savedir=None)
