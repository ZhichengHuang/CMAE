import argparse
import torch
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    args = parser.parse_args()
    return args

def process_function(in_file):
    data = torch.load(in_file)
    out_dict = OrderedDict()
    weight = data['state_dict']
    for key,value in weight.items():
        new_key = key.replace("norm","ln")
        # new_key = new_key.replace("proj","projection")
        new_key = new_key.replace("mlp.fc1","ffn.layers.0.0" )
        new_key = new_key.replace("mlp.fc2","ffn.layers.1")
        new_key = new_key.replace("blocks","layers")
        if "patch_embed" in new_key:
            new_key = new_key.replace("proj.","projection.")
        new_key = new_key
        if new_key.startswith('encoder_s.'):
            new_key = new_key.split('encoder_s.')[-1]
        out_dict[new_key]=value


    data['state_dict']=out_dict
    # del data['model']
    data['author']='CMAE'
    torch.save(data,"convert_new_model.pth")


def main():
    args = parse_args()
    process_function(args.in_file)

if __name__ == '__main__':
    main()