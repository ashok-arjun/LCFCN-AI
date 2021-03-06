from haven import haven_utils as hu

EXP_GROUPS = {}

EXP_GROUPS['shanghai'] =  {"dataset": {'name':'shanghai', 
                            'transform':'rgb_normalize'},
         "model": {'name':'lcfcn','base':"fcn8_resnet"},
         "batch_size": [1],
         "max_epoch": [100],
         'dataset_size': {'train':'all', 'val':'all'},
         'optimizer':['adam'],
         'lr':[1e-5]
         }
EXP_GROUPS = {k: hu.cartesian_exp_group(v) for k, v in EXP_GROUPS.items()}