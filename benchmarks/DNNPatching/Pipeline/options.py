import argparse

parser = argparse.ArgumentParser(description='Patching')
parser.add_argument('--task', type=str, required=False, default='oa')
#TODO: redundant below arg --remove
parser.add_argument('--validate', type =bool, default = True)
#TODO: change status input to control input 
parser.add_argument('--status_inputs_count', type=int, default=4)
parser.add_argument('--train_patch', type =bool, default = True)
parser.add_argument('--train_base', type =bool, default = False)
#TODO: Fix main output test
parser.add_argument('--test_main_output', type=bool, default=False)
parser.add_argument('--epochs', type =int, default=200)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--threshold', type=float, default=0.02) #0.1
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--resume_from', type =str, required=False)
#TODO: redundant below arg --remove
parser.add_argument('--save_training_signals', type =bool, default=False)
# Pruning args
parser.add_argument('--pruning_percent', type =float, default=0.50)
parser.add_argument('--finetune_prunednet', type=bool, default=True)
parser.add_argument('--finetune_epochs', type =int, default=50)
args = parser.parse_args()
