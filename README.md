args/base_arg_parser.py --- base args for everything  
args/train_arg_parser.py --- args for training only  
dataset.py --- dataset class  
environmental.yml --- conda env dump  
models.py --- model classes  
train.py --- training script  
train.sh --- bash script for training (to remember args for experiments)  
utils.py --- helper functions  


# Directories to include later on  
data_exploration/ --- put all scripts for data exploration here  
bash_scripts/ --- if more train.sh appear, put them in a folder (add DATES to run names)  
ckpts/ --- saved checkpoint directory, ideally based on name of run (with DATES on them)  
viz/ --- maybe for dumping visualizations at a top level dir (make sure to add to .gitignore so you don't push pngs)   
tensorboard/ --- tensorboard outputs (one dir allows seeing multiple experiments at once)   
eval/ --- evaluation scripts  
data_preprocessing/ --- data preprocessing and cleaning scripts   
data_postprocessing/ --- data postprocessing scripts to e.g. show doctors in clinical experiment  


utils/ --- if utils.py gets unwieldy, start to modularize it   
datasets/ --- if dataset.py gets unwieldy, start to modularize it   
models/ --- if models.py gets unwieldy, start to modularize it    
optim/ --- if optimizer decisions in utils.py gets unwieldy, start to modularize it   

