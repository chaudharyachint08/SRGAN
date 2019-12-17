for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

# python main.py --seed=42 --name='DIV2K' --train=True --test=True --scale=4 --train_strategy='cnn' --optimizer1='Adam' --lr=0.0001 --outer_epochs=10 --inner_epochs=5 --disk_batch=5 --memory_batch=32 --disk_batches_limit=50 --valid_images_limit=10 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --gen_loss='MAE' --gen_choice='SRResNet' --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --epoch_lr_reduction=True --epoch_lr_red_factor=0.33333333333333 --epoch_lr_red_epochs=500 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

# python main.py --seed=42 --name='DIV2K' --train=True --test=True --scale=4 --train_strategy='cnn' --optimizer1='Adam' --lr=0.0001 --outer_epochs=10 --inner_epochs=5 --disk_batch=5 --memory_batch=32 --disk_batches_limit=50 --valid_images_limit=10 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --gen_loss='MAE' --gen_choice='EDSR' --dis_choice='trivial_dis' --con_choice='trivial_con' --B=32 --epoch_lr_reduction=True --epoch_lr_red_factor=0.33333333333333 --epoch_lr_red_epochs=500 --patch_size=128 --residual_scale=0.1

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

# python main.py --seed=42 --name='DIV2K' --train=True --test=True --scale=4 --train_strategy='cnn' --optimizer1='Adam' --lr=0.0001 --outer_epochs=10 --inner_epochs=5 --disk_batch=5 --memory_batch=32 --disk_batches_limit=50 --valid_images_limit=10 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --gen_loss='MAE' --gen_choice='PS_SRResNet' --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --epoch_lr_reduction=True --epoch_lr_red_factor=0.33333333333333 --epoch_lr_red_epochs=500 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

# python main.py --seed=42 --name='DIV2K' --train=True --test=True --scale=4 --train_strategy='cnn' --optimizer1='Adam' --lr=0.0001 --outer_epochs=10 --inner_epochs=5 --disk_batch=5 --memory_batch=32 --disk_batches_limit=50 --valid_images_limit=10 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --gen_loss='MAE' --gen_choice='SRResNet2' --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --epoch_lr_reduction=True --epoch_lr_red_factor=0.33333333333333 --epoch_lr_red_epochs=500 --patch_size=128 --residual_scale=0.1


for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

# python main.py --seed=42 --name='DIV2K' --train=True --test=True --scale=4 --train_strategy='cnn' --optimizer1='Adam' --lr=0.01 --outer_epochs=5 --inner_epochs=5 --disk_batch=5 --memory_batch=8 --disk_batches_limit=50 --valid_images_limit=1 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --gen_loss='MAE' --gen_choice='DRRN' --dis_choice='trivial_dis' --con_choice='trivial_con' --B=4 --U=4 --epoch_lr_reduction=True --epoch_lr_red_factor=0.33333333333333 --epoch_lr_red_epochs=500 --patch_size=128 --input_interpolation='BICUBIC' --gclip=True --gnclip=0.01

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

# python main.py --seed=42 --name='DIV2K' --train=True --test=False --scale=4 --train_strategy='cnn' --optimizer1='Adam' --lr=0.0001 --outer_epochs=5 --inner_epochs=3 --disk_batch=5 --memory_batch=32 --disk_batches_limit=50 --valid_images_limit=1 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --gen_loss='MAE' --gen_choice='OAM' --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --epoch_lr_reduction=True --epoch_lr_red_factor=0.33333333333333 --epoch_lr_red_epochs=50 --patch_size=96 --input_interpolation='BICUBIC'
