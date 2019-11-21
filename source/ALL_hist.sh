for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

python main.py --gen_choice='PS_SRResNet' --name='BSDS100' --test_phase='test' --train=False --test=True --scale=4 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

python main.py --gen_choice='PS_SRResNet' --name='CLIC_mobile' --test_phase='train' --train=False --test=True --scale=4 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

python main.py --gen_choice='PS_SRResNet' --name='CLIC_mobile' --test_phase='valid' --train=False --test=True --scale=4 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

python main.py --gen_choice='PS_SRResNet' --name='CLIC_professional' --test_phase='train' --train=False --test=True --scale=4 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

python main.py --gen_choice='PS_SRResNet' --name='CLIC_professional' --test_phase='valid' --train=False --test=True --scale=4 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

python main.py --gen_choice='PS_SRResNet' --name='DIV2K' --test_phase='train' --train=False --test=True --scale=4 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

python main.py --gen_choice='PS_SRResNet' --name='DIV2K' --test_phase='valid' --train=False --test=True --scale=4 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

python main.py --gen_choice='PS_SRResNet' --name='Flickr2K' --test_phase='train' --train=False --test=True --scale=4 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

python main.py --gen_choice='PS_SRResNet' --name='Manga109' --test_phase='test' --train=False --test=True --scale=4 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

python main.py --gen_choice='PS_SRResNet' --name='PIRM' --test_phase='valid' --train=False --test=True --scale=4 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

python main.py --gen_choice='PS_SRResNet' --name='PIRM' --test_phase='test' --train=False --test=True --scale=4 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

python main.py --gen_choice='PS_SRResNet' --name='Set5' --test_phase='test' --train=False --test=True --scale=4 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

python main.py --gen_choice='PS_SRResNet' --name='Set14' --test_phase='test' --train=False --test=True --scale=4 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done

python main.py --gen_choice='PS_SRResNet' --name='Urban100' --test_phase='test' --train=False --test=True --scale=4 --min_LR=0 --max_LR=1 --min_HR=-1 --max_HR=1 --dis_choice='trivial_dis' --con_choice='trivial_con' --B=16 --patch_size=128

for i in $(nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9); do kill -9 $i; done