#Train VAE
python train_vae.py --task=walker2d --ds=random --gpu-no=0
python train_vae.py --task=halfcheetah --ds=random --gpu-no=0
python train_vae.py --task=hopper --ds=random --gpu-no=0

python train_vae.py --task=walker2d --ds=medium --gpu-no=0
python train_vae.py --task=halfcheetah --ds=medium --gpu-no=0
python train_vae.py --task=hopper --ds=medium --gpu-no=0

python train_vae.py --task=walker2d --ds=medium-replay --gpu-no=0
python train_vae.py --task=halfcheetah --ds=medium-replay --gpu-no=0
python train_vae.py --task=hopper --ds=medium-replay --gpu-no=0

python train_vae.py --task=walker2d --ds=medium-expert --gpu-no=0
python train_vae.py --task=halfcheetah --ds=medium-expert --gpu-no=0
python train_vae.py --task=hopper --ds=medium-expert --gpu-no=0

python train_vae.py --task=walker2d --ds=expert --gpu-no=0
python train_vae.py --task=halfcheetah --ds=expert --gpu-no=0
python train_vae.py --task=hopper --ds=expert --gpu-no=0

python train_vae.py --task=maze2d --ds=umaze-dense --gpu-no=0
python train_vae.py --task=maze2d --ds=medium-dense --gpu-no=0
python train_vae.py --task=maze2d --ds=large-dense --gpu-no=0


python train_vae.py --task=maze2d --ds=umaze --gpu-no=0
python train_vae.py --task=maze2d --ds=medium --gpu-no=0
python train_vae.py --task=maze2d --ds=large --gpu-no=0