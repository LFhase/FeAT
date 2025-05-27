
data_dir=##
ckpt_path=##
for seed in 0 1 2;
do
    WANDB_MODE=offline python feat_dfr_run.py --data_dir=${data_dir} \
                    --data_transform=BertTokenizeTransform \
                    --dataset=WildsAmazonCoarse  --model=bert_pretrained \
                    -feat -cheat --seed ${seed}  \
                    --ckpt_path=${ckpt_path}
done;

data_dir=##
ckpt_path=##
for seed in {0..9};
do
    WANDB_MODE=offline python feat_dfr_run.py --data_dir=${data_dir} \
                    --data_transform=AugWaterbirdsCelebATransform \
                    --dataset=WildsCamelyon  --model=imagenet_resnet50_pretrained \
                    -feat -cheat --seed ${seed}  \
                    --ckpt_path=${ckpt_path}
done;


data_dir=##
ckpt_path=##
for seed in 0 1 2;
do
    WANDB_MODE=offline python feat_dfr_run.py --data_dir=${data_dir} \
                    --data_transform=BertTokenizeTransform \
                    --dataset=WildsCivilCommentsCoarse  --model=bert_pretrained \
                    -feat --seed ${seed}  \
                    --ckpt_path=${ckpt_path}
done;

data_dir=##
ckpt_path=##
for seed in 0 1 2;
do
    WANDB_MODE=offline python feat_dfr_run.py --data_dir=${data_dir} \
                    --data_transform=AugWaterbirdsCelebATransform \
                    --dataset=WildsFMOW  --model=imagenet_resnet50_pretrained \
                    -feat --seed ${seed}  \
                    --ckpt_path=${ckpt_path}
done;

data_dir=##
ckpt_path=##
for seed in 0 1 2;
do
    WANDB_MODE=offline python feat_dfr_run.py --data_dir=${data_dir} \
                    --data_transform=AugWaterbirdsCelebATransform \
                    --dataset=WildsIwildCam  --model=imagenet_resnet50_pretrained \
                    -feat --seed ${seed}  \
                    --ckpt_path=${ckpt_path}
done;
