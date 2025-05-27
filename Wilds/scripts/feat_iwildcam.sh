data_dir=##
exp_dir=##

pi=48000
pr=2
rp=0.5

# first train a featurizer with FeAT based on an abitary OOD algorithm
for seed in {0..2};do
python feat_run.py  --data-dir ${data_dir}  --exp-dir ${exp_dir} --dataset iwilcam --num_workers 2  -ifat -pi ${pi} --seed ${seed} -pr ${pr} -rp ${rp} --frozen --algorithm erm
done;

# second train the classifier with the featurizer frozen
ood_alg=irm
penalty=0.01
for seed in {0..2};do
python feat_run.py  --data-dir ${data_dir}  --exp-dir ${exp_dir} --dataset iwilcam --num_workers 2  -ifat -pi ${pi} --seed ${seed} -pr ${pr} -rp ${rp} --frozen \
                    --algorithm ${ood_alg} --use_old --epochs 9 -p ${penalty}
done;
