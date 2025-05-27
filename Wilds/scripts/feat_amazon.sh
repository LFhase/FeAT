data_dir=##
exp_dir=##

pi=31000
pr=3
rp=2.0

# first train a featurizer with FeAT based on an abitary OOD algorithm
for seed in {0..2};do
python feat_run.py  --data-dir ${data_dir}  --exp-dir ${exp_dir} --dataset amazon --num_workers 2  -ifat -pi ${pi} --seed ${seed} -pr ${pr} -rp ${rp} --frozen --algorithm erm
done;

# second train the classifier with the featurizer frozen
ood_alg=irm
penalty=0.01
for seed in {0..2};do
python feat_run.py  --data-dir ${data_dir}  --exp-dir ${exp_dir} --dataset amazon --num_workers 2  -ifat -pi ${pi} --seed ${seed} -pr ${pr} -rp ${rp} --frozen \
                    --algorithm ${ood_alg} --use_old --epochs 200 -p ${penalty}
done;
