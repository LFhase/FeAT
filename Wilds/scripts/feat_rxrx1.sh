data_dir=##
exp_dir=##

pi=20000
pr=10
rp=10

# first train a featurizer with FeAT based on an abitary OOD algorithm
for seed in {0..2};do
python feat_run.py  --data-dir ${data_dir}  --exp-dir ${exp_dir} --dataset rxrx1 --num_workers 2  -ifat -pi ${pi} --seed ${seed} -pr ${pr} -rp ${rp} --frozen --algorithm erm
done;

# second train the classifier with the featurizer frozen
ood_alg=irm
penalty=0.01
for seed in {0..2};do
python feat_run.py  --data-dir ${data_dir}  --exp-dir ${exp_dir} --dataset rxrx1 --num_workers 2  -ifat -pi ${pi} --seed ${seed} -pr ${pr} -rp ${rp} --frozen \
                    --algorithm ${ood_alg} --use_old --epochs 90 -p ${penalty}
done;
