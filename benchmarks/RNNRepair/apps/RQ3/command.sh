# >>>> 1. generate flip data and corresponding models;
CUDA_VISIBLE_DEVICES=0
no_flip_mode=0
flip_mode=2
flipfirst=1
flipsecond=7
epoch=30
ratio=0.3
start_seed=1000
end_seed=1010
save_id=0

cd ../../

echo "this will run almost 15-25 mins"

python apps/RQ3/dataclean_bin.py -path "./save/dataflip_"$flipfirst"_"$flipsecond"_"$ratio -flip $flip_mode -flipfirst $flipfirst -flipsecond $flipsecond -epoch $epoch -ratio $ratio -start_seed $start_seed -end_seed $end_seed

# >>>> 2. train sgd models;
python apps/RQ3/sgd_train.py -start_seed $start_seed -end_seed $end_seed -flip $flip_mode -ratio $ratio -first $flipfirst -second $flipsecond

# >>>> 3. calculate influence scores with sgd and icml methods.
### python apps/RQ3/sgd_infl.py -target sst_gru_sgd -infl_type sgd  -target_epoch $epoch  -start_seed $start_seed  -end_seed $end_seed -id $save_id -flip $flip_mode -ratio $ratio -first $flipfirst -second $flipsecond
### python apps/RQ3/sgd_infl.py -target sst_gru_sgd -infl_type icml  -target_epoch $epoch  -start_seed $start_seed  -end_seed $end_seed -id $save_id -flip $flip_mode -ratio $ratio -first $flipfirst -second $flipsecond
python apps/RQ3/sgd_infl.py -target mnist_rnn_sgd -infl_type sgd  -target_epoch $epoch  -start_seed $start_seed  -end_seed $end_seed -id $save_id -flip $flip_mode -ratio $ratio -first $flipfirst -second $flipsecond
python apps/RQ3/sgd_infl.py -target mnist_rnn_sgd -infl_type icml  -target_epoch $epoch  -start_seed $start_seed  -end_seed $end_seed -id $save_id -flip $flip_mode -ratio $ratio -first $flipfirst -second $flipsecond

# >>>> 4. retrain
python apps/RQ3/retrain.py -epoch $epoch -flipfirst $flipfirst -flipsecond $flipsecond -flip $flip_mode -ratio $ratio -start_seed $start_seed -end_seed $end_seed

# >>>> 5. plot results
python apps/RQ3/plot_results.py -flipfirst $flipfirst -flipsecond $flipsecond -flip $flip_mode -ratio $ratio -epoch $epoch -start_seed $start_seed -end_seed $end_seed
