scale=2.0
lr=0.3
base_lr=1.0
vox_loss_weight=1.0
grad_loss_weight=0.005
sdf_loss_weight=1
sphere_radius=0.01
level=14
cuda=2
for j in {0..0}
do
    python gridpull.py \
	--data_dir demo/ \
	--out_dir output/ \
	--class_idx 02691156 \
	--train \
	--lr $lr \
	--base_lr $base_lr \
	--scale $scale \
	--vox_loss_weight $vox_loss_weight \
	--grad_loss_weight $grad_loss_weight \
	--dataset other \
	--sphere_radius $sphere_radius \
	--obj_ind $j \
	--level $level \
	--CUDA $cuda
    for i in {1..1}
    do
        python gridpull.py \
        --data_dir demo/ \
	    --out_dir output/ \
        --class_idx 02691156 \
        --dataset other \
	    --lr $lr \
	    --base_lr $base_lr \
	    --scale $scale \
	    --vox_loss_weight $vox_loss_weight \
	    --grad_loss_weight $grad_loss_weight \
        --index $i \
	    --obj_ind $j \
	    --sphere_radius $sphere_radius \
	    --level $level \
            --CUDA $cuda
    done
done
