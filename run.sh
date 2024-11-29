# MVTecAD 'capsule cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
# MVTecAD 'capsule hazelnut transistor cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
# VISA  'candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum'
# MVTecLoco 'breakfast_box  juice_bottle  pushpins  screw_bag  splicing_connectors'
# MPDD 'tubes metal_plate connector bracket_white bracket_brown bracket_black'
gpu_id=$1
method_setting='LANGCAD'
dataset='visa'
mb_size='196'


for mb in $mb_size
do
  for d in $dataset
  do
    CUDA_VISIBLE_DEVICES=0 python main.py \
      default_setting=./configs/default/$d.yaml \
      model_setting=./configs/model/$method_setting.yaml \
      DEFAULT.exp_name=1223-only_contrastive-memorybank_$mb
  done
done