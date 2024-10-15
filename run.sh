
# MVTecAD 'capsule cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
# MVTecAD 'capsule hazelnut transistor cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
# VISA  'candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum'
# MVTecLoco 'breakfast_box  juice_bottle  pushpins  screw_bag  splicing_connectors'
# MPDD 'tubes metal_plate connector bracket_white bracket_brown bracket_black'
gpu_id=$1
method_setting='LANGCAD'

if [ $gpu_id == '0' ]; then #visa 
  default_setting=./configs/default/mvtecad.yaml

elif [ $gpu_id == '1' ]; then
  default_setting=./configs/default/visa.yaml

else
  echo "Invalid GPU ID. Please provide a valid GPU ID (0 or 1)."
fi

for m in $method_setting
do
      echo "method_setting: $m"
      CUDA_VISIBLE_DEVICES=0 python main.py \
      default_setting=$default_setting \
      model_setting=./configs/model/$m.yaml
done 
