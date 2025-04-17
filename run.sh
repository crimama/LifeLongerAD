# MVTecAD 'capsule cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
# MVTecAD 'capsule hazelnut transistor cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
# VISA  'candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum'
# MVTecLoco 'breakfast_box  juice_bottle  pushpins  screw_bag  splicing_connectors'
# MPDD 'tubes metal_plate connector bracket_white bracket_brown bracket_black'
exp_id=$1

# GPU ID에 따라 method_setting 설정
if [ "$exp_id" -eq 141 ]; then
    method_setting="iuf"
    continual='true'
    dataset='mvtecad_141'
    exp='14_1_with_1_step'
    gpu_id=0
elif [ "$exp_id" -eq 35 ]; then
    method_setting="cfgcad"
    continual='true'
    gpu_id=0
    exp='3_5_with_5_step'
    dataset='mvtecad_35'
elif [ "$exp_id" -eq 1015 ]; then
    method_setting="cfgcad"
    continual='true'
    exp='10_1_with_5_step'
    dataset='mvtecad_1015'
    gpu_id=0
elif [ "$exp_id" -eq 1051 ]; then
    method_setting="cfgcad"
    continual='true'
    exp='10_5_with_1_step'
    dataset='mvtecad_1051'
    gpu_id=0
else
    echo "Invalid EXP number"
    exit 1
fi




for c in $continual
do
    if [ "$c" = "true" ]; then
        # continual_method="EMPTY EWC"
        continual_method="EMPTY"
        online_options="false"
    else
        continual_method="EMPTY"
        online_options="false"
    fi

    for cm in $continual_method
    do 
        for m in $method_setting
        do 
            for o in $online_options
            do 
                for d in $dataset
                do
                    CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
                        default_setting=./configs/default/$d.yaml \
                        model_setting=./configs/model/$m.yaml \
                        DEFAULT.exp_name=temp-$exp \
                        CONTINUAL.continual=$c \
                        CONTINUAL.online=$o \
                        CONTINUAL.method.name=$cm \
                        TRAIN.epochs=1 \
                        TRAIN.wandb.use=false
                done
            done 
        done
    done
done
