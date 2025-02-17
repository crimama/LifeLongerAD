# MVTecAD 'capsule cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
# MVTecAD 'capsule hazelnut transistor cable bottle carpet grid leather metal_nut pill screw tile toothbrush wood zipper'
# VISA  'candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum'
# MVTecLoco 'breakfast_box  juice_bottle  pushpins  screw_bag  splicing_connectors'
# MPDD 'tubes metal_plate connector bracket_white bracket_brown bracket_black'
gpu_id=$1

# GPU ID에 따라 method_setting 설정
if [ "$gpu_id" -eq 0 ]; then
    method_setting="cae"
elif [ "$gpu_id" -eq 1 ]; then
    method_setting="cae"
    gpu_id=0
else
    echo "Invalid GPU ID. Please use 0 or 1."
    exit 1
fi

dataset='mvtecad'
continual='true false'

for c in $continual
do
    if [ "$c" = "true" ]; then
        # continual_method="EMPTY EWC"
        continual_method="EMPTY"
        online_options="false"
    else
        continual_method="EMPTY"
        online_options="true false"
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
                        DEFAULT.exp_name=baseline_$cm \
                        CONTINUAL.continual=$c \
                        CONTINUAL.online=$o \
                        CONTINUAL.method.name=$cm
                done
            done 
        done
    done
done
