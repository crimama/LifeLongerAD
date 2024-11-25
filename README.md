# LANGCAD : Language guided multi prompt continual anomaly detection 

# Record 
 - 24.10.15 push

# Environments

NVIDIA pytorch docker [ [link](https://github.com/ufoym/deepo) ]

```bash
docker pull ufoym/deepo:pytorch
```

## Requirements
[requirements.sh](requirements.sh)

```bash
bash requirements.sh
```
# Run Patchcore
```bash
gpu_id=$1

if [ $gpu_id == '0' ]; then
  class_name='capsule'
  anomaly_ratio='0.0'
  sampling_method='identity'
elif [ $gpu_id == '1' ]; then
  class_name='capsule'
  anomaly_ratio='0.1'
  sampling_method='identity'  
else
  echo "Invalid GPU ID. Please provide a valid GPU ID (0 or 1)."
fi

for s in $sampling_method
  do
  for c in $class_name
  do
    for r in $anomaly_ratio
    do
      echo "sampling_method: $s class_name: $c anomaly_ratio: $r"      
      CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
      default_setting=./configs/default/mvtecad.yaml \
      model_setting=./configs/model/rd.yaml \
      DATASET.class_name=$c \
      DATASET.params.anomaly_ratio=$r \
      MODEL.params.sampling_method=$s
      done
  done
done


```

# Update record 

**2024-11-09**
- commit : [dm24] final backup 
  - dm24 까지의 내용 모두 완료한 뒤 백업함 
  - contrastive loss with only pre-task negative sample 까지 구현 완료 
  - 현재까지, 해당 버전의 SOTA - contrastive with only hard negative + lr 0.005 + no margin	

**2024-11-25**
- commit : [24-11-25] num_neg_sample / sampling method 
  - num neg sample 한 번에 2개 이상 load 할 수 있도록 데이터셋, 모델 수정 
  - coresetsampling - sampling 수 int로 입력 받아서 처리 할 수 있도록 수정 