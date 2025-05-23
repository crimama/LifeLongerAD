# Informative vs. Generic Predicate 비율 분석 (Paired t-test)

import json
import re
from collections import Counter
import numpy as np
from scipy.stats import entropy, ttest_rel

# --------------------
# 1) 데이터 로딩
# --------------------
with open('/workspace/Unbiased_Image_Captioning/caption_parsed/coco_ietrans_karpathy/coco_ietrans_all.json',
          'r', encoding='utf-8') as f:
    data = json.load(f)

# --------------------
# 2) Scene Graph에서 predicate(술어)만 추출하는 함수
# --------------------
def extract_middle_words(scene_graph: str):
    """
    scene_graph 문자열에서 (subj, pred, obj) triplet을 뽑아
    predicate만 리스트로 리턴
    """
    triplets = re.findall(r'\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,]+?)\s*\)', scene_graph)
    return [pred.strip().lower() for (_, pred, _) in triplets]

# --------------------
# 3) Shannon entropy 계산 함수
# --------------------
def predicate_entropy(preds: list[str]) -> float:
    """
    preds: list of predicate strings
    returns: Shannon entropy in bits of the empirical distribution
    """
    if not preds:
        return 0.0
    counts = Counter(preds)
    total = sum(counts.values())
    probs = np.array(list(counts.values()), dtype=float) / total
    # p>0인 것만 골라 –∑ p log2 p
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


# --------------------
# 4) per-image entropy 계산
# --------------------
baseline_ents = []
internal_ents = []
final_ents    = []

for item in data.values():
    # 각 stage별 scene_graph에서 술어를 뽑아서 entropy 계산
    for stage, ents_list in [
        ('baseline_scene_graph', baseline_ents),
        ('internal_scene_graph', internal_ents),
        ('final_scene_graph',    final_ents),
    ]:
        sg = item.get(stage, '')
        preds = extract_middle_words(sg)
        ents_list.append(predicate_entropy(preds))

# numpy 배열로 변환
baseline_ents = np.array(baseline_ents)
internal_ents = np.array(internal_ents)
final_ents    = np.array(final_ents)


# ——————————
# 5) Paired T-test 수행
# --------------------
# (nan이 없어야 정상; 만약 nan이 있다면 drop하거나 nan_policy='omit' 옵션 추가)
t_stat_i, p_val_i = ttest_rel(internal_ents, baseline_ents, nan_policy='omit')
t_stat_f, p_val_f = ttest_rel(final_ents,    baseline_ents, nan_policy='omit')

print(f"[Internal vs Baseline] t-statistic: {t_stat_i:.4f}, p-value: {p_val_i:.4e}")
print(f"[Final vs Baseline]    t-statistic: {t_stat_f:.4f}, p-value: {p_val_f:.4e}")