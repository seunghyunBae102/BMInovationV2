import numpy as np
import pandas as pd

# ==========================================
# Inference Engine v1.5 (Complete)
# ==========================================
# [Update Log]
# - 매체 계층 구조(Hierarchy) 지원
# - 욕구(Needs) 기반 보상 가중치 (재미 vs 성장)
# - 몰입 이론(Flow)에 따른 난이도 패널티
# - 관성(Inertia) 및 매체 포화(Saturation) 로직 추가
# ==========================================

# ---------------------------------------------------------
# 1. Constants & Pre-computation Helpers
# ---------------------------------------------------------

TAG_LIST = [
    "Competition", "Skill", "Growth", "RPG", "Gambling", "Collection", 
    "Free", "Patience", "Humor", "Trend", "Social", "Info", 
    "Responsibility", "Relax", "Strategy", "KillingTime", "Story", 
    "Future", "Knowledge"
]
TAG_TO_IDX = {tag: i for i, tag in enumerate(TAG_LIST)}

# 매체 그룹 정의 (genesis.py의 NUM_MEDIA_TYPES=6 과 순서 일치 필수)
MEDIA_TYPES = ["GAME", "VIDEO", "BOOK", "WORK", "COMM", "LIFE"]
MEDIA_TO_IDX = {m: i for i, m in enumerate(MEDIA_TYPES)}
NUM_MEDIA_TYPES = len(MEDIA_TYPES)

def precompute_activity_tags_matrix(df_activities, num_tags=50):
    """
    활동(M) x 태그(50) One-hot Matrix
    """
    num_acts = len(df_activities)
    act_tag_matrix = np.zeros((num_acts, num_tags))
    
    for i, tags in enumerate(df_activities['Tags']):
        if isinstance(tags, str):
            tag_list = tags.split('|')
        else:
            tag_list = tags if isinstance(tags, list) else []

        for tag in tag_list:
            if tag in TAG_TO_IDX:
                idx = TAG_TO_IDX[tag]
                act_tag_matrix[i, idx] = 1.0
    return act_tag_matrix

def precompute_media_matrix(df_activities):
    """
    활동(M) x 매체유형(K) One-hot Matrix
    목적: 활동이 어떤 매체 그룹에 속하는지 벡터화
    """
    num_acts = len(df_activities)
    num_media = len(MEDIA_TYPES)
    act_media_matrix = np.zeros((num_acts, num_media))
    
    if 'Media_Group' not in df_activities.columns:
        return act_media_matrix # Fallback

    for i, media_group in enumerate(df_activities['Media_Group']):
        if media_group in MEDIA_TO_IDX:
            idx = MEDIA_TO_IDX[media_group]
            act_media_matrix[i, idx] = 1.0
            
    return act_media_matrix

# ---------------------------------------------------------
# 2. Advanced Utility Calculation (The Brain)
# ---------------------------------------------------------

def calculate_utility(agents, df_activities, act_tag_matrix, act_media_matrix, time_context):
    """
    [v1.5 Advanced Formula]
    Utility = (Weighted_Rewards * Interest_Mod) 
              + Inertia_Bonus 
              - (Difficulty_Penalty + Saturation_Penalty + Weighted_Costs)
    """
    n_agents = len(agents['ids'])
    n_acts = len(df_activities)
    
    # --- A. Data Extraction (Broadcasting Prep) ---
    # Activity Attributes [1, M]
    vec_fun = df_activities.get('Fun_Reward', pd.Series(0)).values.reshape(1, -1)
    vec_growth = df_activities.get('Growth_Reward', pd.Series(0)).values.reshape(1, -1)
    vec_diff = df_activities.get('Difficulty', pd.Series(0)).values.reshape(1, -1)
    
    vec_stress_cost = df_activities['Stress_Cost'].values.reshape(1, -1)
    vec_money_cost = df_activities['Cost'].values.reshape(1, -1)
    
    # Agent States [N, 1]
    state_dopamine = agents['state_dopamine']
    state_anxiety = agents['state_anxiety']
    state_stress = agents['state_stress']
    state_current_media = agents['state_current_media'] # Int indices
    traits_intel = agents['traits_intel']
    loss_aversion = agents['loss_aversion']
    
    # -------------------------------------------------------
    # 1. Needs Weighting (욕구 기반 보상 가중치)
    # -------------------------------------------------------
    # 도파민이 낮을수록(부족할수록) Fun 보상 가치 상승
    # 불안도가 높을수록 Growth 보상 가치 상승
    
    w_fun = (100.0 - state_dopamine) / 100.0
    w_fun = np.clip(w_fun, 0.1, 2.0) # 최소 0.1, 최대 2배
    
    w_growth = 1.0 + (state_anxiety / 20.0) # 불안도가 10이면 1.5배, 0이면 1.0배
    
    # [N, M] = ([1, M] * [N, 1]) + ([1, M] * [N, 1])
    base_utility = (vec_fun * w_fun) + (vec_growth * w_growth)
    
    # 취향 매칭 보정 (기존 로직 유지)
    interest_scores = np.dot(agents['interests'], act_tag_matrix.T) # [N, M]
    base_utility *= (1.0 + interest_scores)

    # -------------------------------------------------------
    # 2. Difficulty Penalty (몰입 이론 - Flow)
    # -------------------------------------------------------
    # 활동 난이도가 에이전트 지능보다 높으면 패널티 (좌절감)
    # 낮으면(너무 쉬우면) 약간의 패널티 (지루함) - 여기선 좌절감만 구현
    
    diff_gap = vec_diff - traits_intel # [N, M]
    # Gap > 0 이면 어려움. 어려울수록 가파르게 패널티 증가
    penalty_flow = np.maximum(diff_gap, 0) * 1.5 
    
    # -------------------------------------------------------
    # 3. Inertia Bonus (관성 효과)
    # -------------------------------------------------------
    # 내가 직전에 했던 매체와 같은 매체 그룹의 활동에는 가산점
    
    # Agent Media State를 One-hot으로 변환 [N, K]
    # -1(None)인 경우를 처리하기 위해 +1 크기로 만들고 슬라이싱하거나
    # 단순히 0으로 채워진 배열에 인덱싱
    agent_media_onehot = np.zeros((n_agents, NUM_MEDIA_TYPES))
    
    # 유효한 인덱스(0 이상)를 가진 에이전트만 1.0 설정
    valid_mask = (state_current_media >= 0).flatten()
    valid_indices = state_current_media[valid_mask].flatten()
    # numpy advanced indexing
    # 행: valid한 에이전트 순번, 열: 해당 에이전트의 매체 인덱스
    if np.any(valid_mask):
        agent_media_onehot[valid_mask, valid_indices] = 1.0
        
    # [N, M] = [N, K] @ [M, K].T
    # 값이 1.0이면 매체 일치, 0.0이면 불일치
    inertia_matrix = np.dot(agent_media_onehot, act_media_matrix.T)
    
    # 같은 매체면 +10점 (전환 비용 절감 효과)
    inertia_bonus = inertia_matrix * 10.0

    # -------------------------------------------------------
    # 4. Media Saturation (매체 물림)
    # -------------------------------------------------------
    # 특정 매체에 대한 지루함이 높으면 해당 매체 활동 전체 효용 감소
    # agents['media_boredom']: [N, K]
    
    # [N, M] = [N, K] @ [M, K].T
    saturation_penalty = np.dot(agents['media_boredom'], act_media_matrix.T) * 2.0
    
    # -------------------------------------------------------
    # 5. Cost & Context Calculation
    # -------------------------------------------------------
    stress_mod = time_context['Stress_Mod']
    
    # 스트레스 비용 + 돈 비용
    pain_stress = (vec_stress_cost * stress_mod) * (1.0 + (state_stress * 0.01))
    pain_money = vec_money_cost * 0.001 
    
    # Prospect Theory: Loss Aversion applied to costs
    total_pain = (pain_stress + pain_money) * loss_aversion

    # -------------------------------------------------------
    # 6. Final Aggregation
    # -------------------------------------------------------
    utility_matrix = base_utility + inertia_bonus - penalty_flow - saturation_penalty - total_pain
    
    # Stochasticity (Noise)
    noise = np.random.normal(0, 2.0, size=(n_agents, n_acts))
    utility_matrix += noise
    
    return utility_matrix

# ---------------------------------------------------------
# 3. Decision Making (Knapsack Greedy)
# ---------------------------------------------------------

def decide_actions_knapsack(utility_matrix, df_activities, agents):
    """
    [벡터화된 냅색 알고리즘 - 기존 유지]
    """
    n_agents, n_acts = utility_matrix.shape
    agent_caps = agents['attention_cap'] # [N, 1]
    
    intensities = df_activities['Intensity'].values.reshape(1, -1)
    
    # Intensity가 0인 경우(휴식 등) 나누기 에러 방지
    safe_intensities = intensities.copy()
    safe_intensities[safe_intensities == 0] = 0.1
    
    # 가성비 계산 (음수 효용은 선택 안 함 -> 0 처리보다, 순위 뒤로 밀리게 둠)
    ratios = utility_matrix / safe_intensities
    
    # 정렬
    sorted_indices = np.argsort(ratios, axis=1)[:, ::-1] # [N, M]
    row_indices = np.arange(n_agents)[:, np.newaxis]
    
    sorted_intensities = intensities[0][sorted_indices]
    cum_intensities = np.cumsum(sorted_intensities, axis=1)
    
    allowed_mask_sorted = cum_intensities <= agent_caps
    
    # 효용이 0 이하인 활동은 굳이 하지 않도록 필터링 (선택적)
    # 여기서는 강제하지 않음 (스트레스 해소 등 목적이 있을 수 있으므로)
    
    final_mask = np.zeros((n_agents, n_acts), dtype=bool)
    flat_sorted_indices = row_indices * n_acts + sorted_indices
    final_mask.ravel()[flat_sorted_indices.ravel()] = allowed_mask_sorted.ravel()
    
    return final_mask