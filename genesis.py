import numpy as np
import pandas as pd

# ==========================================
# Genesis Module (에이전트 팩토리)
# ==========================================
# 목표: 10,000명의 에이전트 상태 행렬을 벡터화 연산으로 생성
# ==========================================

def create_agent_population(n_agents=10000):
    """
    n_agents 수만큼의 에이전트 데이터를 생성하여 딕셔너리(Tensor 모음) 형태로 반환
    """
    print(f"Creating {n_agents} agents via Vectorization...")
    
    # ----------------------------------
    # 1. Static Traits (불변 성향)
    # ----------------------------------
    # Big5: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
    # 0.0 ~ 1.0 사이 정규분포 (평균 0.5, 표준편차 0.15) 후 클리핑
    traits_big5 = np.random.normal(loc=0.5, scale=0.15, size=(n_agents, 5))
    traits_big5 = np.clip(traits_big5, 0.0, 1.0)
    
    # 심화 심리 변수 (v1.1)
    # Loss Aversion: 평균 2.25 (전망 이론 기준값), 표준편차 0.5
    loss_aversion = np.random.normal(loc=2.25, scale=0.5, size=(n_agents, 1))
    loss_aversion = np.clip(loss_aversion, 1.0, 5.0) # 최소 1.0 이상
    
    # Attention Capacity (주의력 총량): 평균 100, 표준편차 10
    # 성실성(Big5[1])이 높을수록 주의력 총량이 약간 높도록 보정 (+ noise)
    base_cap = np.random.normal(loc=100, scale=10, size=(n_agents, 1))
    conscientiousness = traits_big5[:, 1].reshape(-1, 1)
    attention_cap = base_cap + (conscientiousness * 20) 
    attention_cap = np.clip(attention_cap, 50, 200).astype(int)

    # ----------------------------------
    # 2. Dynamic States (가변 상태)
    # ----------------------------------
    # 초기 스트레스, 피로도, 지루함 = 0
    state_stress = np.zeros((n_agents, 1))
    state_fatigue = np.zeros((n_agents, 1))
    state_boredom = np.zeros((n_agents, 1))
    
    # 지갑 (Wallet): 로그 정규분포 (빈부격차 반영)
    # 초기 자금 평균 10,000 ~ 1,000,000원 사이 분포
    wallet = np.random.lognormal(mean=10, sigma=1, size=(n_agents, 1)).astype(int)
    
    # ----------------------------------
    # 3. Interests (취향 벡터)
    # ----------------------------------
    # 가상의 흥미 태그 50개에 대한 가중치 (0.0 ~ 1.0)
    # 희소성(Sparsity)을 주기 위해 일부는 0으로 만듦
    num_interest_tags = 50
    interests = np.random.rand(n_agents, num_interest_tags)
    
    # 약 30%의 취향만 활성화 (나머지는 관심 없음 0.0)
    mask = np.random.rand(n_agents, num_interest_tags) > 0.3
    interests[mask] = 0.0
    
    # ----------------------------------
    # 4. Result Packing
    # ----------------------------------
    # 모든 데이터를 딕셔너리에 담아 반환 (DataFrame보다 연산이 빠른 ndarray 유지)
    population = {
        "ids": np.arange(n_agents),
        "traits_big5": traits_big5,       # [N, 5]
        "loss_aversion": loss_aversion,   # [N, 1]
        "attention_cap": attention_cap,   # [N, 1]
        "state_stress": state_stress,     # [N, 1]
        "state_fatigue": state_fatigue,   # [N, 1]
        "state_boredom": state_boredom,   # [N, 1]
        "wallet": wallet,                 # [N, 1]
        "interests": interests            # [N, 50]
    }
    
    return population

def print_agent_sample(population, agent_idx=0):
    """
    특정 에이전트의 데이터를 가독성 있게 출력 (검증용)
    """
    print(f"\n[Agent #{agent_idx} Sample Profile]")
    print(f"- Attention Cap: {population['attention_cap'][agent_idx][0]}")
    print(f"- Loss Aversion: {population['loss_aversion'][agent_idx][0]:.2f}")
    print(f"- Wallet: {population['wallet'][agent_idx][0]:,} Gold")
    print(f"- Big5 (O-C-E-A-N): {population['traits_big5'][agent_idx]}")
    active_interests = np.where(population['interests'][agent_idx] > 0)[0]
    print(f"- Active Interest Tags Count: {len(active_interests)} / 50")