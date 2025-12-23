import numpy as np
import pandas as pd

# ==========================================
# Genesis Module v1.5
# ==========================================
# [Update Log]
# - traits_intel: 지능/숙련도 (난이도 저항력)
# - state_anxiety: 불안도 (성장 욕구 트리거)
# - state_current_media: 현재 머무는 매체 Context
# - media_boredom: 매체 그룹별 피로도 벡터
# ==========================================

# 매체 그룹 상수 정의 (CSV의 Media_Group 종류와 일치해야 함)
# GAME, VIDEO, BOOK, WORK, COMM, LIFE (총 6개 가정)
NUM_MEDIA_TYPES = 6 

def create_agent_population(n_agents=10000):
    print(f"Creating {n_agents} agents with Advanced Psychology (v1.5)...")
    
    # ----------------------------------
    # 1. Static Traits (불변 성향)
    # ----------------------------------
    traits_big5 = np.random.normal(loc=0.5, scale=0.15, size=(n_agents, 5))
    traits_big5 = np.clip(traits_big5, 0.0, 1.0)
    
    loss_aversion = np.random.normal(loc=2.25, scale=0.5, size=(n_agents, 1))
    loss_aversion = np.clip(loss_aversion, 1.0, 5.0)
    
    # [NEW] Intelligence / Literacy (지능/숙련도)
    # 평균 50, 표준편차 15. (0~100)
    # Difficulty가 높은 활동(독서, 하드코어 게임)의 진입장벽을 낮춰줌
    traits_intel = np.random.normal(loc=50, scale=15, size=(n_agents, 1))
    traits_intel = np.clip(traits_intel, 0, 100)

    # Attention Capacity
    base_cap = np.random.normal(loc=100, scale=10, size=(n_agents, 1))
    conscientiousness = traits_big5[:, 1].reshape(-1, 1)
    attention_cap = base_cap + (conscientiousness * 20) 
    attention_cap = np.clip(attention_cap, 50, 200).astype(int)

    # ----------------------------------
    # 2. Dynamic States (가변 상태)
    # ----------------------------------
    state_stress = np.zeros((n_agents, 1))
    state_fatigue = np.zeros((n_agents, 1))
    state_boredom = np.zeros((n_agents, 1))
    
    # [NEW] Anxiety (불안도)
    # 초기값은 0에 가깝지만 약간의 랜덤성 부여 (0~10)
    # 불안도가 높으면 Fun보다 Growth 보상을 추구함
    state_anxiety = np.random.uniform(0, 10, size=(n_agents, 1))
    
    # [NEW] Dopamine (도파민 수치) - 초기값 50 (보통)
    state_dopamine = np.full((n_agents, 1), 50.0)

    # Wallet
    wallet = np.random.lognormal(mean=10, sigma=1, size=(n_agents, 1)).astype(int)
    
    # ----------------------------------
    # 3. Context & Inertia (매체 맥락)
    # ----------------------------------
    # [NEW] Current Media Index
    # -1: None (초기 상태)
    state_current_media = np.full((n_agents, 1), -1, dtype=int)
    
    # [NEW] Media Boredom Vector
    # 각 매체 그룹(GAME, VIDEO...)별로 얼마나 질렸는지 기록 [N, 6]
    media_boredom = np.zeros((n_agents, NUM_MEDIA_TYPES))

    # ----------------------------------
    # 4. Interests (취향 벡터)
    # ----------------------------------
    num_interest_tags = 50
    interests = np.random.rand(n_agents, num_interest_tags)
    mask = np.random.rand(n_agents, num_interest_tags) > 0.3
    interests[mask] = 0.0
    
    # ----------------------------------
    # 5. Result Packing
    # ----------------------------------
    population = {
        "ids": np.arange(n_agents),
        "traits_big5": traits_big5,
        "traits_intel": traits_intel,       # [NEW]
        "loss_aversion": loss_aversion,
        "attention_cap": attention_cap,
        
        "state_stress": state_stress,
        "state_fatigue": state_fatigue,
        "state_boredom": state_boredom,
        "state_anxiety": state_anxiety,     # [NEW]
        "state_dopamine": state_dopamine,   # [NEW]
        "state_current_media": state_current_media, # [NEW]
        "media_boredom": media_boredom,     # [NEW]
        
        "wallet": wallet,
        "interests": interests
    }
    
    return population

def print_agent_sample(population, agent_idx=0):
    print(f"\n[Agent #{agent_idx} Profile v1.5]")
    print(f"- Intel/Skill: {population['traits_intel'][agent_idx][0]:.1f} / 100")
    print(f"- Anxiety: {population['state_anxiety'][agent_idx][0]:.1f}")
    print(f"- Attention Cap: {population['attention_cap'][agent_idx][0]}")
    print(f"- Current Media: {population['state_current_media'][agent_idx][0]}")
    print(f"- Big5: {population['traits_big5'][agent_idx]}")