import pandas as pd
import numpy as np
import os
import create_csv_data

# ==========================================
# Configuration Loader v2.2 (Fix)
# ==========================================
# 이 파일은 시뮬레이션에 필요한 모든 CSV 데이터를 로드합니다.
# load_activity_table: 활동 데이터 로드
# load_life_patterns: 라이프 패턴 데이터 로드
# ==========================================

DATA_PATH = './data'

def load_activity_table():
    """
    data/activities.csv를 로드합니다.
    없으면 create_csv_data를 통해 생성합니다.
    """
    file_path = os.path.join(DATA_PATH, 'activities.csv')
    
    # 파일이 없으면 생성 시도
    if not os.path.exists(file_path):
        print("[System] Activities CSV missing. Creating defaults...")
        create_csv_data.create_initial_csvs()
    
    df = pd.read_csv(file_path)
    
    # 'Tags' 컬럼 전처리: "Tag1|Tag2" -> ["Tag1", "Tag2"]
    if 'Tags' in df.columns:
        df['Tags'] = df['Tags'].fillna("").apply(lambda x: x.split('|') if x else [])
    else:
        # Tags 컬럼이 아예 없으면 빈 리스트로 초기화
        df['Tags'] = [[] for _ in range(len(df))]
        
    return df

def load_life_patterns():
    """
    data/life_patterns.csv를 로드하여 시뮬레이션용 Lookup Table로 변환합니다.
    
    Returns:
        df (DataFrame): 원본 데이터
        stress_table (np.array): [96, 4] (Time x Pattern) 스트레스 계수
        ad_eff_table (np.array): [96, 4] (Time x Pattern) 광고 효율 계수
    """
    file_path = os.path.join(DATA_PATH, 'life_patterns.csv')
    
    if not os.path.exists(file_path):
        print("[System] Life Patterns CSV missing. Creating defaults...")
        create_csv_data.create_initial_csvs()
    
    df = pd.read_csv(file_path)
    
    # Pivot Table을 사용하여 [Time_Index(96) x Pattern_ID(4)] 형태의 행렬 생성
    # 빈 값은 1.0으로 채움
    
    # 1. Stress Modifier Matrix
    if 'Stress_Mod' in df.columns and 'Pattern_ID' in df.columns:
        stress_table = df.pivot(index='Time_Index', columns='Pattern_ID', values='Stress_Mod').fillna(1.0).values
    else:
        # 컬럼이 없으면 기본값 (96, 4) 1.0 행렬 반환
        stress_table = np.ones((96, 4))

    # 2. Ad Efficiency Matrix
    if 'Ad_Eff' in df.columns and 'Pattern_ID' in df.columns:
        ad_eff_table = df.pivot(index='Time_Index', columns='Pattern_ID', values='Ad_Eff').fillna(1.0).values
    else:
        ad_eff_table = np.ones((96, 4))
    
    # shape check: [96, 4] 이어야 함 (96틱, 4개 패턴)
    # 데이터가 부족할 경우를 대비해 resize 혹은 check
    if stress_table.shape[0] != 96:
        # 데이터가 96틱이 아니면 강제로 96으로 맞춤 (Zero padding or cutting)
        print(f"[Warning] Stress table shape mismatch {stress_table.shape}. Resizing to (96, 4)")
        new_stress = np.ones((96, 4))
        rows = min(96, stress_table.shape[0])
        cols = min(4, stress_table.shape[1])
        new_stress[:rows, :cols] = stress_table[:rows, :cols]
        stress_table = new_stress

    if ad_eff_table.shape[0] != 96:
        new_ad = np.ones((96, 4))
        rows = min(96, ad_eff_table.shape[0])
        cols = min(4, ad_eff_table.shape[1])
        new_ad[:rows, :cols] = ad_eff_table[:rows, :cols]
        ad_eff_table = new_ad
        
    return df, stress_table, ad_eff_table