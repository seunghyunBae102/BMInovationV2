import pandas as pd
import os

# ==========================================
# Configuration Loader (CSV Based)
# ==========================================
# CSV 파일을 읽어 Pandas DataFrame으로 반환합니다.
# ==========================================

DATA_PATH = './data'

def load_activity_table():
    """
    data/activities.csv를 로드합니다.
    Tags 컬럼은 문자열('Tag1|Tag2')을 리스트(['Tag1', 'Tag2'])로 변환합니다.
    """
    file_path = os.path.join(DATA_PATH, 'activities.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}. Please run 'create_csv_data.py' first.")
    
    df = pd.read_csv(file_path)
    
    # 'Tags' 컬럼 전처리: "Tag1|Tag2" -> ["Tag1", "Tag2"]
    # NaN 값이 있을 경우 빈 리스트로 처리
    df['Tags'] = df['Tags'].fillna("").apply(lambda x: x.split('|') if x else [])
    
    return df

def load_time_slots():
    """
    data/time_slots.csv를 로드합니다.
    """
    file_path = os.path.join(DATA_PATH, 'time_slots.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}. Please run 'create_csv_data.py' first.")
    
    return pd.read_csv(file_path)