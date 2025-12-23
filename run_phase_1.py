import genesis
import psy_sim_config
import create_csv_data # CSV 생성기 임포트
import time
import os

def main():
    print("=== [Phase 1] Data Initialization (CSV Based) ===")
    
    # 0. CSV 파일이 없으면 생성
    if not os.path.exists('data/activities.csv'):
        print("\n[Init] CSV files not found. Creating default data...")
        create_csv_data.create_initial_csvs()
    
    # 1. Config 로드 테스트 (CSV 로딩)
    print("\n1. Loading Configurations from CSV...")
    try:
        activities = psy_sim_config.load_activity_table()
        time_slots = psy_sim_config.load_time_slots()
        
        print(f"-> Activities Loaded: {len(activities)} types")
        # 로드된 데이터 확인 (Tags가 리스트로 잘 변환되었는지 확인)
        print(activities[['ID', 'Name', 'Intensity', 'Tags']].head().to_string(index=False))
        
        print(f"\n-> Time Slots Loaded: {len(time_slots)} slots")
        print(time_slots.head(5).to_string(index=False))
        
    except Exception as e:
        print(f"\n[Error] Failed to load config: {e}")
        return

    # 2. 에이전트 생성 테스트 (기존 genesis.py 사용 - 변경 없음)
    print("\n2. Generating Agent Population...")
    start_time = time.time()
    
    N_AGENTS = 10000
    population = genesis.create_agent_population(N_AGENTS)
    
    end_time = time.time()
    print(f"-> Generation Complete: {N_AGENTS} agents created in {end_time - start_time:.4f} seconds.")
    
    # 3. 데이터 검증
    print("\n3. Validating Agent Data...")
    genesis.print_agent_sample(population, agent_idx=0)
    
    print("\n=== [Phase 1] Initialization Successful ===")
    print("Now you can edit 'data/activities.csv' to change simulation rules.")

if __name__ == "__main__":
    main()