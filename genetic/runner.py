import csv
import numpy as np
import random

# 경로 설정
import os
import datetime

# --- 1. 데이터 로딩 ---
def load_data(file_path):
    """
    CSV 파일에서 학생 정보를 읽어옵니다.
    """
    try:
        with open(file_path, newline='', encoding='cp949') as csvfile:
            reader = csv.DictReader(csvfile)
            data = [row for row in reader]
            # '기수'는 int로 변환
            for d in data:
                d['기수'] = int(d['기수'])
            return data
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. CSV 파일이 코드와 같은 경로에 있는지 확인하세요.")
        return None

# --- 2. 유전 알고리즘 핵심 함수 ---

def create_chromosome(population_size, num_groups):
    """
    하나의 조 편성(염색체)을 생성합니다.
    학생 인덱스를 무작위로 섞은 후, 조 개수만큼 나눕니다.
    """
    chromosome = list(range(population_size))
    random.shuffle(chromosome)
    return np.array_split(chromosome, num_groups)

def calculate_fitness(chromosome, population_data, options):
    """
    주어진 조 편성(염색체)의 적합도를 계산합니다.
    점수가 높을수록 좋은 편성입니다.
    """
    score = 0
    
    # 각 옵션에 대한 평가 점수를 계산하고 가중치를 곱해 합산합니다.
    score += options.get('gender_balance', 0) * score_gender_balance(chromosome, population_data)
    score += options.get('class_balance', 0) * score_class_balance(chromosome, population_data)
    #score += options.get('contribution_balance', 0) * score_contribution_balance(chromosome, population_data)
    #score += options.get('exclusive_penalty', 0) * penalty_exclusive_members(chromosome, population_data)
    #score += options.get('previous_team_penalty', 0) * penalty_previous_team(chromosome, population_data)
    
    return score

# --- 3. 적합도 평가 세부 함수 ---

def score_gender_balance(chromosome, data):
    """성비 균등 점수: 각 조의 남성 비율의 분산이 작을수록 높은 점수를 반환합니다."""
    gender_ratios = []
    for group in chromosome:
        if not group.size: continue
        males = sum(1 for i in group if data[i]["성별"] == "남")
        ratio = males / len(group)
        gender_ratios.append(ratio)
    # 분산이 작을수록 좋으므로, (1 - 분산) 값을 사용
    return 1 - np.var(gender_ratios)

def score_class_balance(chromosome, data):
    """기수 균등 점수: 각 조의 기수 평균의 분산이 작을수록 높은 점수를 반환합니다."""
    avg_classes = []
    for group in chromosome:
        if not group.size: continue
        total_class = sum(data[i]["기수"] for i in group)
        avg_classes.append(total_class / len(group))
    return 1 - np.var(avg_classes)

def score_contribution_balance(chromosome, data):
    """기여도 분포 점수: 각 조의 기여도 합의 분산이 작을수록 높은 점수를 반환합니다."""
    #sum_contributions = []
    #for group in chromosome:
    #    if not group.size: continue
    #    sum_contributions.append(sum(data[i]["기여도"] for i in group))
    #return 1 - np.var(sum_contributions)

### 수정요망 - 배타인원 데이터 처리방식 수정 필요함
def penalty_exclusive_members(chromosome, data):
    """배타 인원 페널티: 배타 인원이 같은 조에 있으면 페널티(음수)를 부과합니다."""
    #penalty_count = 0
    #name_to_index = {d['이름']: i for i, d in enumerate(data)}
    #for group in chromosome:
    #    for i in group:
    #        exclusive_name = data[i]["배타 인원"]
    #        if exclusive_name and exclusive_name in name_to_index:
    #            exclusive_index = name_to_index[exclusive_name]
    #            if exclusive_index in group:
    #                penalty_count += 1
    ## 페널티는 0 또는 음수여야 하므로 -1을 곱합니다. (한 쌍이므로 2로 나눔)
    #return -penalty_count / 2

### 수정요망 - 상대적으로 낮아보이는 패널티로, 패널티 점수 조정이 필요함
def penalty_previous_team(chromosome, data):
    """이전 조원 페널티: 이전 조원이 같은 조에 포함된 경우 페널티를 부과합니다."""
    #penalty_count = 0
    #for group in chromosome:
    #    if len(group) < 2: continue
    #    previous_teams = [data[i]["이전 조"] for i in group]
    #    # 같은 이전 조 출신이 몇 명인지 카운트 (중복 조합)
    #    unique_teams, counts = np.unique(previous_teams, return_counts=True)
    #    for count in counts:
    #        if count > 1:
    #            # nC2 계산: n * (n-1) / 2
    #            penalty_count += count * (count - 1) / 2
    #return -penalty_count

# --- 4. 유전 알고리즘 실행부 ---
def genetic_algorithm(population_data, num_groups, options, generations, population_size, elite_size, mutation_rate):
    """
    유전 알고리즘을 실행하여 최적의 조 편성을 찾습니다.
    """
    population_num = len(population_data)
    
    # 1. 초기 세대 생성
    population = [create_chromosome(population_num, num_groups) for _ in range(population_size)]

    for gen in range(generations):
        # 2. 적합도 평가

        fitness_scores = np.array([calculate_fitness(c, population_data, options) for c in population])
        # 모든 점수가 0 이상이 되도록 shift
        min_score = np.min(fitness_scores)
        if min_score < 0:
            fitness_scores = fitness_scores - min_score + 1e-6  # 0보다 작으면 shift

        # 3. 선택 (엘리트 보존 + 룰렛 휠 선택)
        elites_indices = np.argsort(fitness_scores)[-elite_size:]
        next_generation = [population[i] for i in elites_indices]

        # 룰렛 휠 선택을 위한 준비
        fitness_sum = np.sum(fitness_scores)
        selection_probs = fitness_scores / fitness_sum if fitness_sum > 0 else np.full(population_size, 1/population_size)

        # 4. 교차 및 변이
        while len(next_generation) < population_size:
            # 부모 선택
            parents_indices = np.random.choice(population_size, 2, p=selection_probs, replace=False)
            parent1, parent2 = population[parents_indices[0]], population[parents_indices[1]]
            
            # 교차 (Cycle Crossover)
            flat_p1 = np.concatenate(parent1)
            flat_p2 = np.concatenate(parent2)
            
            child1_flat = [-1] * population_num
            
            cycle_indices = []
            start_index = 0
            while start_index not in cycle_indices:
                cycle_indices.append(start_index)
                start_index = np.where(flat_p1 == flat_p2[start_index])[0][0]

            for i in range(population_num):
                if i in cycle_indices:
                    child1_flat[i] = flat_p1[i]
                else:
                    child1_flat[i] = flat_p2[i]
            
            child = np.array_split(child1_flat, num_groups)

            # 변이 (Swap Mutation)
            if random.random() < mutation_rate:
                group_idx = random.randrange(num_groups)
                if len(child[group_idx]) >= 2:
                    idx1, idx2 = random.sample(range(len(child[group_idx])), 2)
                    child[group_idx][idx1], child[group_idx][idx2] = child[group_idx][idx2], child[group_idx][idx1]
            
            next_generation.append(child)
        
        population = next_generation
        print(f"Generation {gen+1}/{generations}, Best Fitness: {np.max(fitness_scores):.4f}")

    # 최종 결과 반환
    final_fitness_scores = np.array([calculate_fitness(c, population_data, options) for c in population])
    best_chromosome_index = np.argmax(final_fitness_scores)
    return population[best_chromosome_index], final_fitness_scores[best_chromosome_index]


def print_result(chromosome, data, output_base):
    """결과를 보기 쉽게 출력합니다."""
    print("\n=== 최종 조 편성 결과 ===")
    result_rows = []
    for i, group in enumerate(chromosome):
        group_members = [data[idx] for idx in group]
        member_names = [member['이름'] for member in group_members]
        males = sum(1 for m in group_members if m['성별'] == 'M')
        females = len(group_members) - males
        avg_class = np.mean([m['기수'] for m in group_members])
        #sum_contrib = sum(m['기여도'] for m in group_members)

        print(f"\n--- 조 {i+1} (인원: {len(group_members)}명) ---")
        print(f"  - 멤버: {', '.join(member_names)}")
        print(f"  - 성비: 남 {males}명, 여 {females}명")
        print(f"  - 평균 기수: {avg_class:.2f}")
        #print(f"  - 기여도 합계: {sum_contrib}")

        for member in group_members:
            result_rows.append({
                '조': i+1,
                '이름': member['이름'],
                '기수': member['기수'],
                '성별': member['성별']
            })

    # CSV로 저장 (타임스탬프 기반 파일명)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_base, exist_ok=True)
    filename = os.path.join(output_base, f'result_{timestamp}.csv')
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['조', '이름', '기수', '성별'])
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"\n결과가 {filename} 파일로 저장되었습니다.")
        
# --- 5. 메인 실행부 ---
if __name__ == "__main__":
    # --- 설정 변수 ---
    # input/output base 경로 선언
    INPUT_BASE = "./data/input"
    OUTPUT_BASE = "./data/output"
    FILE_PATH = os.path.join(INPUT_BASE, "회원명단.csv")    # 학생 정보 CSV 파일 경로

    # --- 한 조당 인원수 제약조건 ---
    MIN_GROUP_SIZE = 4   # 한 조당 최소 인원수
    MAX_GROUP_SIZE = 6   # 한 조당 최대 인원수

    # --- 유전 알고리즘 하이퍼파라미터 ---
    GENERATIONS = 200              # 세대 수
    POPULATION_SIZE = 100          # 한 세대의 염색체(인구) 수
    ELITE_SIZE = 10                # 다음 세대에 보존할 상위 엘리트 개수
    MUTATION_RATE = 0.1            # 변이 확률

    # --- 옵션별 가중치 설정 (중요도에 따라 조절) ---
    OPTIONS = {
        'gender_balance': 1.0,           # 성비 균등 (1점에 가까울수록 좋음)
        'class_balance': 1.5,            # 기수 균등 (1점에 가까울수록 좋음)
        #'contribution_balance': 1.2,     # 기여도 균등 (1점에 가까울수록 좋음)
        #'exclusive_penalty': -10.0,      # 배타 인원 페널티 (0점에 가까울수록 좋음)
        #'previous_team_penalty': -5.0,   # 이전 조원 페널티 (0점에 가까울수록 좋음)
    }

    # --- 프로그램 실행 ---
    student_data = load_data(FILE_PATH)
    if student_data:
        total_students = len(student_data)
        # 가능한 조 개수 범위 계산
        min_groups = (total_students + MAX_GROUP_SIZE - 1) // MAX_GROUP_SIZE
        max_groups = total_students // MIN_GROUP_SIZE
        best_overall_score = float('-inf')
        best_overall_grouping = None
        best_num_groups = None

        print(f"전체 학생 수: {total_students}명")
        print(f"한 조당 최소 인원: {MIN_GROUP_SIZE}, 최대 인원: {MAX_GROUP_SIZE}")
        print(f"가능한 조 개수: {min_groups} ~ {max_groups}개")

        for num_groups in range(min_groups, max_groups + 1):
            # 조 개수에 맞게 각 조 인원수가 제약조건을 만족하는지 확인
            base_size = total_students // num_groups
            extra = total_students % num_groups
            group_sizes = [base_size + 1 if i < extra else base_size for i in range(num_groups)]
            if min(group_sizes) < MIN_GROUP_SIZE or max(group_sizes) > MAX_GROUP_SIZE:
                continue

            print(f"\n--- {num_groups}개 조로 시도 중... ---")
            grouping, score = genetic_algorithm(
                population_data=student_data,
                num_groups=num_groups,
                options=OPTIONS,
                generations=GENERATIONS,
                population_size=POPULATION_SIZE,
                elite_size=ELITE_SIZE,
                mutation_rate=MUTATION_RATE
            )
            print(f"적합도 점수: {score:.4f}")
            if score > best_overall_score:
                best_overall_score = score
                best_overall_grouping = grouping
                best_num_groups = num_groups

        if best_overall_grouping is not None:
            print(f"\n=== 최적의 조 개수: {best_num_groups}개 ===")
            print_result(best_overall_grouping, student_data, OUTPUT_BASE)
            print(f"\n최종 적합도 점수: {best_overall_score:.4f}")
        else:
            print("제약조건을 만족하는 조 개수가 없습니다. 최소/최대 인원수를 조정해보세요.")