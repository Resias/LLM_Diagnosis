from collections import Counter

def count_classes(dataset):
    """
    dataset: OrderFreqDataset
    Returns:
        class_counter: Counter mapping {class_index: count}
    """
    # 클래스 이름(string) 기준으로 카운트
    name_counts = Counter(dataset.dataset_df['class_name'])

    # 클래스 인덱스 기준으로 변환
    class_counter = Counter()
    for name, count in name_counts.items():
        idx = dataset.classes.index(name)
        class_counter[idx] = count

    return class_counter