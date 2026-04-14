import pandas as pd
import numpy as np
import scipy.stats
import itertools
import warnings

# ==========================================================
# DeLong Test Implementation
# ==========================================================

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T + 1
    return T2

def delong_roc_variance(ground_truth, predictions):
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    predictions_sorted_transposed = predictions[:, order]
    m = label_1_count
    n = len(ground_truth) - m
    
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    p_value = 2 * scipy.stats.norm.sf(np.abs(z))
    return float(p_value)

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    try:
        aucs, cov = delong_roc_variance(ground_truth, np.vstack((predictions_one, predictions_two)))
        return calc_pvalue(aucs, cov)
    except Exception as e:
        import traceback
        traceback.print_exc()
        warnings.warn(f"DeLong test failed: {e}")
        return np.nan

# ==========================================================
# Script Execution
# ==========================================================

if __name__ == "__main__":
    EXCEL_PATH = r"D:\Data\Jmszxyy\职称申报骨质疏松中文课题\投稿\20260414返修\最终表格.xlsx"
    LABEL_COL = "Label"
    MODELS = ["CM", "DLM", "FM"]
    
    # === 新增：子集过滤选项 (留空或设为 None 则不进行过滤) ===
    SUBSET_COLUMN = "split_fold_1"  # 例如 "split_fold_1"
    SUBSET_VALUE = "val"           # 例如 "test1"
    
    print(f"正在读取表格数据: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    
    # 如果指定了过滤条件，先筛选出指定行
    if SUBSET_COLUMN and SUBSET_VALUE:
        if SUBSET_COLUMN in df.columns:
            original_len = len(df)
            df = df[df[SUBSET_COLUMN] == SUBSET_VALUE].copy()
            print(f"应用子集过滤 [{SUBSET_COLUMN} == '{SUBSET_VALUE}']: 保留 {len(df)} / {original_len} 行")
        else:
            print(f"⚠️ 警告: 表格中未找到指定的过滤列 {SUBSET_COLUMN}，将略过过滤！")
    
    # 丢弃存在缺失值的行
    df_clean = df[[LABEL_COL] + MODELS].dropna()
    print(f"清洗完成，有效测试样本数: {len(df_clean)} \n")
    
    ground_truth = df_clean[LABEL_COL].values
    
    results = []
    # 两两对比计算 Delong P value
    for m1, m2 in itertools.combinations(MODELS, 2):
        pred1 = df_clean[m1].values
        pred2 = df_clean[m2].values
        
        p_val = delong_roc_test(ground_truth, pred1, pred2)
        p_val = float(np.squeeze(p_val)) if pd.notna(p_val) else np.nan
        
        stars = ""
        if pd.notna(p_val):
            if p_val < 0.001:
                stars = "***"
            elif p_val < 0.01:
                stars = "**"
            elif p_val < 0.05:
                stars = "*"
            else:
                stars = "ns"
        
        results.append((f"{m1} vs {m2}", p_val, stars))
    
    print("=" * 50)
    print("DeLong Test 对比结果 (P Value)")
    print("=" * 50)
    print(f"{'对比项':<15} | {'P-Value':<12} | {'显著性'}")
    print("-" * 50)
    for res in results:
        # Avoid scientific notation, round to 5 decimal places
        p_str = f"{res[1]:.5f}" if hasattr(res[1], 'real') and res[1] > 0.00001 else f"{res[1]:.2e}"
        print(f"{res[0]:<15} | {p_str:<12} | {res[2]}")
    print("\n注: * p < 0.05, ** p < 0.01, *** p < 0.001, ns: 无显著差异")

