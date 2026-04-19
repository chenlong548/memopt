"""
自动格式选择模块

根据稀疏矩阵的特征自动选择最优存储格式。
"""

from typing import Optional, Dict, Any, List
import numpy as np

from ..core.sparse_array import SparseArray
from ..core.config import SparseArrayConfig, SparseFormat, FormatFeatures, SelectionStrategy


# ==================== 格式选择常量 ====================
# 稀疏度阈值
SPARSITY_VERY_HIGH = 0.99      # 极高稀疏度阈值，使用COO
SPARSITY_HIGH = 0.95           # 高稀疏度阈值
SPARSITY_MEDIUM = 0.5          # 中等稀疏度阈值
DENSITY_HIGH = 0.3             # 高密度阈值

# 结构特征阈值
DIAGONAL_RATIO_HIGH = 0.9      # 对角线比例高阈值
BANDWIDTH_RATIO_LOW = 0.1      # 带宽比例低阈值
BLOCK_STRUCTURE_HIGH = 0.7     # 块结构评分高阈值
BLOCK_STRUCTURE_MEDIUM = 0.5   # 块结构评分中等阈值

# 非零元素分布阈值
NNZ_VARIANCE_HIGH = 100        # 非零元素方差高阈值
NNZ_VARIANCE_LOW = 10          # 非零元素方差低阈值

# 带宽阈值
BANDWIDTH_SMALL = 10           # 小带宽阈值
BANDWIDTH_MEDIUM = 100         # 中等带宽阈值

# 性能历史记录限制
MAX_PERFORMANCE_HISTORY = 100  # 最大性能历史记录数


class FormatSelector:
    """
    格式选择器

    根据稀疏矩阵特征和预期操作选择最优存储格式。
    """

    def __init__(self, strategy: SelectionStrategy = SelectionStrategy.RULE_BASED):
        """
        初始化选择器

        Args:
            strategy: 选择策略
        """
        self.strategy = strategy
        self._performance_history = {}

    def select(self,
               features: FormatFeatures,
               config: Optional[SparseArrayConfig] = None) -> SparseFormat:
        """
        选择最优格式

        Args:
            features: 矩阵特征
            config: 配置对象

        Returns:
            SparseFormat: 选择的格式
        """
        config = config or SparseArrayConfig()

        if self.strategy == SelectionStrategy.RULE_BASED:
            return self._rule_based_selection(features, config)
        elif self.strategy == SelectionStrategy.ML_BASED:
            return self._ml_based_selection(features, config)
        elif self.strategy == SelectionStrategy.PERFORMANCE_BASED:
            return self._performance_based_selection(features, config)
        else:
            return self._rule_based_selection(features, config)

    def _rule_based_selection(self,
                              features: FormatFeatures,
                              config: SparseArrayConfig) -> SparseFormat:
        """
        基于规则的选择

        使用启发式规则选择格式。
        """
        # 规则1：极高稀疏度，使用COO
        if features.sparsity > SPARSITY_VERY_HIGH:
            return SparseFormat.COO

        # 规则2：对角线矩阵，使用COO
        if features.diagonal_ratio > DIAGONAL_RATIO_HIGH:
            return SparseFormat.COO

        # 规则3：带状矩阵，使用CSR
        if features.bandwidth_ratio < BANDWIDTH_RATIO_LOW:
            return SparseFormat.CSR

        # 规则4：块结构明显，使用BCSR
        if features.block_structure_score > BLOCK_STRUCTURE_HIGH:
            return SparseFormat.BCSR

        # 规则5：预期操作是SpMV，使用CSR
        if features.expected_operation == 'spmv':
            return SparseFormat.CSR

        # 规则6：预期操作是列操作，使用CSC
        if features.expected_operation == 'col_access':
            return SparseFormat.CSC

        # 规则7：非零元素分布不均匀
        if features.row_nnz_variance > NNZ_VARIANCE_HIGH or features.col_nnz_variance > NNZ_VARIANCE_HIGH:
            return SparseFormat.COO

        # 规则8：中等稀疏度，规则分布
        if features.sparsity > SPARSITY_MEDIUM and features.is_regular:
            return SparseFormat.BITMAP

        # 默认：使用CSR
        return SparseFormat.CSR

    def _ml_based_selection(self,
                            features: FormatFeatures,
                            config: SparseArrayConfig) -> SparseFormat:
        """
        基于机器学习的选择

        使用训练好的模型选择格式。
        """
        # 简化实现：使用决策树规则
        feature_vector = features.to_feature_vector()

        # 这里可以加载预训练模型
        # 目前使用简化的决策规则

        if features.sparsity > SPARSITY_HIGH:
            if features.block_structure_score > BLOCK_STRUCTURE_MEDIUM:
                return SparseFormat.BCSR
            return SparseFormat.COO

        if features.density > DENSITY_HIGH:
            return SparseFormat.CSR

        if features.bandwidth < BANDWIDTH_SMALL:
            return SparseFormat.CSR

        return SparseFormat.CSR

    def _performance_based_selection(self,
                                     features: FormatFeatures,
                                     config: SparseArrayConfig) -> SparseFormat:
        """
        基于性能历史的选择

        根据历史性能数据选择格式。
        """
        # 查找相似矩阵的历史性能
        best_format = SparseFormat.CSR
        best_score = -1

        for fmt, scores in self._performance_history.items():
            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_format = fmt

        return best_format

    def update_performance(self,
                          format_type: SparseFormat,
                          performance_score: float):
        """
        更新性能历史

        Args:
            format_type: 格式类型
            performance_score: 性能评分
        """
        if format_type not in self._performance_history:
            self._performance_history[format_type] = []

        self._performance_history[format_type].append(performance_score)

        # 保持历史记录有限
        if len(self._performance_history[format_type]) > MAX_PERFORMANCE_HISTORY:
            self._performance_history[format_type] = \
                self._performance_history[format_type][-MAX_PERFORMANCE_HISTORY:]


def select_format(features: FormatFeatures,
                  config: Optional[SparseArrayConfig] = None) -> SparseFormat:
    """
    选择最优格式

    Args:
        features: 矩阵特征
        config: 配置对象

    Returns:
        SparseFormat: 选择的格式
    """
    config = config or SparseArrayConfig()
    selector = FormatSelector(config.selection_strategy)
    return selector.select(features, config)


def select_format_ml(features: FormatFeatures,
                     model_path: Optional[str] = None) -> SparseFormat:
    """
    使用机器学习模型选择格式

    Args:
        features: 矩阵特征
        model_path: 模型路径

    Returns:
        SparseFormat: 选择的格式
    """
    selector = FormatSelector(SelectionStrategy.ML_BASED)
    return selector.select(features)


def recommend_format(arr: SparseArray,
                     operation: str = 'spmv') -> Dict[str, Any]:
    """
    推荐最优格式

    Args:
        arr: 稀疏数组
        operation: 预期操作

    Returns:
        Dict: 推荐结果
    """
    from .features import extract_features_from_sparse

    features = extract_features_from_sparse(arr)
    features.expected_operation = operation

    config = SparseArrayConfig()
    selector = FormatSelector()

    recommended_format = selector.select(features, config)

    # 计算各格式的适用性评分
    scores = {}
    for fmt in [SparseFormat.CSR, SparseFormat.CSC, SparseFormat.COO,
                SparseFormat.BCSR, SparseFormat.BITMAP]:
        scores[fmt.value] = _compute_format_score(fmt, features, operation)

    return {
        'recommended_format': recommended_format.value,
        'scores': scores,
        'features': features.get_summary(),
        'reason': _get_recommendation_reason(recommended_format, features)
    }


def _compute_format_score(fmt: SparseFormat,
                          features: FormatFeatures,
                          operation: str) -> float:
    """计算格式适用性评分"""
    score = 0.0

    if fmt == SparseFormat.CSR:
        # CSR适合行操作和SpMV
        if operation == 'spmv':
            score += 0.4
        if operation == 'row_access':
            score += 0.3
        if features.row_nnz_variance < NNZ_VARIANCE_LOW:
            score += 0.2

    elif fmt == SparseFormat.CSC:
        # CSC适合列操作
        if operation == 'col_access':
            score += 0.4
        if operation == 'spmv_transpose':
            score += 0.3
        if features.col_nnz_variance < NNZ_VARIANCE_LOW:
            score += 0.2

    elif fmt == SparseFormat.COO:
        # COO适合构建和转换
        if operation == 'construction':
            score += 0.4
        if features.sparsity > SPARSITY_HIGH:
            score += 0.3
        if features.diagonal_ratio > 0.8:
            score += 0.2

    elif fmt == SparseFormat.BCSR:
        # BCSR适合块结构和GPU
        if features.block_structure_score > BLOCK_STRUCTURE_MEDIUM:
            score += 0.4
        if operation == 'gpu_spmm':
            score += 0.3

    elif fmt == SparseFormat.BITMAP:
        # Bitmap适合规则分布
        if features.is_regular:
            score += 0.3
        if features.sparsity > 0.7:
            score += 0.2

    return min(score, 1.0)


def _get_recommendation_reason(fmt: SparseFormat, features: FormatFeatures) -> str:
    """获取推荐原因"""
    reasons = {
        SparseFormat.CSR: [],
        SparseFormat.CSC: [],
        SparseFormat.COO: [],
        SparseFormat.BCSR: [],
        SparseFormat.BITMAP: []
    }

    if fmt == SparseFormat.CSR:
        reasons[fmt].append("CSR格式适合稀疏矩阵-向量乘法(SpMV)")
        if features.row_nnz_variance < NNZ_VARIANCE_LOW:
            reasons[fmt].append("行非零元素分布均匀")
        if features.bandwidth < BANDWIDTH_MEDIUM:
            reasons[fmt].append("带宽较小")

    elif fmt == SparseFormat.CSC:
        reasons[fmt].append("CSC格式适合列操作")

    elif fmt == SparseFormat.COO:
        reasons[fmt].append("COO格式适合矩阵构建和转换")
        if features.sparsity > SPARSITY_HIGH:
            reasons[fmt].append("稀疏度极高")

    elif fmt == SparseFormat.BCSR:
        reasons[fmt].append("BCSR格式适合GPU加速")
        if features.block_structure_score > BLOCK_STRUCTURE_MEDIUM:
            reasons[fmt].append("具有明显的块结构")

    elif fmt == SparseFormat.BITMAP:
        reasons[fmt].append("Bitmap格式适合规则稀疏模式")

    return "; ".join(reasons[fmt]) if reasons[fmt] else "默认推荐"
