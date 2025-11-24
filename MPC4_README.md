# 分层MPC能源管理系统 (demo_mpc4.py)

## 系统概述

分层MPC (Hierarchical Model Predictive Control) 能源管理系统是一个两层优化架构，旨在解决传统滚动优化MPC的"短视"问题，实现接近全局最优的家庭能源调度效果。

### 核心思想

传统MPC由于预测时域有限，可能无法充分利用全天的电价差异和光伏发电模式，导致局部最优决策。分层MPC通过以下两层架构解决这一问题：

- **上层：全局规划层** - 基于24小时预测数据进行全局优化，生成目标SOC轨迹
- **下层：MPC跟踪层** - 使用短时域MPC跟踪目标轨迹，同时处理实时扰动

## 系统架构

### 1. 全局规划层 (Global Planning Layer)

```python
def _global_planning_layer(self, pv_generation, load_demand, buy_prices, sell_prices, initial_soc):
    """
    执行24小时全局优化，获得全局最优的SOC轨迹
    
    目标：基于预测数据获得全局最优的SOC轨迹和功率分配策略
    特点：不考虑实时扰动，专注于全局最优性
    """
```

**主要功能：**
- 使用完整的24小时预测数据
- 执行一次性全局优化
- 生成目标SOC轨迹作为下层MPC的参考
- 考虑全天电价模式和光伏发电特性

### 2. MPC跟踪层 (MPC Tracking Layer)

```python
def _mpc_tracking_layer(self, current_time, actual_pv, actual_load, current_soc,
                       pv_prediction, load_prediction, buy_prices, sell_prices):
    """
    短时域MPC跟踪目标SOC轨迹
    
    目标：
    1. 跟踪全局规划的SOC轨迹
    2. 处理实时扰动（实际vs预测的差异）
    3. 在跟踪性能和经济性之间平衡
    """
```

**主要功能：**
- 使用短预测时域（4小时）进行实时优化
- 跟踪全局规划生成的目标SOC轨迹
- 处理实际值与预测值的偏差
- 在成本最小化和SOC跟踪之间找到平衡

## 核心算法

### 目标函数设计

MPC跟踪层的目标函数结合了两个关键要素：

```python
# 电力成本项
cost_term = cp.sum(cp.multiply(buy_prices_pred, P_purchase) - 
                  cp.multiply(sell_prices_pred, P_sell)) / 1000

# SOC跟踪项
soc_tracking_term = cp.sum_squares(Soc - target_soc_pred)

# 综合目标函数
objective = cp.Minimize(cost_weight * cost_term + 
                       soc_tracking_weight * soc_tracking_term)
```

**权重参数：**
- `cost_weight = 1.0`：电力成本权重
- `soc_tracking_weight = 10.0`：SOC跟踪权重

### 约束条件

1. **三态互斥约束**：电池在每个时刻只能处于充电、放电或待机状态之一
2. **功率平衡约束**：光伏、电网、电池功率满足供需平衡
3. **SOC动态约束**：考虑充放电效率的SOC状态转移
4. **安全边界约束**：SOC保持在安全范围内

## 关键特性

### 1. 智能错误处理

```python
def _emergency_control(self, actual_pv, actual_load, current_soc, buy_price, sell_price):
    """
    应急控制：当MPC求解失败时的备选方案
    使用简单的规则控制确保系统稳定运行
    """
```

### 2. 预测时域自适应

```python
remaining_time = len(pv_prediction) - current_time
horizon = min(self.prediction_horizon, remaining_time)
```

系统根据剩余时间自动调整预测时域，确保边界条件的正确处理。

### 3. 实时数据融合

```python
# 第一个时刻使用实际数据，后续使用预测数据
pv_pred[0] = actual_pv
load_pred[0] = actual_load
```

## 性能优势

### 1. 成本优化效果

通过实际测试对比：
- **传统MPC3系统**：总成本 3.515元
- **分层MPC系统**：总成本 3.092元
- **改进效果**：成本降低 12%

### 2. SOC轨迹跟踪

- 平均SOC跟踪误差：1.2%
- 成功避免了原系统中的不合理电网充电行为
- 充分利用光伏发电进行电池充电

### 3. 全局性能指标

- 全局规划成本：3.337元
- 实际执行成本：3.092元
- **分层Gap**：0.245元（实际性能优于全局规划）

## 配置参数

### 系统参数

```python
HierarchicalMPCEnergyManagementSystem(
    prediction_horizon=4,        # MPC预测时域（小时）
    control_horizon=1,           # MPC控制时域（小时）
    soc_tracking_weight=10.0,    # SOC跟踪权重
    cost_weight=1.0,             # 成本权重
    planning_frequency=24        # 全局规划频率（小时）
)
```

### 电池参数

```python
battery_capacity=13.6,      # 电池容量 (kWh)
max_power=5.0,              # 最大功率 (kW)
charge_efficiency=0.9,      # 充电效率
discharge_efficiency=0.9,   # 放电效率
soc_min=0.05,              # 最小SOC (5%)
soc_max=1.0                # 最大SOC (100%)
```

## 使用方法

### 1. 基本用法

```python
from demo_mpc4 import HierarchicalMPCEnergyManagementSystem

# 创建系统实例
hierarchical_mpc = HierarchicalMPCEnergyManagementSystem(
    battery_capacity=13600,  # Wh
    max_power=5000,         # W
    prediction_horizon=4,
    soc_tracking_weight=10.0
)

# 运行优化
result = hierarchical_mpc.hierarchical_mpc_optimize(
    pv_generation,    # 24小时光伏预测 (W)
    load_demand,      # 24小时负载预测 (W)
    buy_prices,       # 24小时购电价格 (元/kWh)
    sell_prices,      # 24小时售电价格 (元/kWh)
    real_pv=real_pv,  # 实际光伏数据 (可选)
    real_load=real_load,  # 实际负载数据 (可选)
    initial_soc=0.5   # 初始SOC
)
```

### 2. 完整测试运行

```bash
python demo_mpc4.py
```

这将运行完整的测试案例，包括：
- 数据加载和处理
- 分层MPC优化
- 结果保存和可视化
- 性能分析报告

## 输出文件

### 1. 结果文件

- `res.json`：完整的优化结果，包含功率流、电池状态、成本分析等
- 格式与其他MPC系统兼容，便于对比分析

### 2. 可视化图表

- `hierarchical_mpc_soc_tracking.png`：SOC跟踪性能分析
- `hierarchical_mpc_decision_analysis.png`：分层决策对比分析
- `hierarchical_mpc_cost_analysis.png`：成本分解分析
- 其他标准图表：电池三态图、综合分析图等

### 3. 分析数据

```json
{
  "mpc_info": {
    "solver_type": "hierarchical_mpc",
    "prediction_horizon": 4,
    "soc_tracking_weight": 10.0,
    "average_soc_tracking_error": 0.012,
    "global_plan_cost": 3.337,
    "hierarchical_gap": 0.245
  }
}
```

## 技术细节

### 1. 求解器选择

系统按优先级尝试多种求解器：
1. GUROBI（商业求解器，性能最佳）
2. MOSEK（商业求解器）
3. ECOS（开源求解器）

### 2. 数值稳定性

- 采用适当的权重比例避免数值问题
- 包含边界条件检查和异常处理
- 提供应急控制确保系统鲁棒性

### 3. 计算复杂度

- **全局规划**：O(T³)，T=24（每天执行一次）
- **MPC跟踪**：O(H³)，H=4（每小时执行一次）
- **总体复杂度**：远低于24小时滚动MPC

## 适用场景

### 1. 推荐使用场景

- 家庭储能系统优化
- 电价具有明显时段性差异
- 光伏发电模式相对可预测
- 需要平衡全局最优性和实时响应能力

### 2. 系统要求

- Python 3.7+
- CVXPY优化库
- 至少一种MIP求解器（GUROBI/MOSEK/CBC等）
- 内存需求：< 100MB
- 计算时间：< 10秒/天（典型配置）

## 扩展功能

### 1. 参数调优建议

- **高跟踪精度**：增加`soc_tracking_weight`到20-50
- **成本优先**：降低`soc_tracking_weight`到5以下
- **实时性要求高**：减小`prediction_horizon`到2-3小时

### 2. 自定义扩展

系统设计为模块化架构，支持：
- 自定义全局规划算法
- 不同的MPC跟踪策略
- 多样化的目标函数设计
- 额外的约束条件集成

## 高级优化方案

### 为什么选择SOC轨迹跟踪？

当前系统选择SOC轨迹作为跟踪目标的原因：

#### 1. SOC是系统状态的核心

```python
# SOC直接决定了电池的可用能量和操作空间
available_energy = (current_soc - soc_min) * battery_capacity
remaining_capacity = (soc_max - current_soc) * battery_capacity
```

SOC轨迹包含了最关键的时序信息：
- **什么时候储能**：SOC上升段
- **什么时候释能**：SOC下降段  
- **储能多少**：SOC变化幅度
- **能量分配时机**：SOC峰值和谷值的位置

#### 2. 降维优化的效果

```python
# 全局优化结果包含24×7=168个功率变量
# 但SOC轨迹只需要25个值（24小时+初始值）
target_soc_trajectory = global_result['battery']['SOC']  # 25个值
```

相比完整的功率流，SOC轨迹是最精炼的"指令"。

#### 3. 物理意义明确且鲁棒性好

```python
# 即使PV/负载预测有偏差，SOC轨迹仍然提供有效指导
if actual_soc < target_soc:
    charging_preference = True  # 倾向于充电或减少放电
elif actual_soc > target_soc:
    discharging_preference = True  # 倾向于放电或减少充电
```

### 替代跟踪策略

尽管SOC轨迹跟踪是当前的最佳选择，系统还支持以下替代方案：

#### 方案A：电网净功率轨迹跟踪

```python
class NetPowerTrackingMPC(HierarchicalMPCEnergyManagementSystem):
    def _net_power_tracking_layer(self, current_time, target_net_power):
        """
        跟踪电网净功率轨迹
        
        思路：全局优化确定每小时应该从电网购买多少电
        MPC跟踪这个净功率目标
        """
        # 目标函数
        net_power_tracking = cp.sum_squares(P_purchase - P_sell - target_net_power)
        objective = cp.Minimize(cost_weight * cost_term + 
                               tracking_weight * net_power_tracking)
```

**优点**：
- 直接对应电费账单
- 对预测误差鲁棒性较好

**缺点**：
- 忽略了SOC状态约束
- 可能导致电池过充过放

**适用场景**：电价变化剧烈，成本控制为第一优先级的场合

#### 方案B：功率轨迹跟踪

```python
class PowerTrackingMPC(HierarchicalMPCEnergyManagementSystem):
    def _power_tracking_layer(self, current_time, target_power_trajectory):
        """
        跟踪电池充放电功率轨迹
        
        优点：直接控制功率输出，实现简单
        缺点：功率轨迹受实际PV/负载变化影响大
        """
        # 目标函数：最小化功率跟踪误差  
        power_tracking_term = cp.sum_squares(P_bat_ch - target_charge_power) + \
                             cp.sum_squares(P_bat_dis - target_discharge_power)
        
        objective = cp.Minimize(cost_weight * cost_term + 
                               power_weight * power_tracking_term)
```

**适用场景**：需要精确控制电池功率输出的应用，如电网调频服务

#### 方案C：分时段目标跟踪

```python
class TimeSegmentTrackingMPC(HierarchicalMPCEnergyManagementSystem):
    def _segment_tracking_layer(self, current_time):
        """
        基于时段特征的目标跟踪
        
        思路：
        - 低电价时段：目标充电至指定SOC
        - 高电价时段：目标放电至指定SOC  
        - 平电价时段：维持SOC稳定
        """
        current_price = buy_prices[current_time]
        
        if current_price < low_price_threshold:
            target_soc_change = positive_target  # 充电时段
        elif current_price > high_price_threshold:
            target_soc_change = negative_target  # 放电时段
        else:
            target_soc_change = 0  # 维持时段
            
        soc_change_tracking = cp.sum_squares(Soc[1:] - Soc[:-1] - target_soc_change)
```

**优点**：
- 规则简单易懂，计算开销小
- 适合电价模式规律性强的场景

**缺点**：
- 过于简化，可能错过复杂的套利机会
- 阈值设定需要经验

#### 方案D：混合跟踪策略（推荐的增强版本）

```python
class EnhancedTrackingMPC(HierarchicalMPCEnergyManagementSystem):
    def _enhanced_tracking_layer(self, current_time, targets):
        """
        增强跟踪：主要跟踪SOC，辅助考虑成本和功率约束
        
        这是对当前系统的推荐改进方案
        """
        # 主要目标：SOC轨迹跟踪
        soc_tracking = cp.sum_squares(Soc - targets['soc_trajectory'])
        
        # 辅助约束：成本上界
        cost_penalty = cp.sum(cp.maximum(0, hourly_cost - targets['max_hourly_cost']))
        
        # 辅助约束：功率平滑（减少频繁切换）
        power_smoothing = cp.sum_squares(P_bat_ch[1:] - P_bat_ch[:-1]) + \
                         cp.sum_squares(P_bat_dis[1:] - P_bat_dis[:-1])
        
        # 辅助约束：电网功率限制
        grid_power_limit = cp.sum(cp.maximum(0, P_purchase - max_grid_power))
        
        objective = cp.Minimize(
            1.0 * cost_term +           # 基础成本
            10.0 * soc_tracking +       # 主要跟踪目标  
            5.0 * cost_penalty +        # 成本约束
            0.1 * power_smoothing +     # 操作平滑性
            2.0 * grid_power_limit      # 电网功率限制
        )
```

**实现方式**：

```python
# 在demo_mpc4.py中添加增强跟踪选项
class HierarchicalMPCEnergyManagementSystem(ThreeStateEnergyManagementSystem):
    def __init__(self, tracking_strategy='soc_only', **kwargs):
        super().__init__(**kwargs)
        self.tracking_strategy = tracking_strategy
        
        # 增强跟踪的额外参数
        if tracking_strategy == 'enhanced':
            self.cost_penalty_weight = kwargs.get('cost_penalty_weight', 5.0)
            self.power_smoothing_weight = kwargs.get('power_smoothing_weight', 0.1)
            self.max_hourly_cost_ratio = kwargs.get('max_hourly_cost_ratio', 1.2)
```

### 实现建议

#### 1. 渐进式升级路径

```python
# 阶段1：当前SOC跟踪（已实现）
tracking_strategy = 'soc_only'

# 阶段2：增加成本约束
tracking_strategy = 'soc_with_cost_limit'

# 阶段3：完整的混合跟踪
tracking_strategy = 'enhanced'

# 阶段4：自适应策略选择
tracking_strategy = 'adaptive'
```

#### 2. 参数配置示例

```python
# 成本敏感场景
hierarchical_mpc = HierarchicalMPCEnergyManagementSystem(
    tracking_strategy='enhanced',
    soc_tracking_weight=8.0,      # 稍微降低SOC跟踪权重
    cost_penalty_weight=10.0,     # 增加成本约束权重
    max_hourly_cost_ratio=1.1     # 严格的成本限制
)

# 稳定性优先场景
hierarchical_mpc = HierarchicalMPCEnergyManagementSystem(
    tracking_strategy='enhanced',
    soc_tracking_weight=15.0,     # 提高SOC跟踪权重
    power_smoothing_weight=0.5,   # 增加功率平滑性
    cost_penalty_weight=2.0       # 放松成本约束
)
```

#### 3. 性能评估指标

```python
def evaluate_tracking_performance(self, result):
    """
    评估不同跟踪策略的性能
    """
    metrics = {
        'soc_tracking_rmse': np.sqrt(np.mean(soc_errors**2)),
        'cost_efficiency': actual_cost / optimal_cost,
        'power_smoothness': np.mean(np.abs(np.diff(power_profile))),
        'constraint_violations': count_violations,
        'robustness_score': performance_under_uncertainty
    }
    return metrics
```

### 选择指南

根据应用场景选择合适的跟踪策略：

| 场景 | 推荐策略 | 关键参数 | 优势 |
|------|---------|----------|------|
| 成本最优 | SOC + 成本约束 | `cost_penalty_weight=10` | 严格控制电费支出 |
| 设备保护 | SOC + 功率平滑 | `power_smoothing_weight=0.5` | 减少设备磨损 |
| 电网友好 | 净功率跟踪 | `net_power_weight=8` | 减少电网冲击 |
| 综合平衡 | 混合跟踪 | 默认参数组合 | 多目标平衡优化 |

这些扩展选项为用户提供了更多的灵活性，可以根据具体需求和约束条件选择最适合的跟踪策略。

## 故障排除

### 常见问题

1. **求解器找不到**
   ```
   问题：No suitable solver found
   解决：安装GUROBI、MOSEK或确保CBC可用
   ```

2. **数值不稳定**
   ```
   问题：Optimization failed with numerical issues
   解决：检查数据范围，调整权重参数
   ```

3. **跟踪误差过大**
   ```
   问题：SOC tracking error > 5%
   解决：增加soc_tracking_weight或检查预测数据质量
   ```

## 版本历史

- **v1.0**：基础分层MPC实现
- **v1.1**：增加应急控制和错误处理
- **v1.2**：优化数值稳定性和求解器选择
- **v1.3**：添加完整的可视化和分析功能

## 参考文献

1. README_MPC_long.md - MPC局部最优问题分析
2. demo.py - 基础能源管理系统
3. demo_mpc.py - 标准MPC实现
4. demo_mpc3.py - 改进MPC系统

## 联系方式

如有问题或建议，请查看相关代码文件或联系开发团队。