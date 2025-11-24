# 家庭能源管理系统算法详解

本文档详细介绍了 `demo.py` 中实现的三种能源管理算法：**线性规划（Linear Programming）**、**基于规则的方法（Rule-based）** 和 **动态规划（Dynamic Programming）**。重点解析线性规划和动态规划的核心原理与实现细节。

## 目录
1. [系统概述](#系统概述)
2. [线性规划算法详解](#线性规划算法详解)
3. [动态规划算法详解](#动态规划算法详解)
4. [算法对比分析](#算法对比分析)
5. [代码示例与使用](#代码示例与使用)

---

## 系统概述

本系统是一个**三态电池能源管理系统**，旨在优化家庭光伏储能系统的运行策略。系统核心组件包括：

- **光伏发电 (PV)**: 清洁能源输入
- **储能电池 (Battery)**: 具有充电、放电、待机三种状态
- **家庭负载 (Load)**: 用电需求
- **电网 (Grid)**: 双向电力交易

### 系统约束条件
- 电池SOC范围：10%-90%
- 电池功率限制：最大充/放电功率
- 三态互斥：电池同一时刻只能处于一种状态
- 功率平衡：供需必须平衡

---

## 线性规划算法详解

### 核心思想
线性规划通过构建**凸优化问题**，在满足所有约束条件下寻找全局最优解。该方法将能源管理问题转化为数学优化问题，能够保证解的最优性。

### 数学建模

#### 决策变量 (87-112行)
```python
# 功率流变量 - 描述能量在各组件间的流动
P_pv_load = cp.Variable(T, nonneg=True)      # 光伏直接供负载
P_pv_bat = cp.Variable(T, nonneg=True)       # 光伏给电池充电
P_pv_grid = cp.Variable(T, nonneg=True)      # 光伏售电给电网
P_grid_load = cp.Variable(T, nonneg=True)    # 电网直接供负载
P_grid_bat = cp.Variable(T, nonneg=True)     # 电网给电池充电
P_bat_load = cp.Variable(T, nonneg=True)     # 电池供负载
P_bat_grid = cp.Variable(T, nonneg=True)     # 电池售电给电网

# 电池状态变量
P_bat_ch = cp.Variable(T, nonneg=True)       # 电池总充电功率
P_bat_dis = cp.Variable(T, nonneg=True)      # 电池总放电功率
Soc = cp.Variable(T+1, nonneg=True)          # SOC轨迹

# 三态二进制变量 - 确保电池状态互斥
x_charge = cp.Variable(T, boolean=True)      # 充电状态
x_discharge = cp.Variable(T, boolean=True)   # 放电状态
x_idle = cp.Variable(T, boolean=True)        # 待机状态
```

#### 约束条件详解

**1. 电池三态互斥约束 (116-118行)**
```python
# 关键约束：确保每个时刻电池只能处于一种状态
for t in range(T):
    constraints.append(x_charge[t] + x_discharge[t] + x_idle[t] == 1)
```

**2. 电池功率约束 (120-132行)**
```python
for t in range(T):
    # 充电状态约束：只有在充电状态才允许充电功率
    constraints.append(P_bat_ch[t] <= self.P_bat_max * x_charge[t])
    
    # 放电状态约束：只有在放电状态才允许放电功率
    constraints.append(P_bat_dis[t] <= self.P_bat_max * x_discharge[t])
    
    # 待机状态约束：待机时充放电功率都为0
    constraints.append(P_bat_ch[t] <= self.P_bat_max * (1 - x_idle[t]))
    constraints.append(P_bat_dis[t] <= self.P_bat_max * (1 - x_idle[t]))
```

**3. 功率平衡约束 (134-158行)**
```python
for t in range(T):
    # 光伏功率分配平衡
    constraints.append(
        P_pv_load[t] + P_pv_bat[t] + P_pv_grid[t] == pv_generation[t]
    )
    
    # 负载功率平衡 - 所有供电源之和等于负载需求
    constraints.append(
        P_pv_load[t] + P_bat_load[t] + P_grid_load[t] == load_demand[t]
    )
    
    # 电池充电功率平衡
    constraints.append(P_pv_bat[t] + P_grid_bat[t] == P_bat_ch[t])
    
    # 电池放电功率分配
    constraints.append(P_bat_load[t] + P_bat_grid[t] == P_bat_dis[t])
```

**4. SOC动态约束 (160-178行)**
```python
# SOC状态转移方程 - 描述电池电量随时间的变化
for t in range(T):
    constraints.append(
        Soc[t+1] == Soc[t] + 
        (P_bat_ch[t] * self.eta_c - P_bat_dis[t] / self.eta_d) / self.E_bat
    )
    constraints.append(Soc[t+1] >= 0.0)
    constraints.append(Soc[t+1] <= 1.0)
```

#### 目标函数设计 (181-222行)

线性规划的目标函数包含多个优化目标：

```python
# 基本经济成本
money_spend = cp.sum(cp.multiply(P_purchase, buy_prices)) / 1000
money_earn = cp.sum(cp.multiply(P_sell, sell_prices)) / 1000

# 储能价值激励 - 防止短视行为
storage_incentive = 0
for t in range(T):
    future_prices = sell_prices[t:] + buy_prices[t:]
    if future_prices:
        future_max_price = max(future_prices)
        price_spread = max(0, future_max_price - sell_prices[t])
        storage_value_per_kwh = price_spread * self.eta_c * self.eta_d * 1.2
        storage_incentive += storage_value_per_kwh * P_pv_bat[t] / 1000

# SOC违反惩罚项 - 软约束处理
soc_penalty = 0
for t in range(T+1):
    soc_min_violation = cp.maximum(0, self.Soc_min - Soc[t])
    soc_max_violation = cp.maximum(0, Soc[t] - self.Soc_max)
    soc_penalty += penalty_weight * (soc_min_violation + soc_max_violation)

# 总目标函数
total_cost = money_spend - money_earn - storage_incentive + soc_penalty
```

### 线性规划的优势
1. **全局最优性**: 保证找到全局最优解
2. **高效求解**: 成熟的求解器支持（GUROBI, MOSEK等）
3. **约束处理**: 能够精确处理复杂约束
4. **可解释性**: 结果具有明确的数学意义

---

## 动态规划算法详解

### 核心思想
动态规划通过**状态分解**和**最优子结构**原理，将复杂的多时段优化问题分解为逐时段的决策问题。每个状态记录"从当前状态到终点的最小成本"。

### 算法实现详解

#### 状态空间设计 (894-914行)
```python
def dynamic_programming(self, pv_generation, load_demand, buy_prices, sell_prices, initial_soc=0.5):
    T = len(pv_generation)
    
    # SOC离散化 - 将连续状态空间离散化
    soc_resolution = 100  # SOC被分成100个离散点
    soc_range_min = min(self.Soc_min, initial_soc)
    soc_range_max = self.Soc_max
    soc_states = np.linspace(soc_range_min, soc_range_max, soc_resolution)
    
    # DP表定义：dp_cost[t][soc_idx] = 从时刻t、SOC状态soc_idx到终点的最小成本
    dp_cost = np.full((T+1, soc_resolution), np.inf)
    dp_decisions = {}  # 存储最优决策路径
    
    # 初始化：找到最接近initial_soc的离散状态
    init_soc_idx = np.argmin(np.abs(soc_states - initial_soc))
    dp_cost[0][init_soc_idx] = 0
```

#### 动态规划状态转移 (916-965行)

动态规划的核心是**状态转移方程**：
```
V(t, soc) = min_{action} [cost(t, soc, action) + V(t+1, next_soc)]
```

```python
# 动态规划主循环 - 逐时段向前递推
for t in range(T):
    pv_t = pv_generation[t]
    load_t = load_demand[t]
    buy_price_t = buy_prices[t]
    sell_price_t = sell_prices[t]
    
    for soc_idx, current_soc in enumerate(soc_states):
        if dp_cost[t][soc_idx] == np.inf:
            continue  # 当前状态不可达
        
        # 计算当前SOC下的可用功率约束
        max_charge_power = min(self.P_bat_max, 
                             (self.Soc_max - current_soc) * self.E_bat / self.eta_c)
        max_discharge_power = min(self.P_bat_max, 
                                (current_soc - self.Soc_min) * self.E_bat * self.eta_d)
        
        # 生成所有可能的电池动作
        battery_actions = self._generate_battery_actions(
            pv_t, load_t, max_charge_power, max_discharge_power)
        
        # 评估每个动作，更新DP表
        for action in battery_actions:
            next_soc, cost, valid = self._evaluate_action(
                current_soc, action, pv_t, load_t, buy_price_t, sell_price_t)
            
            if not valid:
                continue
            
            next_soc_idx = np.argmin(np.abs(soc_states - next_soc))
            new_cost = dp_cost[t][soc_idx] + cost
            
            # Bellman方程更新
            if new_cost < dp_cost[t+1][next_soc_idx]:
                dp_cost[t+1][next_soc_idx] = new_cost
                dp_decisions[(t+1, next_soc_idx)] = {
                    'prev_soc_idx': soc_idx,
                    'action': action
                }
```

#### 动作空间生成 (978-1073行)

动态规划的关键在于**智能动作空间设计**：

```python
def _generate_battery_actions(self, pv_t, load_t, max_charge_power, max_discharge_power):
    actions = []
    
    # 1. 待机状态动作
    idle_alloc = self._allocate_power_for_idle(pv_t, load_t)
    if idle_alloc:
        actions.append(('idle', 0, idle_alloc))
    
    # 2. 充电状态动作 - 智能功率点选择
    if max_charge_power > 1:
        charge_powers = []
        
        # 关键充电功率点
        if pv_t > load_t:
            pv_surplus = pv_t - load_t
            charge_powers.append(min(pv_surplus, max_charge_power))  # 光伏剩余充电
        
        charge_powers.append(max_charge_power)  # 最大充电功率
        
        # 电网充电选项
        if self.grid_charge_max > 0:
            grid_charge_powers = [
                min(self.grid_charge_max * 0.25, max_charge_power),
                min(self.grid_charge_max * 0.5, max_charge_power),
                min(self.grid_charge_max * 0.75, max_charge_power),
                min(self.grid_charge_max, max_charge_power)
            ]
            charge_powers.extend(grid_charge_powers)
        
        # 去重并生成动作
        charge_powers = sorted(list(set([p for p in charge_powers if p >= 1])))
        for charge_power in charge_powers:
            power_alloc = self._allocate_power_for_charging(pv_t, load_t, charge_power)
            if power_alloc:
                actions.append(('charge', charge_power, power_alloc))
    
    # 3. 放电状态动作（类似逻辑）
    # ...
    
    return actions
```

#### 成本评估函数 (1178-1253行)

每个动作的成本评估考虑多个因素：

```python
def _evaluate_action(self, current_soc, bat_state, bat_power, power_alloc, 
                    pv_t, load_t, buy_price_t, sell_price_t):
    # 1. 计算SOC状态转移
    if bat_state == 'charge':
        energy_change = bat_power * self.eta_c / self.E_bat
    elif bat_state == 'discharge':
        energy_change = -bat_power / (self.eta_d * self.E_bat)
    else:  # idle
        energy_change = 0
    
    next_soc = current_soc + energy_change
    
    # 2. 严格检查SOC约束
    if current_soc < self.Soc_min and energy_change < -1e-6:
        return next_soc, float('inf'), False  # 禁止进一步放电
    
    if next_soc < self.Soc_min - 1e-6 or next_soc > self.Soc_max + 1e-6:
        return next_soc, float('inf'), False  # SOC越界
    
    # 3. 计算基本经济成本
    purchase_cost = (power_alloc['P_grid_load'] + power_alloc['P_grid_bat']) * buy_price_t / 1000
    sell_revenue = (power_alloc['P_pv_grid'] + power_alloc['P_bat_grid']) * sell_price_t / 1000
    base_cost = purchase_cost - sell_revenue
    
    # 4. 添加SOC约束违反惩罚
    penalty_cost = 0
    if next_soc < self.Soc_min:
        soc_violation = self.Soc_min - next_soc
        penalty_cost = soc_violation * 10  # 每1%违反增加10元成本
    elif next_soc > self.Soc_max:
        soc_violation = next_soc - self.Soc_max
        penalty_cost = soc_violation * 10
    
    # 5. 储能价值激励
    storage_incentive = 0
    if bat_state == 'charge' and next_soc < 0.7:
        max_future_sell_price = 0.53
        current_charge_cost = buy_price_t if power_alloc['P_grid_bat'] > 0 else 0
        if max_future_sell_price > current_charge_cost:
            storage_value_per_kwh = (max_future_sell_price - current_charge_cost) * self.eta_c * self.eta_d
            stored_energy_kwh = bat_power / 1000
            storage_incentive = -storage_value_per_kwh * stored_energy_kwh * 0.3
    
    total_cost = base_cost + penalty_cost + storage_incentive
    return next_soc, total_cost, True
```

#### 最优解回溯 (1255-1278行)

```python
def _backtrack_solution(self, dp_decisions, T, final_soc_idx, soc_states):
    """从终点回溯找到完整的最优策略路径"""
    path = []
    current_t = T
    current_soc_idx = final_soc_idx
    
    # 从终点向起点回溯
    while current_t > 0:
        if (current_t, current_soc_idx) not in dp_decisions:
            break
            
        decision = dp_decisions[(current_t, current_soc_idx)]
        path.append({
            'time': current_t - 1,
            'prev_soc': soc_states[decision['prev_soc_idx']],
            'current_soc': soc_states[current_soc_idx],
            'action': decision['action'],
            'power_allocation': decision['power_allocation']
        })
        
        current_soc_idx = decision['prev_soc_idx']
        current_t -= 1
    
    path.reverse()  # 反转得到正向路径
    return path
```

### 动态规划的优势与挑战

**优势：**
1. **最优子结构**: 保证局部最优决策组成全局最优
2. **灵活性**: 可处理非线性成本和复杂约束
3. **可扩展性**: 容易添加新的状态变量和约束

**挑战：**
1. **维数灾难**: 状态空间随变量数量指数增长
2. **离散化误差**: 连续状态离散化带来精度损失
3. **计算复杂度**: O(T × S × A)，其中T是时间步数，S是状态数，A是动作数

---

## 算法对比分析

| 特征 | 线性规划 | 动态规划 | 基于规则 |
|------|----------|----------|----------|
| **最优性** | 全局最优 | 全局最优（离散化误差） | 次优 |
| **计算复杂度** | 多项式时间 | 指数时间（状态数） | 线性时间 |
| **约束处理** | 精确 | 灵活 | 简单 |
| **可解释性** | 数学最优 | 策略明确 | 直观易懂 |
| **实现难度** | 中等 | 高 | 低 |
| **扩展性** | 好 | 中等 | 高 |

### 适用场景

**线性规划适用于：**
- 追求最优经济效益
- 约束条件复杂但线性
- 有成熟求解器支持

**动态规划适用于：**
- 非线性成本函数
- 复杂的状态转移逻辑
- 需要明确决策策略

**基于规则适用于：**
- 快速部署和调试
- 用户可理解的策略
- 实时响应要求高

---

## 代码示例与使用

### 基本使用示例

```python
# 初始化能源管理系统
ems = ThreeStateEnergyManagementSystem(
    battery_capacity=10000,    # 电池容量 10kWh
    max_power=3000,           # 最大功率 3kW
    charge_efficiency=0.9,    # 充电效率
    discharge_efficiency=0.9, # 放电效率
    soc_min=0.1,             # 最小SOC
    soc_max=0.9              # 最大SOC
)

# 线性规划优化
lp_result = ems.optimize_complete_model(
    pv_generation,    # 24小时光伏发电预测
    load_demand,      # 24小时负载需求预测
    buy_prices,       # 24小时购电价格
    sell_prices,      # 24小时售电价格
    initial_soc=0.5   # 初始SOC
)

# 动态规划优化
dp_result = ems.dynamic_programming(
    pv_generation,
    load_demand,
    buy_prices,
    sell_prices,
    initial_soc=0.5
)

# 结果分析
df_lp = ems.create_hourly_analysis(lp_result, pv_generation, load_demand, buy_prices, sell_prices)
df_dp = ems.create_hourly_analysis(dp_result, pv_generation, load_demand, buy_prices, sell_prices)

print(f"线性规划总成本: {lp_result['total_cost']:.3f} 元")
print(f"动态规划总成本: {dp_result['total_cost']:.3f} 元")
```

### 高级配置

```python
# 自定义约束参数
ems.grid_charge_max = 2000    # 允许电网充电最大功率
ems.grid_discharge_max = 2000 # 允许电池售电最大功率

# 可视化分析
ems.plot_battery_three_states(df_lp)           # 电池三态分析
ems.plot_comprehensive_analysis(df_lp, lp_result)  # 综合分析
ems.plot_energy_flow_sankey(df_lp)             # 能量流向桑基图
```

---

## 总结

本系统通过**线性规划**和**动态规划**两种优化算法，为家庭能源管理提供了科学、高效的解决方案。线性规划保证了数学意义上的全局最优，而动态规划则提供了更灵活的策略决策框架。两种算法各有优势，可根据具体应用需求选择使用。

核心技术亮点：
- **三态电池建模**: 精确描述电池充电、放电、待机状态
- **多目标优化**: 兼顾经济效益和储能价值
- **约束处理**: 严格满足SOC和功率限制
- **状态空间优化**: 智能动作空间设计减少计算复杂度

该系统为智能电网和分布式能源管理提供了重要的理论基础和实践指导。