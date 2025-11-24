from functools import partial
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Arrow
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import ConnectionPatch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import json
import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from utils import *

# 设置中文字体和全局样式
#plt.rcParams['font.sans-serif'] = ['Heiti TC', 'DejaVu Sans']
font_path = 'SourceHanSansSC-Normal.otf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.dpi'] = 300

# JSON序列化辅助函数
def convert_to_json_serializable(obj):
    """将numpy数组和其他不可序列化的对象转换为可JSON序列化的格式"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

# 定义统一的颜色主题
COLOR_THEME = {
    'pv': '#FF8C00',        # 橙色 - 光伏
    'battery': '#32CD32',   # 绿色 - 电池
    'load': '#4169E1',      # 蓝色 - 负载  
    'grid': '#696969',      # 灰色 - 电网
    'charge': '#90EE90',    # 浅绿色 - 充电
    'discharge': '#FF6347', # 红色 - 放电
    'idle': '#D3D3D3',     # 浅灰色 - 待机
    'profit': '#00CED1',    # 青色 - 收益
    'loss': '#DC143C'       # 深红色 - 损失
}

class ThreeStateEnergyManagementSystem:
    def __init__(
        self, 
        battery_capacity=10000, 
        max_power=3000, 
        charge_efficiency=0.9, 
        discharge_efficiency=0.9, 
        grid_charge_max=0, 
        grid_discharge_max=0, 
        soc_min=0.1, 
        soc_max=0.9,
        grid_max=1e7):

        self.E_bat = battery_capacity
        self.P_bat_max = max_power  # 电池充放电最大功率
        self.eta_c = charge_efficiency
        self.eta_d = discharge_efficiency
        self.Soc_min = soc_min
        self.Soc_max = soc_max
        self.grid_charge_max = grid_charge_max  #电网给电池充电最大功率， 0表示不允许, pcs约束
        self.grid_discharge_max = grid_discharge_max  #电池给电网放电最大功率，0表示不允许，pcs约束
        self.grid_max = grid_max # 电网的约束限制，可以给个很大值
        
    def optimize_complete_model(
        self, 
        pv_generation, 
        load_demand, 
        buy_prices, 
        sell_prices, 
        initial_soc=0.5):

        T = len(pv_generation)
        
        # === 变量定义 ===
        # 功率流变量
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
        Bat_remain = cp.Variable(T+1, nonneg=True)   # 电池电量
        
        # === 三态二进制变量 ===
        x_charge = cp.Variable(T, boolean=True)      # 充电状态
        x_discharge = cp.Variable(T, boolean=True)   # 放电状态
        x_idle = cp.Variable(T, boolean=True)        # 待机状态

        # === 电网三态二进制变量 ===
        g_charge = cp.Variable(T, boolean=True)    # 电网接收能量（售电给电网）
        g_discharge = cp.Variable(T, boolean=True) # 电网释放能量（向电网购电）
        g_idle = cp.Variable(T, boolean=True)      # 电网待机（能量平衡）
        
        # 辅助变量：用于线性化目标函数
        P_purchase = cp.Variable(T, nonneg=True)
        P_sell = cp.Variable(T, nonneg=True)
        
        # === 约束条件 ===
        constraints = []
        
        # 1. 电池/电网三态互斥约束
        for t in range(T):
            constraints.append(x_charge[t] + x_discharge[t] + x_idle[t] == 1)
            constraints.append(g_charge[t] + g_discharge[t] + g_idle[t] == 1)
        
        # 2. 电池功率约束
        for t in range(T):
            # 充电状态约束
            constraints.append(P_bat_ch[t] <= self.P_bat_max * x_charge[t])
            # 放电状态约束
            constraints.append(P_bat_dis[t] <= self.P_bat_max * x_discharge[t])
            # 待机状态约束（充电和放电功率都为0）
            constraints.append(P_bat_ch[t] <= self.P_bat_max * (1 - x_idle[t]))
            constraints.append(P_bat_dis[t] <= self.P_bat_max * (1 - x_idle[t]))

            # 电池-电网充放电约束
            constraints.append(P_grid_bat[t] <= self.grid_charge_max * x_charge[t])
            constraints.append(P_bat_grid[t] <= self.grid_discharge_max * x_discharge[t])

        # 电网约束，0821添加
        for t in range(T):
            # g_charge=1时，只允许售电给电网
            constraints.append(P_pv_grid[t] <= self.grid_max * g_charge[t])
            constraints.append(P_bat_grid[t] <= self.grid_max * g_charge[t])
            constraints.append(P_grid_load[t] <= self.grid_max * (1 - g_charge[t]))
            constraints.append(P_grid_bat[t] <= self.grid_max * (1 - g_charge[t]))

            # g_discharge=1时，只允许从电网购电
            constraints.append(P_grid_load[t] <= self.grid_max * g_discharge[t])
            constraints.append(P_grid_bat[t] <= self.grid_max * g_discharge[t])
            constraints.append(P_pv_grid[t] <= self.grid_max * (1 - g_discharge[t]))
            constraints.append(P_bat_grid[t] <= self.grid_max * (1 - g_discharge[t]))

            # g_idle=1时，电网交易功率都为0
            constraints.append(P_pv_grid[t] <= self.grid_max * (1 - g_idle[t]))
            constraints.append(P_bat_grid[t] <= self.grid_max * (1 - g_idle[t]))
            constraints.append(P_grid_load[t] <= self.grid_max * (1 - g_idle[t]))
            constraints.append(P_grid_bat[t] <= self.grid_max * (1 - g_idle[t]))
        

        # 3. 功率平衡约束
        for t in range(T):
            # 光伏功率分配
            constraints.append(
                P_pv_load[t] + P_pv_bat[t] + P_pv_grid[t] == pv_generation[t]
            )
            
            # 负载功率平衡  
            constraints.append(
                P_pv_load[t] + P_bat_load[t] + P_grid_load[t] == load_demand[t]
            )
            
            # 电池充电功率平衡
            constraints.append(P_pv_bat[t] + P_grid_bat[t] == P_bat_ch[t])
            
            # 电池放电功率分配
            constraints.append(P_bat_load[t] + P_bat_grid[t] == P_bat_dis[t])
            
            # 电网净功率平衡
            constraints.append(
                P_purchase[t] - P_sell[t] == 
                (P_grid_load[t] + P_grid_bat[t]) - \
                (P_pv_grid[t] + P_bat_grid[t])
            )

            # 卖电平衡
            constraints.append(P_sell[t] == P_pv_grid[t] + P_bat_grid[t])
        
        # 4. SOC动态约束
        constraints.append(Soc[0] == initial_soc)
        # 确保初始SOC也满足约束
        #constraints.append(Soc[0] >= self.Soc_min)
        #constraints.append(Soc[0] <= self.Soc_max)
        
        for t in range(T):
            constraints.append(
                Soc[t+1] == Soc[t] + 
                (P_bat_ch[t] * self.eta_c - P_bat_dis[t] / self.eta_d) / self.E_bat
            )
            # 屏蔽SOC约束，改为在目标函数中用惩罚项处理
            constraints.append(Soc[t+1] >= 0.0)
            constraints.append(Soc[t+1] <= 1.0)
        
        # 电池电量约束，0722添加 - 电池剩余电量应该与SOC保持一致
        for t in range(T+1):
            constraints.append(Bat_remain[t] == Soc[t] * self.E_bat)
            constraints.append(Bat_remain[t] >= 0)  # 确保电池剩余电量不为负


        # === 目标函数 ===
        pv_to_load = cp.sum(P_pv_load) / 1000 # 光伏到负载
        pv_to_bat = cp.sum(P_pv_bat) / 1000 # 光伏到电池
        bat_to_load = cp.sum(P_bat_load) / 1000 # 电池到负载
        money_spend = cp.sum(cp.multiply(P_purchase, buy_prices)) / 1000
        money_earn = cp.sum(cp.multiply(P_sell, sell_prices)) / 1000

        # 消除短视 - 动态储能价值
        storage_incentive = 0
        for t in range(T):
            # 计算全天剩余时间的最高电价(售电价或购电价) - 不限制窗口！
            future_prices = sell_prices[t:] + buy_prices[t:]
            if future_prices:
                future_max_price = max(future_prices)  
                # 储能价值 = (未来最高电价 - 当前售电价) × 充放电效率
                price_spread = max(0, future_max_price - sell_prices[t])
                # 提高储能激励权重，确保比售电更有吸引力
                storage_value_per_kwh = price_spread * self.eta_c * self.eta_d * 1.2  # 加20%激励
                storage_incentive += storage_value_per_kwh * P_pv_bat[t] / 1000

        # SOC违反惩罚项
        soc_penalty = 0
        penalty_weight = 100  # 惩罚权重，可以调整
        
        for t in range(T+1):
            # SOC低于最小值的惩罚
            soc_min_violation = cp.maximum(0, self.Soc_min - Soc[t])
            soc_penalty += penalty_weight * soc_min_violation
            
            # SOC高于最大值的惩罚
            soc_max_violation = cp.maximum(0, Soc[t] - self.Soc_max)
            soc_penalty += penalty_weight * soc_max_violation

        total_cost = money_spend - \
                     money_earn - \
                     1*pv_to_load - \
                     0*pv_to_bat - \
                     0*bat_to_load + \
                     soc_penalty - \
                     storage_incentive
        
        objective = cp.Minimize(total_cost)
        
        # === 求解 ===
        problem = cp.Problem(objective, constraints)
        
        # 尝试多种求解器，优先使用更精确的
        """
        solver_used = None
        try:
            problem.solve(solver=cp.GUROBI, verbose=False)
            solver_used = "GUROBI"
        except Exception as e:
            print(e)
            try:
                problem.solve(solver=cp.MOSEK, verbose=False)
                solver_used = "MOSEK"
            except:
                try:
                    problem.solve(solver=cp.CBC, verbose=False, maximumSeconds=600)
                    solver_used = "CBC"
                except:
                    try:
                        problem.solve(solver=cp.GLPK_MI, verbose=False)
                        solver_used = "GLPK_MI"
                    except:
                        problem.solve(solver=cp.ECOS_BB, verbose=False)
                        solver_used = "ECOS_BB"
        """
        #scip_params={"limits/time": 5}
        #try:
        problem.solve(solver=cp.GUROBI, verbose=False, TimeLimit=15, MIPGap=0.01)
        #except:
            #problem.solve(solver=cp.SCIP, verbose=False, scip_params={"limits/time": 15, "limits/gap": 0.01})
        # solver_used = "cuopt"
        # print(f"使用求解器: {solver_used}")
        
        # print(f"Solver status: {problem.status}")
        if problem.status == cp.OPTIMAL:
            # 计算净电网功率用于分析
            P_net_grid_value = []
            for t in range(T):
                net_power = P_purchase.value[t] - P_sell.value[t]
                P_net_grid_value.append(net_power)
            
            """
            for t in range(T):
                print(Bat_remain.value[t], P_bat_dis.value[t], Bat_remain.value[t] - P_bat_dis.value[t])
            """

            # 验证SOC约束
            """
            print("\nSOC约束验证:")
            for t in range(len(Soc.value)):
                soc_val = Soc.value[t]
                if soc_val < self.Soc_min - 1e-6:
                    print(f"⚠️  时间{t}: SOC={soc_val:.6f} < soc_min={self.Soc_min}")
                elif soc_val > self.Soc_max + 1e-6:
                    print(f"⚠️  时间{t}: SOC={soc_val:.6f} > soc_max={self.Soc_max}")
            print("SOC约束验证完成\n")
            """

            return {
                'status': 'optimal',
                'total_cost': money_spend.value - money_earn.value,
                'money_spend': money_spend.value,
                'money_earn': money_earn.value,
                'power_flows': {
                    'P_pv_load': P_pv_load.value,
                    'P_pv_bat': P_pv_bat.value, 
                    'P_pv_grid': P_pv_grid.value,
                    'P_grid_load': P_grid_load.value,
                    'P_grid_bat': P_grid_bat.value,
                    'P_bat_load': P_bat_load.value,
                    'P_bat_grid': P_bat_grid.value,
                    'P_purchase': P_purchase.value,
                    'P_sell': P_sell.value,
                    'P_net_grid': P_net_grid_value
                },
                'battery': {
                    'P_bat_ch': P_bat_ch.value,
                    'P_bat_dis': P_bat_dis.value,
                    'x_charge': x_charge.value,      # 充电状态
                    'x_discharge': x_discharge.value, # 放电状态  
                    'x_idle': x_idle.value,          # 待机状态
                    'SOC': Soc.value
                },
                'grid': {
                    'P_pv_grid': P_pv_grid.value,
                    'P_bat_grid': P_bat_grid.value,
                    'P_grid_load': P_grid_load.value,
                    'P_grid_bat': P_grid_bat.value,
                    'g_charge': g_charge.value,
                    'g_discharge': g_discharge.value,
                    'g_idle': g_idle.value
                }
            }
        else:
            return {'status': 'failed', 'message': problem.status}
    
    def rule_based(
        self, 
        code,
        solar_priority,
        load_priority,
        grid_charge,
        grid_discharge,
        pv_generation, 
        load_demand, 
        buy_prices, 
        sell_prices,
        use_optimizeed_gh=False, 
        initial_soc=0.5):
        """
        基于优先级规则的能源管理系统
        
        参数:
        - solar_priority: 光伏发电流向的优先级，可以是：
          1) 对所有小时使用相同优先级: [电网，负载，电池]，比如[1,2,0]
          2) 每小时不同优先级: 24个元素的列表，每个元素是[电网，负载，电池]的优先级
        - load_priority: 负载需求的优先级，可以是：
          1) 对所有小时使用相同优先级: [电网，光伏，电池]，比如[1,3,2] 
          2) 每小时不同优先级: 24个元素的列表，每个元素是[电网，光伏，电池]的优先级
        """
        T = len(pv_generation)
        
        # 检查优先级参数格式并规范化
        # 如果是单一优先级（适用于所有小时）
        if isinstance(solar_priority[0], (int, float)):
            solar_priorities = [solar_priority] * T  # 复制为24小时
        else:
            solar_priorities = solar_priority  # 已经是24小时的列表
            
        if isinstance(load_priority[0], (int, float)):
            load_priorities = [load_priority] * T  # 复制为24小时
        else:
            load_priorities = load_priority  # 已经是24小时的列表
        
        # 验证长度
        assert len(solar_priorities) == T, f"solar_priority长度({len(solar_priorities)})必须等于时间步数({T})"
        assert len(load_priorities) == T, f"load_priority长度({len(load_priorities)})必须等于时间步数({T})"
        
        # 初始化变量数组
        P_pv_load = np.zeros(T)
        P_pv_bat = np.zeros(T) 
        P_pv_grid = np.zeros(T)
        P_grid_load = np.zeros(T)
        P_grid_bat = np.zeros(T)
        P_bat_load = np.zeros(T)
        P_bat_grid = np.zeros(T)
        P_bat_ch = np.zeros(T)
        P_bat_dis = np.zeros(T)
        P_purchase = np.zeros(T)
        P_sell = np.zeros(T)
        
        # 电池状态
        x_charge = np.zeros(T, dtype=int)
        x_discharge = np.zeros(T, dtype=int)
        x_idle = np.zeros(T, dtype=int)
        Soc = np.zeros(T+1)
        Bat_remain = np.zeros(T+1)
        
        # 初始化SOC
        Soc[0] = initial_soc
        Bat_remain[0] = initial_soc * self.E_bat
        
        total_cost = 0
        money_spend = 0
        money_earn = 0
        
        for t in range(T):
            code_t = code[t]
            grid_charge_t = grid_charge[t]
            grid_discharge_t = grid_discharge[t]
            self.grid_charge_max = grid_charge_t  
            self.grid_discharge_max = grid_discharge_t

            pv_t = pv_generation[t]
            load_t = load_demand[t]
            
            # Debug: Print exact values for hour 0
            """
            if t == 0:
                print(f"DEBUG Hour {t}: exact load_t = {load_t:.10f}")
                print(f"DEBUG Hour {t}: exact pv_t = {pv_t:.10f}")
            """
            buy_price_t = buy_prices[t]
            sell_price_t = sell_prices[t]
            current_soc = Soc[t]
            
            # 获取当前小时的优先级
            solar_priority_t = solar_priorities[t]
            load_priority_t = load_priorities[t]
            
            # 可用电池容量计算
            max_charge_power = min(self.P_bat_max, 
                                 (self.Soc_max - current_soc) * self.E_bat / self.eta_c)
            max_discharge_power = max(0, min(self.P_bat_max, 
                                           (current_soc - self.Soc_min) * self.E_bat * self.eta_d))
            
            # 判断是否使用特殊工况
            use_g_mode = (self.grid_charge_max > 0 and 
                         solar_priority_t == [1,2,3] and 
                         load_priority_t == [1,2,0])
            
            use_h_mode = (self.grid_discharge_max > 0 and 
                         solar_priority_t == [1,2,0] and 
                         load_priority_t == [1,2,3])
            
            if use_g_mode:
                # g工况：电网充电模式
                current_battery_energy = current_soc * self.E_bat
                if use_optimizeed_gh:
                    # 使用优化后的功率
                    result = self._apply_g_mode_optimized(pv_t, load_t, current_battery_energy, self.E_bat, self.grid_charge_max)
                else:
                    result = self._apply_g_mode(pv_t, load_t, current_battery_energy, self.E_bat, self.grid_charge_max)
                
                P_pv_load[t] = result['pv_to_load']
                P_pv_bat[t] = result['pv_to_bat']
                P_pv_grid[t] = result['pv_to_grid']
                P_grid_load[t] = result['grid_to_load']
                P_grid_bat[t] = result['grid_to_bat']
                P_bat_load[t] = result['bat_to_load']
                P_bat_grid[t] = result['bat_to_grid']
                
                # 更新电池电量
                new_battery_energy = result['new_battery_energy']
                
            elif use_h_mode:
                # h工况：电池放电模式
                current_battery_energy = current_soc * self.E_bat
                if use_optimizeed_gh:
                    # 使用优化后的功率
                    result = self._apply_h_mode_optimized(pv_t, load_t, current_battery_energy, self.Soc_min, self.E_bat, self.grid_discharge_max)
                else:
                    result = self._apply_h_mode(pv_t, load_t, current_battery_energy, self.Soc_min, self.E_bat, self.grid_discharge_max)
                
                P_pv_load[t] = result['pv_to_load']
                P_pv_bat[t] = result['pv_to_bat']
                P_pv_grid[t] = result['pv_to_grid']
                P_grid_load[t] = result['grid_to_load']
                P_grid_bat[t] = result['grid_to_bat']
                P_bat_load[t] = result['bat_to_load']
                P_bat_grid[t] = result['bat_to_grid']
                
                # 更新电池电量
                new_battery_energy = result['new_battery_energy']
                
            else:
                # 标准模式：使用原有的优先级规则
                remaining_pv = pv_t
                pv_to_grid, pv_to_load, pv_to_bat = self._allocate_solar_power(
                    remaining_pv, load_t, max_charge_power, solar_priority_t)
                
                P_pv_load[t] = pv_to_load
                P_pv_bat[t] = pv_to_bat
                P_pv_grid[t] = pv_to_grid
                
                # 根据load_priority满足剩余负载需求
                remaining_load = load_t - pv_to_load
                
                # Debug: Print allocation details for hour 0
                """
                if t == 0:
                    print(f"DEBUG Hour {t}: pv_to_load = {pv_to_load:.10f}")
                    print(f"DEBUG Hour {t}: remaining_load = {remaining_load:.10f}")
                    print(f"DEBUG Hour {t}: max_discharge_power = {max_discharge_power:.10f}")
                """
                grid_to_load, bat_to_load = self._allocate_load_power(
                    remaining_load, max_discharge_power, load_priority_t)
                
                # Debug: Print results for hour 0
                """
                if t == 0:
                    print(f"DEBUG Hour {t}: grid_to_load = {grid_to_load:.10f}")
                    print(f"DEBUG Hour {t}: bat_to_load = {bat_to_load:.10f}")
                    print(f"DEBUG Hour {t}: total allocated = {grid_to_load + bat_to_load:.10f}")
                """

                P_grid_load[t] = grid_to_load
                P_bat_load[t] = bat_to_load
                P_grid_bat[t] = 0  # 标准模式下电网不给电池充电
                P_bat_grid[t] = 0  # 标准模式下电池不向电网放电
                
                # 计算新的电池电量
                energy_change = (pv_to_bat * self.eta_c - bat_to_load / self.eta_d)
                new_battery_energy = current_soc * self.E_bat + energy_change
            
            # 设置电池充放电功率
            P_bat_ch[t] = P_pv_bat[t] + P_grid_bat[t]
            P_bat_dis[t] = P_bat_load[t] + P_bat_grid[t]
            
            # 设置电池状态
            if P_bat_ch[t] > 1:  # 充电
                x_charge[t] = 1
            elif P_bat_dis[t] > 1:  # 放电
                x_discharge[t] = 1
            else:  # 待机
                x_idle[t] = 1
            
            # 计算电网交易
            P_purchase[t] = P_grid_load[t] + P_grid_bat[t]
            P_sell[t] = P_pv_grid[t] + P_bat_grid[t]
            
            # 更新SOC
            if use_g_mode or use_h_mode:
                # g/h工况直接使用计算出的电池电量
                # Soc[t+1] = max(self.Soc_min, min(self.Soc_max, new_battery_energy / self.E_bat))
                Soc[t+1] = new_battery_energy / self.E_bat
            else:
                # 标准模式使用能量变化计算SOC
                energy_change = (P_bat_ch[t] * self.eta_c - P_bat_dis[t] / self.eta_d) / self.E_bat
                #Soc[t+1] = max(self.Soc_min, min(self.Soc_max, Soc[t] + energy_change))
                Soc[t+1] = Soc[t] + energy_change
            Bat_remain[t+1] = Soc[t+1] * self.E_bat
            
            # 计算成本
            money_spend += P_purchase[t] * buy_price_t / 1000
            money_earn += P_sell[t] * sell_price_t / 1000
            total_cost += P_purchase[t] * buy_price_t / 1000 - P_sell[t] * sell_price_t / 1000
        
        # 计算净电网功率
        P_net_grid_value = P_purchase - P_sell
        
        return {
            'status': 'optimal',
            'total_cost': total_cost,  # 减去光伏直供收益
            'money_spend': money_spend,
            'money_earn': money_earn,
            'power_flows': {
                'P_pv_load': P_pv_load,
                'P_pv_bat': P_pv_bat,
                'P_pv_grid': P_pv_grid,
                'P_grid_load': P_grid_load,
                'P_grid_bat': P_grid_bat,
                'P_bat_load': P_bat_load,
                'P_bat_grid': P_bat_grid,
                'P_purchase': P_purchase,
                'P_sell': P_sell,
                'P_net_grid': P_net_grid_value
            },
            'battery': {
                'P_bat_ch': P_bat_ch,
                'P_bat_dis': P_bat_dis,
                'x_charge': x_charge,
                'x_discharge': x_discharge,
                'x_idle': x_idle,
                'SOC': Soc
            }
        }
    
    def _allocate_solar_power(self, pv_power, load_demand, max_charge_power, solar_priority):
        """
        根据solar_priority分配光伏发电
        solar_priority: [电网，负载，电池] 的优先级
        """
        pv_to_grid = 0
        pv_to_load = 0
        pv_to_bat = 0
        
        remaining_pv = pv_power
        
        # 创建优先级排序 (index, priority)
        priorities = [
            (0, solar_priority[0]),  # 电网
            (1, solar_priority[1]),  # 负载
            (2, solar_priority[2])   # 电池
        ]
        # 按优先级降序排列
        priorities.sort(key=lambda x: x[1], reverse=True)
        
        for target_idx, priority in priorities:
            if remaining_pv <= 0:
                break
                
            if target_idx == 0:  # 电网
                # 光伏全部可以送电网（不考虑上网限制）
                pv_to_grid = remaining_pv
                remaining_pv = 0
                
            elif target_idx == 1:  # 负载
                # 光伏给负载，但不能超过负载需求
                allocation = min(remaining_pv, load_demand)
                pv_to_load = allocation
                remaining_pv -= allocation
                
            elif target_idx == 2:  # 电池
                # 光伏给电池充电，受电池充电功率和容量限制
                allocation = min(remaining_pv, max_charge_power)
                pv_to_bat = allocation
                remaining_pv -= allocation
        
        return pv_to_grid, pv_to_load, pv_to_bat
    
    def _allocate_load_power(self, remaining_load, max_discharge_power, load_priority):
        """
        根据load_priority分配负载功率来源
        load_priority: [电网，光伏，电池] 的优先级
        注意：光伏部分在solar allocation阶段已处理，这里只处理电网和电池
        """
        grid_to_load = 0
        bat_to_load = 0
        
        if remaining_load <= 0:
            return grid_to_load, bat_to_load
        
        # 只考虑电网和电池（光伏已在前面处理）
        # load_priority[0] = 电网优先级, load_priority[2] = 电池优先级
        
        if load_priority[0] >= load_priority[2]:  # 电网优先级 >= 电池优先级
            # 先电网后电池
            grid_to_load = remaining_load  # 电网可以满足所有剩余负载
            bat_to_load = 0
        else:  # 电池优先级 > 电网优先级
            # 先电池后电网
            bat_allocation = min(remaining_load, max_discharge_power)
            bat_to_load = bat_allocation
            grid_to_load = remaining_load - bat_allocation
        
        return grid_to_load, bat_to_load
    
    def _apply_g_mode(self, solar, load, battery, capacity, grid_charge_max):
        """
        g工况：电网充电模式
        优先级：光伏→电池充电 → 电网→电池充电 → 满足负载
        
        参数与function_gh.py中的func_g保持一致，但考虑充电效率
        """
        solar_to_load = solar_to_grid = solar_to_battery = 0
        battery_to_load = battery_to_grid = grid_to_load = grid_to_battery = 0
        
        # 考虑充电效率的可用容量
        available_capacity_for_solar = (capacity - battery) / self.eta_c
        available_capacity_for_grid = (capacity - battery) / self.eta_c
        
        if solar <= available_capacity_for_solar:
            # 情况1：光伏全部用于电池充电，电网补充充电并供负载
            solar_to_battery = solar
            remaining_capacity = capacity - battery - solar_to_battery * self.eta_c
            grid_to_battery = min(grid_charge_max, remaining_capacity / self.eta_c)
            grid_to_load = load
            battery += (solar_to_battery + grid_to_battery) * self.eta_c
        elif solar <= load + available_capacity_for_solar:
            # 情况2：光伏部分充电，部分供负载，电网补充负载
            solar_to_battery = (capacity - battery) / self.eta_c
            solar_to_load = solar - solar_to_battery
            grid_to_load = load - solar_to_load
            battery += solar_to_battery * self.eta_c
        else:
            # 情况3：电池充满，光伏供负载并售电
            solar_to_battery = (capacity - battery) / self.eta_c
            solar_to_load = load
            solar_to_grid = solar - solar_to_battery - solar_to_load
            battery += solar_to_battery * self.eta_c
        
        return {
            'pv_to_load': solar_to_load,
            'pv_to_grid': solar_to_grid,
            'pv_to_bat': solar_to_battery,
            'bat_to_load': battery_to_load,
            'bat_to_grid': battery_to_grid,
            'grid_to_load': grid_to_load,
            'grid_to_bat': grid_to_battery,
            'new_battery_energy': battery
        }
    
    def _apply_h_mode(self, solar, load, battery, soc_min, capacity, grid_discharge_max):
        """
        h工况：电池放电模式  
        优先级：电池→负载 → 电池→电网售电 → 光伏→电网售电
        
        参数与function_gh.py中的func_h保持一致，但考虑放电效率
        """
        solar_to_load = solar_to_grid = solar_to_battery = 0
        battery_to_load = battery_to_grid = grid_to_load = grid_to_battery = 0
        
        # 计算可用电池电量（高于备用SOC的部分），考虑放电效率
        battery_available_energy = max(0, battery - capacity * soc_min)
        battery_available_power = battery_available_energy * self.eta_d
        
        if battery_available_power >= load:
            # 情况1：电池能完全满足负载，剩余向电网放电，光伏全部售电
            battery_to_load = load
            remaining_power = battery_available_power - battery_to_load
            battery_to_grid = min(grid_discharge_max, remaining_power)
            solar_to_grid = solar
            # 考虑放电效率：实际消耗的电池能量 = 放电功率 / 放电效率
            battery = battery - (battery_to_load + battery_to_grid) / self.eta_d
        elif battery_available_power + solar >= load:
            # 情况2：电池+光伏能满足负载，光伏剩余售电
            battery_to_load = battery_available_power
            solar_to_load = load - battery_to_load
            solar_to_grid = solar - solar_to_load
            battery -= battery_to_load / self.eta_d
        else:
            # 情况3：电池+光伏不足，需要电网补充
            battery_to_load = battery_available_power
            solar_to_load = solar
            grid_to_load = load - battery_to_load - solar_to_load
            battery -= battery_to_load / self.eta_d
        
        return {
            'pv_to_load': solar_to_load,
            'pv_to_grid': solar_to_grid,
            'pv_to_bat': solar_to_battery,
            'bat_to_load': battery_to_load,
            'bat_to_grid': battery_to_grid,
            'grid_to_load': grid_to_load,
            'grid_to_bat': grid_to_battery,
            'new_battery_energy': battery
        }

    def _apply_g_mode_optimized(self, solar, load, battery, capacity, grid_charge_max):
        """
        优化的g工况：电网充电模式 - 满足SOC约束后停止充电
        
        优化逻辑：
        1. 如果当前SOC >= soc_min，则不进行电网充电，只进行必要的光伏利用
        2. 如果当前SOC < soc_min，则优先充电到soc_min即可，不强制充到满容量
        
        参数与demo.py中的func_g保持一致，但考虑充电效率和SOC约束
        """
        solar_to_load = solar_to_grid = solar_to_battery = 0
        battery_to_load = battery_to_grid = grid_to_load = grid_to_battery = 0
        
        # 计算当前SOC和目标SOC
        current_soc = battery / capacity
        target_battery = capacity * self.Soc_min  # 目标电池电量（满足最小SOC）
        
        # 如果当前SOC已经满足约束，采用标准光伏优先策略
        if current_soc >= self.Soc_min:
            # 光伏优先供负载，剩余售电，不进行电网充电
            solar_to_load = min(solar, load)
            solar_to_grid = solar - solar_to_load
            grid_to_load = load - solar_to_load
            new_battery_energy = battery
            
            return {
                'pv_to_load': solar_to_load,
                'pv_to_grid': solar_to_grid,
                'pv_to_bat': solar_to_battery,
                'bat_to_load': battery_to_load,
                'bat_to_grid': battery_to_grid,
                'grid_to_load': grid_to_load,
                'grid_to_bat': grid_to_battery,
                'new_battery_energy': new_battery_energy
            }
        
        # 当前SOC < soc_min，需要充电到目标SOC
        needed_charge_energy = target_battery - battery  # 需要充入的电量
        needed_charge_power = needed_charge_energy / self.eta_c  # 考虑效率损失
        
        # 计算可用的充电功率
        available_solar_for_charge = max(0, solar - load)  # 光伏满足负载后的剩余
        max_charge_from_grid = min(grid_charge_max, needed_charge_power)  # 电网充电上限
        
        # 分配充电功率
        if available_solar_for_charge >= needed_charge_power:
            # 情况1：光伏剩余功率足够充电到目标SOC
            solar_to_load = min(solar, load)
            solar_to_battery = needed_charge_power
            solar_to_grid = solar - solar_to_load - solar_to_battery
            grid_to_load = load - solar_to_load
            grid_to_battery = 0
            new_battery_energy = target_battery
            
        elif available_solar_for_charge > 0:
            # 情况2：光伏剩余不足，需要电网补充充电
            solar_to_load = min(solar, load)
            solar_to_battery = available_solar_for_charge
            solar_to_grid = 0
            grid_to_load = load - solar_to_load
            grid_to_battery = min(max_charge_from_grid, 
                                needed_charge_power - solar_to_battery)
            new_battery_energy = battery + (solar_to_battery + grid_to_battery) * self.eta_c
            
        else:
            # 情况3：光伏全部用于供负载，电网充电+供负载
            solar_to_load = solar
            solar_to_battery = 0
            solar_to_grid = 0
            grid_to_load = load - solar
            grid_to_battery = max_charge_from_grid
            new_battery_energy = battery + grid_to_battery * self.eta_c
        
        return {
            'pv_to_load': solar_to_load,
            'pv_to_grid': solar_to_grid,
            'pv_to_bat': solar_to_battery,
            'bat_to_load': battery_to_load,
            'bat_to_grid': battery_to_grid,
            'grid_to_load': grid_to_load,
            'grid_to_bat': grid_to_battery,
            'new_battery_energy': new_battery_energy
        }

    def _apply_h_mode_optimized(self, solar, load, battery, soc_min, capacity, grid_discharge_max):
        """
        优化的h工况：电池放电模式 - 满足负载需求后停止过度放电
        
        优化逻辑：
        1. 优先使用电池满足负载需求
        2. 如果电池+光伏能满足负载，不向电网放电
        3. 只有在有显著剩余电池电量时才向电网放电
        
        参数与demo.py中的func_h保持一致，但考虑放电效率和合理的放电策略
        """
        solar_to_load = solar_to_grid = solar_to_battery = 0
        battery_to_load = battery_to_grid = grid_to_load = grid_to_battery = 0
        
        # 计算可用电池电量（高于备用SOC的部分）
        reserve_battery = capacity * soc_min
        available_battery_energy = max(0, battery - reserve_battery)
        available_battery_power = available_battery_energy * self.eta_d
        
        # 计算总可用功率（光伏+电池）
        total_available_power = solar + available_battery_power
        
        if total_available_power >= load:
            # 情况1：光伏+电池能满足负载需求
            if solar >= load:
                # 光伏足够满足负载，电池不需要放电
                solar_to_load = load
                solar_to_grid = solar - load
                battery_to_load = 0
                battery_to_grid = 0
                grid_to_load = 0
                new_battery_energy = battery
                
            else:
                # 光伏不足，电池补充满足负载
                solar_to_load = solar
                needed_from_battery = load - solar
                battery_to_load = min(needed_from_battery, available_battery_power)
                grid_to_load = load - solar_to_load - battery_to_load
                
                # 计算剩余电池功率，只有显著剩余时才考虑向电网放电
                remaining_battery_power = available_battery_power - battery_to_load
                
                # 设置一个阈值，只有剩余功率大于一定值时才向电网放电
                # 避免为了少量收益而频繁充放电
                discharge_threshold = min(1000, capacity * 0.05)  # 至少5%容量或1kW
                
                if remaining_battery_power > discharge_threshold:
                    battery_to_grid = min(grid_discharge_max, 
                                        remaining_battery_power - discharge_threshold)
                else:
                    battery_to_grid = 0
                
                # 剩余光伏售电
                solar_to_grid = 0
                
                # 更新电池电量
                total_discharge = battery_to_load + battery_to_grid
                new_battery_energy = battery - total_discharge / self.eta_d
                
        else:
            # 情况2：光伏+电池都不足以满足负载，需要电网补充
            solar_to_load = solar
            battery_to_load = available_battery_power
            grid_to_load = load - solar_to_load - battery_to_load
            battery_to_grid = 0  # 电池电量不足时不向电网放电
            solar_to_grid = 0
            
            # 更新电池电量
            new_battery_energy = battery - battery_to_load / self.eta_d
        
        return {
            'pv_to_load': solar_to_load,
            'pv_to_grid': solar_to_grid,
            'pv_to_bat': solar_to_battery,
            'bat_to_load': battery_to_load,
            'bat_to_grid': battery_to_grid,
            'grid_to_load': grid_to_load,
            'grid_to_bat': grid_to_battery,
            'new_battery_energy': new_battery_energy
        }

    def dynamic_programming(
        self, 
        pv_generation, 
        load_demand, 
        buy_prices, 
        sell_prices, 
        initial_soc=0.5):
        """
        基于动态规划的能源管理系统
        
        动态规划状态定义：
        - 状态: (时间t, SOC离散值)
        - 决策: 电池动作（充电/放电/待机）及功率
        - 目标: 最小化总成本
        """
        T = len(pv_generation)
        
        # SOC离散化 - 将连续的SOC空间离散化以适应DP
        soc_resolution = 100  # SOC精度，表示SOC被分成100个离散点
        
        # 扩展SOC状态空间，包含初始SOC（即使它低于soc_min）
        soc_range_min = min(self.Soc_min, initial_soc)
        soc_range_max = self.Soc_max
        soc_states = np.linspace(soc_range_min, soc_range_max, soc_resolution)
        
        # DP表：dp[t][soc_idx] = (最小成本, 最优决策路径)
        dp_cost = np.full((T+1, soc_resolution), np.inf)
        dp_decisions = {}  # 存储决策路径
        
        # 初始化：找到最接近initial_soc的离散状态
        init_soc_idx = np.argmin(np.abs(soc_states - initial_soc))
        dp_cost[0][init_soc_idx] = 0
        
        print(f"DP初始化: initial_soc={initial_soc:.3f}, soc_min={self.Soc_min:.3f}")
        print(f"SOC状态空间: [{soc_range_min:.3f}, {soc_range_max:.3f}]")
        print(f"初始状态索引: {init_soc_idx}, 对应SOC值: {soc_states[init_soc_idx]:.3f}")
        
        # 动态规划主循环
        for t in range(T):
            pv_t = pv_generation[t]
            load_t = load_demand[t]
            buy_price_t = buy_prices[t]
            sell_price_t = sell_prices[t]
            
            for soc_idx, current_soc in enumerate(soc_states):
                if dp_cost[t][soc_idx] == np.inf:
                    continue  # 当前状态不可达
                
                # 计算当前SOC下的可用功率
                max_charge_power = min(self.P_bat_max, 
                                     (self.Soc_max - current_soc) * self.E_bat / self.eta_c)
                # 如果当前SOC低于最小值，不允许放电
                if current_soc < self.Soc_min:
                    max_discharge_power = 0
                else:
                    max_discharge_power = min(self.P_bat_max, 
                                            (current_soc - self.Soc_min) * self.E_bat * self.eta_d)
                
                # 尝试三种电池状态的所有可能决策
                battery_actions = self._generate_battery_actions(
                    pv_t, load_t, max_charge_power, max_discharge_power)
                
                for action in battery_actions:
                    # 解析动作
                    bat_state, bat_power, power_allocation = action
                    
                    # 计算状态转移
                    next_soc, cost, valid = self._evaluate_action(
                        current_soc, bat_state, bat_power, power_allocation,
                        pv_t, load_t, buy_price_t, sell_price_t)
                    
                    if not valid:
                        continue
                    
                    # 找到next_soc对应的离散状态索引
                    next_soc_idx = np.argmin(np.abs(soc_states - next_soc))
                    
                    # 更新DP表
                    new_cost = dp_cost[t][soc_idx] + cost
                    if new_cost < dp_cost[t+1][next_soc_idx]:
                        dp_cost[t+1][next_soc_idx] = new_cost
                        dp_decisions[(t+1, next_soc_idx)] = {
                            'prev_soc_idx': soc_idx,
                            'action': action,
                            'power_allocation': power_allocation
                        }
        
        # 回溯找到最优解
        min_final_cost = np.min(dp_cost[T])
        if min_final_cost == np.inf:
            return {'status': 'failed', 'message': 'No feasible solution found'}
        
        final_soc_idx = np.argmin(dp_cost[T])
        optimal_path = self._backtrack_solution(dp_decisions, T, final_soc_idx, soc_states)
        
        # 构造输出格式，与optimize_complete_model保持一致
        return self._format_dp_result(optimal_path, pv_generation, load_demand, 
                                     buy_prices, sell_prices, min_final_cost, initial_soc)
    
    def _generate_battery_actions(self, pv_t, load_t, max_charge_power, max_discharge_power):
        """生成电池可能的动作空间，考虑电网-电池交互约束"""
        actions = []
        
        # 1. 待机状态
        idle_alloc = self._allocate_power_for_idle(pv_t, load_t)
        if idle_alloc:
            actions.append(('idle', 0, idle_alloc))
        
        # 2. 充电状态 - 考虑电池功率和电网充电约束
        if max_charge_power > 1:  # 降低最小充电功率阈值
            # 电池总充电功率不能超过P_bat_max
            max_bat_charge = min(max_charge_power, self.P_bat_max)
            
            # 改进充电功率离散化策略
            charge_powers = []
            
            # 添加关键充电功率点
            # 1. 光伏剩余功率充电点（当光伏大于负载时）
            if pv_t > load_t:
                pv_surplus = pv_t - load_t  # 光伏剩余功率
                charge_powers.append(min(pv_surplus, max_bat_charge))  # 用光伏剩余功率充电
            
            # 2. 最大可用充电功率点
            charge_powers.append(max_bat_charge)
            
            # 3. 如果允许电网充电，添加混合充电点
            if self.grid_charge_max > 0:
                # 光伏+电网组合充电点
                if pv_t > load_t:
                    pv_available = pv_t - load_t
                    grid_supplement = min(self.grid_charge_max, max_bat_charge - pv_available)
                    if grid_supplement > 0:
                        total_charge = pv_available + grid_supplement
                        charge_powers.append(min(total_charge, max_bat_charge))
                
                # 纯电网充电点（当光伏不足时）
                if pv_t <= load_t:
                    grid_charge_powers = [
                        min(self.grid_charge_max * 0.25, max_bat_charge),  # 25%电网充电功率
                        min(self.grid_charge_max * 0.5, max_bat_charge),   # 50%电网充电功率  
                        min(self.grid_charge_max * 0.75, max_bat_charge),  # 75%电网充电功率
                        min(self.grid_charge_max, max_bat_charge)          # 100%电网充电功率
                    ]
                    charge_powers.extend(grid_charge_powers)
            
            # 添加标准离散化点
            power_resolution = 20  # 增加分辨率
            if max_bat_charge > 50:
                standard_powers = np.linspace(50, max_bat_charge, min(power_resolution, int(max_bat_charge/100)))
                charge_powers.extend(standard_powers)
            elif max_bat_charge > 10:
                # 对于较小的最大充电功率，也提供一些选择点
                standard_powers = np.linspace(10, max_bat_charge, min(10, int(max_bat_charge/10)))
                charge_powers.extend(standard_powers)
            
            # 去重并排序
            charge_powers = sorted(list(set([p for p in charge_powers if p >= 1 and p <= max_bat_charge])))
            
            for charge_power in charge_powers:
                power_alloc = self._allocate_power_for_charging(pv_t, load_t, charge_power)
                if power_alloc:
                    actions.append(('charge', charge_power, power_alloc))
        
        # 3. 放电状态 - 考虑电池功率和电网放电约束  
        if max_discharge_power > 1:  # 降低最小放电功率阈值
            # 电池总放电功率不能超过P_bat_max
            max_bat_discharge = min(max_discharge_power, self.P_bat_max)
            
            # 改进放电功率离散化策略
            discharge_powers = []
            
            # 添加关键放电功率点
            if pv_t < load_t:  # 如果光伏不足以满足负载
                needed_discharge = load_t - pv_t  # 理想放电功率应该正好补充负载缺口
                discharge_powers.append(min(needed_discharge, max_bat_discharge))
            
            # 添加满负载放电点
            if load_t <= max_bat_discharge:
                discharge_powers.append(load_t)  # 完全满足负载需求的放电功率
            
            # 添加标准离散化点
            power_resolution = 20  # 增加分辨率
            if max_bat_discharge > 50:
                standard_powers = np.linspace(50, max_bat_discharge, min(power_resolution, int(max_bat_discharge/100)))
                discharge_powers.extend(standard_powers)
            
            # 去重并排序
            discharge_powers = sorted(list(set([p for p in discharge_powers if p >= 1 and p <= max_bat_discharge])))
            
            for discharge_power in discharge_powers:
                power_alloc = self._allocate_power_for_discharging(pv_t, load_t, discharge_power)
                if power_alloc:
                    actions.append(('discharge', discharge_power, power_alloc))
        
        return actions
    
    def _allocate_power_for_idle(self, pv_t, load_t):
        """电池待机时的功率分配"""
        # 光伏优先供负载，剩余售电；不足部分从电网购电
        pv_to_load = min(pv_t, load_t)
        pv_to_grid = pv_t - pv_to_load
        grid_to_load = load_t - pv_to_load
        
        return {
            'P_pv_load': pv_to_load,
            'P_pv_bat': 0,
            'P_pv_grid': pv_to_grid,
            'P_grid_load': grid_to_load,
            'P_grid_bat': 0,
            'P_bat_load': 0,
            'P_bat_grid': 0,
            'P_bat_ch': 0,
            'P_bat_dis': 0
        }
    
    def _allocate_power_for_charging(self, pv_t, load_t, charge_power):
        """电池充电时的功率分配，符合约束: P_bat_ch = P_pv_bat + P_grid_bat"""
        # 光伏优先供负载
        pv_to_load = min(pv_t, load_t)
        remaining_pv = pv_t - pv_to_load
        remaining_load = load_t - pv_to_load
        
        # 分配充电功率来源：优先光伏，不足时电网补充（如果允许）
        pv_to_bat = min(remaining_pv, charge_power)
        needed_grid_charge = charge_power - pv_to_bat
        
        # 检查电网充电约束
        if needed_grid_charge > 0:
            if self.grid_charge_max <= 0:  # 不允许电网充电
                return None
            if needed_grid_charge > self.grid_charge_max:  # 超过电网充电限制
                return None
        
        grid_to_bat = max(0, needed_grid_charge)
        
        # 剩余光伏售电
        pv_to_grid = remaining_pv - pv_to_bat
        
        # 剩余负载从电网购电
        grid_to_load = remaining_load
        
        return {
            'P_pv_load': pv_to_load,
            'P_pv_bat': pv_to_bat,
            'P_pv_grid': pv_to_grid,
            'P_grid_load': grid_to_load,
            'P_grid_bat': grid_to_bat,
            'P_bat_load': 0,
            'P_bat_grid': 0,
            'P_bat_ch': charge_power,  # P_bat_ch = P_pv_bat + P_grid_bat
            'P_bat_dis': 0
        }
    
    def _allocate_power_for_discharging(self, pv_t, load_t, discharge_power):
        """电池放电时的功率分配，符合约束: P_bat_dis = P_bat_load + P_bat_grid"""
        # 光伏优先供负载
        pv_to_load = min(pv_t, load_t)
        remaining_load = load_t - pv_to_load
        remaining_pv = pv_t - pv_to_load
        
        # 电池优先供负载，剩余可以售电（如果允许）
        bat_to_load = min(remaining_load, discharge_power)
        remaining_discharge = discharge_power - bat_to_load
        
        # 检查电网放电约束
        bat_to_grid = 0
        if remaining_discharge > 0:
            if self.grid_discharge_max > 0:  # 允许电池售电给电网
                bat_to_grid = min(remaining_discharge, self.grid_discharge_max)
            # 如果不允许或超出限制，剩余放电功率就浪费了
            # 但我们仍然需要满足 P_bat_dis = P_bat_load + P_bat_grid
        
        # 剩余光伏售电
        pv_to_grid = remaining_pv
        
        # 剩余负载从电网购电
        grid_to_load = remaining_load - bat_to_load
        
        # 验证功率平衡
        if bat_to_load + bat_to_grid != discharge_power:
            # 如果不允许电池售电或超出限制，则减少放电功率
            actual_discharge = bat_to_load + bat_to_grid
            if actual_discharge < 10:  # 如果实际可用放电功率太小，返回None
                return None
        else:
            actual_discharge = discharge_power
        
        return {
            'P_pv_load': pv_to_load,
            'P_pv_bat': 0,
            'P_pv_grid': pv_to_grid,
            'P_grid_load': grid_to_load,
            'P_grid_bat': 0,
            'P_bat_load': bat_to_load,
            'P_bat_grid': bat_to_grid,
            'P_bat_ch': 0,
            'P_bat_dis': actual_discharge  # P_bat_dis = P_bat_load + P_bat_grid
        }
    
    def _evaluate_action(self, current_soc, bat_state, bat_power, power_alloc, 
                        pv_t, load_t, buy_price_t, sell_price_t):
        """评估一个动作的成本和状态转移"""
        try:
            # 计算SOC变化
            if bat_state == 'charge':
                energy_change = bat_power * self.eta_c / self.E_bat
            elif bat_state == 'discharge':
                energy_change = -bat_power / (self.eta_d * self.E_bat)
            else:  # idle
                energy_change = 0
            
            next_soc = current_soc + energy_change
            
            # 严格检查SOC约束
            # 注意：如果当前SOC已经低于soc_min，只允许充电或保持，不允许进一步放电
            if current_soc < self.Soc_min:
                # 当前已低于最小SOC，不允许放电（energy_change < 0）
                if energy_change < -1e-6:
                    return next_soc, float('inf'), False
                # 允许充电或待机
            else:
                # 当前SOC正常，不允许降到最小SOC以下
                if next_soc < self.Soc_min - 1e-6:
                    return next_soc, float('inf'), False
            
            # 检查上限约束
            if next_soc > self.Soc_max + 1e-6:
                return next_soc, float('inf'), False
            
            # 计算成本 - 购电包括电网供负载和电网给电池充电
            purchase_cost = (power_alloc['P_grid_load'] + power_alloc['P_grid_bat']) * buy_price_t / 1000
            sell_revenue = (power_alloc['P_pv_grid'] + power_alloc['P_bat_grid']) * sell_price_t / 1000
            base_cost = purchase_cost - sell_revenue
            
            # 添加电池充电时考虑效率损耗的调整
            if bat_state == 'charge' and power_alloc['P_grid_bat'] > 0:
                # 电网充电需要考虑效率损耗：实际存储的电量少于购买的电量
                grid_charge_loss = power_alloc['P_grid_bat'] * (1 - self.eta_c) * buy_price_t / 1000
                base_cost += grid_charge_loss
            
            # 添加SOC约束违反的惩罚成本
            penalty_cost = 0
            if next_soc < self.Soc_min:
                # 对低于最小SOC的状态施加额外惩罚，激励尽快充电
                soc_violation = self.Soc_min - next_soc
                penalty_cost = soc_violation * 10  # 惩罚系数：每1%的SOC违反增加10元成本
            elif next_soc > self.Soc_max:
                # 对超过最大SOC的状态施加额外惩罚，激励尽快放电
                soc_violation = next_soc - self.Soc_max
                penalty_cost = soc_violation * 10  # 惩罚系数：每1%的SOC违反增加10元成本
            
            # 添加储能价值激励：当SOC较低时，给电池充电有额外价值；当SOC过高时，激励放电
            storage_incentive = 0
            if bat_state == 'charge' and next_soc < 0.7:  # SOC低于70%时激励充电
                # 根据当前电价和历史高电价的差值来评估储能价值
                # 假设未来可以在更高电价时段放电，给予一定的价值激励
                max_future_sell_price = 0.53  # 基于历史数据的最高售电价
                current_charge_cost = buy_price_t if power_alloc['P_grid_bat'] > 0 else 0
                if max_future_sell_price > current_charge_cost:
                    # 每kWh储能价值 = (未来售价 - 当前充电成本) * 效率
                    storage_value_per_kwh = (max_future_sell_price - current_charge_cost) * self.eta_c * self.eta_d
                    stored_energy_kwh = bat_power / 1000  # 充电功率转换为kWh
                    storage_incentive = -storage_value_per_kwh * stored_energy_kwh * 0.3  # 30%的储能价值激励
            elif bat_state == 'discharge' and current_soc > self.Soc_max:  # SOC超过最大值时激励放电
                # 当SOC过高时，给放电行为额外的价值激励
                discharge_incentive_per_kwh = 0.05  # 每kWh放电给予5分钱的激励
                discharged_energy_kwh = bat_power / 1000
                storage_incentive = -discharge_incentive_per_kwh * discharged_energy_kwh
            
            total_cost = base_cost + penalty_cost + storage_incentive
            
            return next_soc, total_cost, True
            
        except Exception:
            return 0, float('inf'), False
    
    def _backtrack_solution(self, dp_decisions, T, final_soc_idx, soc_states):
        """回溯最优解路径"""
        path = []
        current_t = T
        current_soc_idx = final_soc_idx
        
        while current_t > 0:
            if (current_t, current_soc_idx) not in dp_decisions:
                break
                
            decision = dp_decisions[(current_t, current_soc_idx)]
            path.append({
                'time': current_t - 1,
                'prev_soc': soc_states[decision['prev_soc_idx']],  # 前一状态的SOC
                'current_soc': soc_states[current_soc_idx],       # 当前状态的SOC
                'action': decision['action'],
                'power_allocation': decision['power_allocation']
            })
            
            current_soc_idx = decision['prev_soc_idx']
            current_t -= 1
        
        path.reverse()
        return path
    
    def _format_dp_result(self, optimal_path, pv_generation, load_demand, buy_prices, sell_prices, total_cost, initial_soc):
        """将DP结果格式化为与optimize_complete_model一致的输出"""
        T = len(pv_generation)
        
        # 初始化输出数组
        P_pv_load = np.zeros(T)
        P_pv_bat = np.zeros(T)
        P_pv_grid = np.zeros(T)
        P_grid_load = np.zeros(T)
        P_grid_bat = np.zeros(T)
        P_bat_load = np.zeros(T)
        P_bat_grid = np.zeros(T)
        P_purchase = np.zeros(T)
        P_sell = np.zeros(T)
        P_bat_ch = np.zeros(T)
        P_bat_dis = np.zeros(T)
        x_charge = np.zeros(T, dtype=int)
        x_discharge = np.zeros(T, dtype=int)
        x_idle = np.zeros(T, dtype=int)
        Soc = np.zeros(T+1)
        
        # 设置初始SOC
        Soc[0] = initial_soc
        
        money_spend = 0
        money_earn = 0
        
        # 填充结果数组
        for i, step in enumerate(optimal_path):
            if i >= T:
                break
                
            t = step['time']
            power_alloc = step['power_allocation']
            action = step['action']
            
            P_pv_load[t] = power_alloc['P_pv_load']
            P_pv_bat[t] = power_alloc['P_pv_bat']
            P_pv_grid[t] = power_alloc['P_pv_grid']
            P_grid_load[t] = power_alloc['P_grid_load']
            P_grid_bat[t] = power_alloc['P_grid_bat']
            P_bat_load[t] = power_alloc['P_bat_load']
            P_bat_grid[t] = power_alloc['P_bat_grid']
            P_bat_ch[t] = power_alloc['P_bat_ch']
            P_bat_dis[t] = power_alloc['P_bat_dis']
            
            P_purchase[t] = P_grid_load[t] + P_grid_bat[t]
            P_sell[t] = P_pv_grid[t] + P_bat_grid[t]
            
            # 设置电池状态
            bat_state, _, _ = action
            if bat_state == 'charge':
                x_charge[t] = 1
            elif bat_state == 'discharge':
                x_discharge[t] = 1
            else:
                x_idle[t] = 1
            
            # 更新SOC
            if t + 1 < len(Soc):
                if bat_state == 'charge':
                    energy_change = P_bat_ch[t] * self.eta_c / self.E_bat
                elif bat_state == 'discharge':
                    energy_change = -P_bat_dis[t] / (self.eta_d * self.E_bat)
                else:
                    energy_change = 0
                # SOC自然更新，不强制约束（动态规划已确保路径可行）
                Soc[t+1] = Soc[t] + energy_change
            
            # 计算成本
            money_spend += P_purchase[t] * buy_prices[t] / 1000
            money_earn += P_sell[t] * sell_prices[t] / 1000
        
        # 计算净电网功率
        P_net_grid_value = P_purchase - P_sell
        
        return {
            'status': 'optimal',
            'total_cost': money_spend - money_earn,
            'money_spend': money_spend,
            'money_earn': money_earn,
            'power_flows': {
                'P_pv_load': P_pv_load,
                'P_pv_bat': P_pv_bat,
                'P_pv_grid': P_pv_grid,
                'P_grid_load': P_grid_load,
                'P_grid_bat': P_grid_bat,
                'P_bat_load': P_bat_load,
                'P_bat_grid': P_bat_grid,
                'P_purchase': P_purchase,
                'P_sell': P_sell,
                'P_net_grid': P_net_grid_value
            },
            'battery': {
                'P_bat_ch': P_bat_ch,
                'P_bat_dis': P_bat_dis,
                'x_charge': x_charge,
                'x_discharge': x_discharge,
                'x_idle': x_idle,
                'SOC': Soc
            }
        }

    def create_hourly_analysis(self, result, pv_gen, load, buy_prices, sell_prices, real_pv=None, real_load=None):
        """创建每小时详细分析数据"""
        if result['status'] != 'optimal':
            return None
        
        power_flows = result['power_flows']
        battery = result['battery']
        T = len(pv_gen)
        
        hourly_data = []
        
        for t in range(T):
            # 功率流数据
            pv_to_load = power_flows['P_pv_load'][t]
            pv_to_bat = power_flows['P_pv_bat'][t]
            pv_to_grid = power_flows['P_pv_grid'][t]
            grid_to_load = power_flows['P_grid_load'][t]
            grid_to_bat = power_flows['P_grid_bat'][t]
            bat_to_load = power_flows['P_bat_load'][t]
            bat_to_grid = power_flows['P_bat_grid'][t]
            
            # 电池状态（三态）
            bat_charge = battery['P_bat_ch'][t]
            bat_discharge = battery['P_bat_dis'][t]
            soc = battery['SOC'][t]
            
            # 确定电池状态（基于实际功率而非仅仅二进制变量）
            if bat_charge > 10:  # 实际有充电功率
                battery_state = "充电"
                state_code = 1
            elif bat_discharge > 10:  # 实际有放电功率
                battery_state = "放电" 
                state_code = -1
            else:  # 无显著充放电功率
                battery_state = "待机"
                state_code = 0
            
            # 收益计算
            purchase_cost = power_flows['P_purchase'][t] * buy_prices[t] / 1000
            sell_revenue = power_flows['P_sell'][t] * sell_prices[t] / 1000
            hourly_profit = sell_revenue - purchase_cost
            
            # 确定主要操作
            main_operation = self.determine_main_operation(
                pv_gen[t], load[t], bat_charge, bat_discharge, 
                pv_to_grid, bat_to_grid, grid_to_bat, battery_state
            )
            
            # 获取真实数据（如果提供）
            actual_pv = real_pv[t] if real_pv is not None and t < len(real_pv) else pv_gen[t]
            actual_load = real_load[t] if real_load is not None and t < len(real_load) else load[t]
            
            hourly_data.append({
                'hour': t,
                'pv_generation': pv_gen[t],           # 预测光伏发电
                'load_demand': load[t],               # 预测负载需求
                'real_pv_generation': actual_pv,     # 真实光伏发电
                'real_load_demand': actual_load,     # 真实负载需求
                'buy_price': buy_prices[t],
                'sell_price': sell_prices[t],
                
                # 功率流
                'pv_to_load': pv_to_load,
                'pv_to_bat': pv_to_bat,
                'pv_to_grid': pv_to_grid,
                'grid_to_load': grid_to_load,
                'grid_to_bat': grid_to_bat,
                'bat_to_load': bat_to_load,
                'bat_to_grid': bat_to_grid,
                
                # 电池状态（三态）
                'bat_charge': bat_charge,
                'bat_discharge': bat_discharge,
                'soc': soc * 100,  # 转换为百分比
                'battery_state': battery_state,
                'state_code': state_code,
                
                # 经济指标
                'purchase_cost': purchase_cost,
                'sell_revenue': sell_revenue,
                'hourly_profit': hourly_profit,
                'cumulative_profit': 0,  # 稍后计算
                
                # 操作描述
                'main_operation': main_operation,
                'self_consumption_rate': pv_to_load / max(pv_gen[t], 1) if pv_gen[t] > 0 else 0,
                'load_coverage_rate': (pv_to_load + bat_to_load) / max(load[t], 1),
                
                # 预测误差分析
                'pv_forecast_error': actual_pv - pv_gen[t],
                'load_forecast_error': actual_load - load[t],
                'pv_forecast_error_rate': (actual_pv - pv_gen[t]) / max(pv_gen[t], 1) * 100 if pv_gen[t] > 0 else 0,
                'load_forecast_error_rate': (actual_load - load[t]) / max(load[t], 1) * 100 if load[t] > 0 else 0
            })
        
        # 计算累计收益
        cumulative = 0
        for data in hourly_data:
            cumulative += data['hourly_profit']
            data['cumulative_profit'] = cumulative
        
        return pd.DataFrame(hourly_data)
    
    def determine_main_operation(self, pv, load, bat_charge, bat_discharge, 
                               pv_to_grid, bat_to_grid, grid_to_bat, battery_state):
        """确定每小时的主要操作类型（考虑三态）"""
        
        if battery_state == "充电" and bat_charge > 10:
            if grid_to_bat > 10:
                return "电网充电"
            else:
                return "光伏充电"
        elif battery_state == "放电" and bat_discharge > 10:
            if bat_to_grid > 10:
                return "电池售电"
            else:
                return "电池供负载"
        elif battery_state == "待机":
            if pv_to_grid > 50:
                return "光伏售电"
            elif pv - load > 100:
                return "光伏剩余"
            elif pv - load < -100:
                return "电网购电"
            else:
                return "功率平衡"
        else:
            return "系统待机"
    
    def plot_battery_three_states(self, df):
        """专门绘制电池三态分析图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        hours = df['hour']
        
        # 1. 电池状态时间线
        state_colors = {'充电': COLOR_THEME['charge'], '放电': COLOR_THEME['discharge'], '待机': COLOR_THEME['idle']}
        
        for i, (hour, state) in enumerate(zip(hours, df['battery_state'])):
            color = state_colors.get(state, 'white')
            rect = Rectangle((hour-0.4, 0), 0.8, 1, 
                           facecolor=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax1.add_patch(rect)
            ax1.text(hour, 0.5, state, ha='center', va='center', 
                    fontsize=10, weight='bold')
        
        ax1.set_xlim(-0.5, 23.5)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('时间 (小时)')
        ax1.set_title('电池状态时间线', fontsize=14, fontweight='bold')
        ax1.set_yticks([])
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. SOC变化与状态
        ax2.plot(hours, df['soc'], color=COLOR_THEME['battery'], linewidth=3, label='SOC')
        ax2.axhline(y=self.Soc_min*100, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=self.Soc_max*100, color='red', linestyle='--', alpha=0.7)
        
        # 根据状态着色背景
        for i in range(len(hours)-1):
            if df.iloc[i]['battery_state'] == '充电':
                ax2.axvspan(hours[i]-0.5, hours[i]+0.5, alpha=0.2, color=COLOR_THEME['charge'])
            elif df.iloc[i]['battery_state'] == '放电':
                ax2.axvspan(hours[i]-0.5, hours[i]+0.5, alpha=0.2, color=COLOR_THEME['discharge'])
            else:
                ax2.axvspan(hours[i]-0.5, hours[i]+0.5, alpha=0.1, color=COLOR_THEME['idle'])
        
        ax2.set_title('SOC变化与电池状态', fontsize=14, fontweight='bold')
        ax2.set_ylabel('SOC (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 状态分布饼图
        state_counts = df['battery_state'].value_counts()
        colors_pie = [COLOR_THEME['charge'], COLOR_THEME['discharge'], COLOR_THEME['idle']]
        wedges, texts, autotexts = ax3.pie(state_counts.values, labels=state_counts.index, 
                                          autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax3.set_title('电池状态分布', fontsize=14, fontweight='bold')
        
        # 4. 充放电功率对比
        charge_power = df['bat_charge'] / 1000
        discharge_power = df['bat_discharge'] / 1000
        
        ax4.bar(hours, charge_power, alpha=0.8, color=COLOR_THEME['charge'], 
               width=0.4, label='充电功率(kW)')
        ax4.bar(hours, -discharge_power, alpha=0.8, color=COLOR_THEME['discharge'], 
               width=0.4, label='放电功率(kW)')
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        ax4.set_title('电池充放电功率', fontsize=14, fontweight='bold')
        ax4.set_ylabel('功率 (kW)')
        ax4.set_xlabel('时间 (小时)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(VIS_DIR, "1.png"))
    
    def analyze_battery_behavior(self, df):
        """分析电池行为模式"""
        print("\n" + "="*50)
        print("🔋 电池三态行为分析")
        print("="*50)
        
        # 状态统计
        state_counts = df['battery_state'].value_counts()
        total_hours = len(df)
        
        print("状态分布:")
        for state, count in state_counts.items():
            percentage = count / total_hours * 100
            print(f"  {state}: {count}小时 ({percentage:.1f}%)")
        
        # 充放电统计
        charging_hours = df[df['battery_state'] == '充电']
        discharging_hours = df[df['battery_state'] == '放电']
        idle_hours = df[df['battery_state'] == '待机']
        
        print(f"\n充电行为:")
        if len(charging_hours) > 0:
            print(f"  充电时段: {charging_hours['hour'].tolist()}")
            print(f"  平均充电功率: {charging_hours['bat_charge'].mean():.1f}W")
            print(f"  最大充电功率: {charging_hours['bat_charge'].max():.1f}W")
            print(f"  总充电量: {charging_hours['bat_charge'].sum()/1000:.2f}kWh")
        
        print(f"\n放电行为:")
        if len(discharging_hours) > 0:
            print(f"  放电时段: {discharging_hours['hour'].tolist()}")
            print(f"  平均放电功率: {discharging_hours['bat_discharge'].mean():.1f}W")
            print(f"  最大放电功率: {discharging_hours['bat_discharge'].max():.1f}W")
            print(f"  总放电量: {discharging_hours['bat_discharge'].sum()/1000:.2f}kWh")
        
        print(f"\n待机行为:")
        if len(idle_hours) > 0:
            print(f"  待机时段: {idle_hours['hour'].tolist()}")
            print(f"  待机时平均SOC: {idle_hours['soc'].mean():.1f}%")
        
        # SOC变化分析
        soc_change = df['soc'].iloc[-1] - df['soc'].iloc[0]
        print(f"\nSOC变化:")
        print(f"  初始SOC: {df['soc'].iloc[0]:.1f}%")
        print(f"  最终SOC: {df['soc'].iloc[-1]:.1f}%")
        print(f"  净变化: {soc_change:+.1f}%")
        print(f"  SOC范围: {df['soc'].min():.1f}% - {df['soc'].max():.1f}%")
    
    def create_enhanced_report_table(self, df, VIS_DIR):
        """创建增强的报告表格（包含三态信息）"""
        # 选择关键列用于显示 - 添加真实数据列和功率流向列
        display_cols = ['hour', 'pv_generation', 'real_pv_generation', 'load_demand', 'real_load_demand',
                       'buy_price', 'sell_price', 
                       # 功率流向详细信息
                       'pv_to_load', 'pv_to_bat', 'pv_to_grid',
                       'grid_to_load', 'grid_to_bat', 
                       'bat_to_load', 'bat_to_grid',
                       # 电池综合功率
                       'bat_charge', 'bat_discharge', 'soc', 'battery_state', 
                       'main_operation', 'hourly_profit', 'cumulative_profit']
        
        # 只包含存在的列
        available_cols = [col for col in display_cols if col in df.columns]
        
        # 重命名列
        col_names = {
            'hour': '时间',
            'pv_generation': '预测光伏(W)',
            'real_pv_generation': '实际光伏(W)',
            'load_demand': '预测负载(W)',
            'real_load_demand': '实际负载(W)', 
            'buy_price': '购电价',
            'sell_price': '售电价',
            # 功率流向列
            'pv_to_load': '光伏→负载(W)',
            'pv_to_bat': '光伏→电池(W)',
            'pv_to_grid': '光伏→电网(W)',
            'grid_to_load': '电网→负载(W)',
            'grid_to_bat': '电网→电池(W)',
            'bat_to_load': '电池→负载(W)',
            'bat_to_grid': '电池→电网(W)',
            # 电池综合功率
            'bat_charge': '总充电(W)',
            'bat_discharge': '总放电(W)',
            'soc': 'SOC(%)',
            'battery_state': '电池状态',
            'main_operation': '主要操作',
            'hourly_profit': '小时收益(元)',
            'cumulative_profit': '累计收益(元)'
        }
        
        # 格式化数据
        report_df = df[available_cols].copy()
        report_df['hourly_profit'] = report_df['hourly_profit'].round(3)
        report_df['cumulative_profit'] = report_df['cumulative_profit'].round(3)
        report_df['soc'] = report_df['soc'].round(1)
        
        # 格式化真实数据列（如果存在）
        if 'real_pv_generation' in report_df.columns:
            report_df['real_pv_generation'] = report_df['real_pv_generation'].round(0)
        if 'real_load_demand' in report_df.columns:
            report_df['real_load_demand'] = report_df['real_load_demand'].round(0)
            
        # 格式化功率流向列（如果存在）- 保留整数瓦特
        power_flow_cols = ['pv_to_load', 'pv_to_bat', 'pv_to_grid', 
                          'grid_to_load', 'grid_to_bat', 'bat_to_load', 'bat_to_grid']
        for col in power_flow_cols:
            if col in report_df.columns:
                report_df[col] = report_df[col].round(0)
            
        report_df = report_df.rename(columns=col_names)
        
        # 创建表格图
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.axis('tight')
        ax.axis('off')
        
        # 创建表格
        table_data = [report_df.columns.tolist()] + report_df.values.tolist()
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                        cellLoc='center', loc='center')
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # 着色
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i == 0:  # 表头
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    # 根据电池状态着色
                    if j == 8:  # 电池状态列
                        state = table_data[i][j]
                        if state == '充电':
                            cell.set_facecolor('#E3F2FD')
                        elif state == '放电':
                            cell.set_facecolor('#FFEBEE')
                        else:  # 待机
                            cell.set_facecolor('#F5F5F5')
                    # 根据收益着色
                    elif j == len(table_data[0]) - 2:  # 小时收益列
                        try:
                            value = float(table_data[i][j])
                            if value > 0:
                                cell.set_facecolor('#E8F5E8')
                            elif value < 0:
                                cell.set_facecolor('#FFE8E8')
                            else:
                                cell.set_facecolor('white')
                        except:
                            cell.set_facecolor('white')
                    else:
                        cell.set_facecolor('#f9f9f9' if i % 2 == 0 else 'white')
        
        plt.title('24小时详细运行报告（含电池三态）', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(VIS_DIR, "2.png"))
        
        return report_df
    
    # 保持原有的其他可视化方法...
    def plot_comprehensive_analysis(self, df, result):
        """综合分析图表（略作修改以显示三态）"""
        # 这里可以保持大部分原有代码，只需在电池状态部分做小调整
        if result['status'] != 'optimal':
            return
            
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 1, 0.8], 
                             width_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.3)
        
        hours = df['hour']
        
        # 1. 功率平衡总览（保持不变）
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(hours, df['pv_generation'], 'orange', linewidth=3, label='光伏发电')
        ax1.plot(hours, df['load_demand'], 'blue', linewidth=3, label='负载需求')
        ax1.fill_between(hours, 0, df['pv_generation'], alpha=0.3, color='orange')
        ax1.fill_between(hours, 0, df['load_demand'], alpha=0.3, color='blue')
        ax1.set_title('功率平衡总览', fontsize=14, fontweight='bold')
        ax1.set_ylabel('功率 (W)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 电池状态（修改为显示三态）
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.plot(hours, df['soc'], 'green', linewidth=3, label='SOC')
        ax2.axhline(y=self.Soc_min*100, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=self.Soc_max*100, color='red', linestyle='--', alpha=0.7)
        
        # 根据电池状态着色背景
        for i in range(len(hours)-1):
            if df.iloc[i]['battery_state'] == '充电':
                ax2.axvspan(hours[i]-0.5, hours[i]+0.5, alpha=0.2, color='blue', label='充电' if i==0 else "")
            elif df.iloc[i]['battery_state'] == '放电':
                ax2.axvspan(hours[i]-0.5, hours[i]+0.5, alpha=0.2, color='red', label='放电' if i==0 else "")
            else:
                ax2.axvspan(hours[i]-0.5, hours[i]+0.5, alpha=0.1, color='gray', label='待机' if i==0 else "")
        
        ax2.set_title('电池SOC与状态', fontsize=14, fontweight='bold')
        ax2.set_ylabel('SOC (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 其他图表保持原有逻辑...
        # 这里可以复制之前的代码，或者调用原有方法
        
        plt.suptitle('家庭能源管理系统 - 24小时运行分析（电池三态）', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        #plt.show()
        plt.savefig(os.path.join(VIS_DIR, "3.png"))

    def plot_energy_flow_sankey(self, df):
        """绘制能量流向桑基图（每小时）"""
        import plotly.graph_objects as go
        
        fig = make_subplots(
            rows=4, cols=6,
            subplot_titles=[f'{h}时' for h in range(24)],
            specs=[[{"type": "sankey"}]*6 for _ in range(4)],
            vertical_spacing=0.05,
            horizontal_spacing=0.02
        )
        
        # 为每个小时创建简化的桑基图
        for hour in range(24):
            row = hour // 6 + 1
            col = hour % 6 + 1
            
            hour_data = df.iloc[hour]
            
            # 定义节点
            nodes = ["光伏", "电网", "电池", "负载"]
            node_colors = ["orange", "gray", "green", "blue"]
            
            # 定义流向和流量
            sources = []
            targets = []
            values = []
            
            # 光伏到负载
            if hour_data['pv_to_load'] > 10:
                sources.append(0)  # 光伏
                targets.append(3)  # 负载
                values.append(hour_data['pv_to_load'])
            
            # 光伏到电池
            if hour_data['pv_to_bat'] > 10:
                sources.append(0)  # 光伏
                targets.append(2)  # 电池
                values.append(hour_data['pv_to_bat'])
                
            # 光伏到电网
            if hour_data['pv_to_grid'] > 10:
                sources.append(0)  # 光伏
                targets.append(1)  # 电网
                values.append(hour_data['pv_to_grid'])
            
            # 电网到负载
            if hour_data['grid_to_load'] > 10:
                sources.append(1)  # 电网
                targets.append(3)  # 负载
                values.append(hour_data['grid_to_load'])
                
            # 电池到负载
            if hour_data['bat_to_load'] > 10:
                sources.append(2)  # 电池
                targets.append(3)  # 负载
                values.append(hour_data['bat_to_load'])
            
            # 如果有流量数据，创建桑基图
            if sources and targets and values:
                fig.add_trace(
                    go.Sankey(
                        node=dict(
                            pad=5,
                            thickness=10,
                            line=dict(color="black", width=0.5),
                            label=nodes,
                            color=node_colors
                        ),
                        link=dict(
                            source=sources,
                            target=targets,
                            value=values,
                            color="rgba(255,255,255,0.4)"
                        )
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title_text="24小时能量流向桑基图",
            font_size=10,
            height=1000,
            showlegend=False
        )
        
        fig.write_html(os.path.join(VIS_DIR, "energy_flow_sankey.html"))
        print("桑基图已保存为 vis/energy_flow_sankey.html")

    def plot_energy_flow_animated(self, df):
        """绘制能量流向动态图"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 定义组件位置
        positions = {
            '光伏': (2, 6),
            '电池': (2, 3),
            '负载': (6, 4.5),
            '电网': (10, 4.5)
        }
        
        # 绘制组件
        components = {}
        colors = {'光伏': 'orange', '电池': 'green', '负载': 'blue', '电网': 'gray'}
        
        for comp, pos in positions.items():
            circle = plt.Circle(pos, 0.8, color=colors[comp], alpha=0.7)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], comp, ha='center', va='center', fontsize=12, fontweight='bold')
            components[comp] = circle
        
        def animate(frame):
            ax.clear()
            
            # 重新绘制组件
            for comp, pos in positions.items():
                circle = plt.Circle(pos, 0.8, color=colors[comp], alpha=0.7)
                ax.add_patch(circle)
                ax.text(pos[0], pos[1], comp, ha='center', va='center', fontsize=12, fontweight='bold')
            
            # 获取当前小时数据
            hour_data = df.iloc[frame]
            
            # 绘制能量流箭头，粗细代表流量大小
            flows = [
                ('光伏', '负载', hour_data['pv_to_load'], 'orange'),
                ('光伏', '电池', hour_data['pv_to_bat'], 'orange'),
                ('光伏', '电网', hour_data['pv_to_grid'], 'orange'),
                ('电网', '负载', hour_data['grid_to_load'], 'gray'),
                ('电池', '负载', hour_data['bat_to_load'], 'green'),
            ]
            
            for source, target, flow, color in flows:
                if flow > 50:  # 只显示显著的能量流
                    start = positions[source]
                    end = positions[target]
                    
                    # 计算箭头粗细 (flow越大越粗)
                    width = max(0.5, min(10, flow / 200))
                    
                    # 绘制箭头
                    arrow = plt.annotate('', xy=end, xytext=start,
                                       arrowprops=dict(arrowstyle='->', lw=width, color=color, alpha=0.8))
                    
                    # 添加流量标签
                    mid_x = (start[0] + end[0]) / 2
                    mid_y = (start[1] + end[1]) / 2 + 0.3
                    ax.text(mid_x, mid_y, f'{flow:.0f}W', ha='center', va='center', 
                           fontsize=8, bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
            
            # 显示当前时间和电池状态
            ax.text(6, 7, f'时间: {frame}:00', fontsize=16, fontweight='bold')
            ax.text(6, 6.5, f'电池状态: {hour_data["battery_state"]}', fontsize=14)
            ax.text(6, 6, f'SOC: {hour_data["soc"]:.1f}%', fontsize=14)
            
            ax.set_xlim(0, 12)
            ax.set_ylim(1, 8)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title('能量流向动态变化', fontsize=16, fontweight='bold')
        
        # 创建动画
        anim = animation.FuncAnimation(fig, animate, frames=24, interval=1000, repeat=True)
        
        # 保存动画
        anim.save(os.path.join(VIS_DIR, 'energy_flow_animation.gif'), writer='pillow', fps=1)
        plt.savefig(os.path.join(VIS_DIR, 'energy_flow_static.png'))
        plt.close()
        print("能量流向动画已保存为 vis/energy_flow_animation.gif")

    def plot_energy_flow_heatmap(self, df):
        """绘制能量流向热图矩阵"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 提取各种能量流数据
        flows = {
            '光伏→负载': df['pv_to_load'],
            '光伏→电池': df['pv_to_bat'], 
            '光伏→电网': df['pv_to_grid'],
            '电网→负载': df['grid_to_load'],
            '电池→负载': df['bat_to_load'],
            '电池状态': df['state_code'] * 1000  # 放大显示
        }
        
        axes = axes.flatten()
        
        for idx, (flow_name, flow_data) in enumerate(flows.items()):
            # 创建热图数据矩阵 (24小时 x 1)
            heatmap_data = flow_data.values.reshape(24, 1)
            
            im = axes[idx].imshow(heatmap_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
            
            # 设置标签
            axes[idx].set_title(f'{flow_name}', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('时间 (小时)')
            axes[idx].set_xticks(range(0, 24, 4))
            axes[idx].set_xticklabels(range(0, 24, 4))
            axes[idx].set_yticks([])
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label('功率 (W)' if '状态' not in flow_name else '状态', rotation=270, labelpad=15)
            
            # 在每个小时位置显示数值
            for hour in range(24):
                value = flow_data.iloc[hour]
                color = 'white' if value > flow_data.max() * 0.5 else 'black'
                axes[idx].text(hour, 0, f'{value:.0f}', ha='center', va='center', 
                             color=color, fontsize=8)
        
        plt.suptitle('24小时能量流向热图矩阵', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, 'energy_flow_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("能量流向热图已保存为 vis/energy_flow_heatmap.png")

    def plot_energy_waterfall(self, df):
        """绘制能量平衡瀑布图"""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 计算每小时的净能量变化
        hours = df['hour']
        
        # 能量来源（正值）
        pv_gen = df['pv_generation'] 
        bat_discharge = df['bat_discharge']
        grid_purchase = df['grid_to_load'] + df['grid_to_bat']  # 电网购电 = 电网到负载 + 电网到电池
        
        # 能量消耗（负值）
        load_consume = -df['load_demand']
        bat_charge = -df['bat_charge']
        grid_export = -(df['pv_to_grid'] + df['bat_to_grid'])  # 电网吸纳 = 光伏售电 + 电池售电
        
        # 创建堆叠条形图
        width = 0.6
        
        # 正值（能量输入）
        ax.bar(hours, pv_gen, width, label='光伏发电', color=COLOR_THEME['pv'], alpha=0.8)
        ax.bar(hours, bat_discharge, width, bottom=pv_gen, label='电池放电', color=COLOR_THEME['discharge'], alpha=0.8)
        ax.bar(hours, grid_purchase, width, bottom=pv_gen+bat_discharge, label='电网购电', color=COLOR_THEME['grid'], alpha=0.8)
        
        # 负值（能量输出）
        ax.bar(hours, load_consume, width, label='负载消耗', color=COLOR_THEME['load'], alpha=0.8)
        ax.bar(hours, bat_charge, width, bottom=load_consume, label='电池充电', color=COLOR_THEME['charge'], alpha=0.8)
        ax.bar(hours, grid_export, width, bottom=load_consume+bat_charge, label='电网吸纳', color='lightgray', alpha=0.8)
        
        # 添加零线
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # 设置标签和标题
        ax.set_xlabel('时间 (小时)', fontsize=12)
        ax.set_ylabel('功率 (W)', fontsize=12)
        ax.set_title('24小时能量平衡瀑布图', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # 在每个bar的每个组成部分上添加数值标签
        for hour in range(24):
            # 正值部分的标签（能量输入）
            # 光伏发电标签
            if pv_gen.iloc[hour] > 50:  # 只显示显著的值
                ax.text(hour, pv_gen.iloc[hour]/2, f'{pv_gen.iloc[hour]:.0f}', 
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            # 电池放电标签
            if bat_discharge.iloc[hour] > 50:
                y_pos = pv_gen.iloc[hour] + bat_discharge.iloc[hour]/2
                ax.text(hour, y_pos, f'{bat_discharge.iloc[hour]:.0f}', 
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            # 电网购电标签
            if grid_purchase.iloc[hour] > 50:
                y_pos = pv_gen.iloc[hour] + bat_discharge.iloc[hour] + grid_purchase.iloc[hour]/2
                ax.text(hour, y_pos, f'{grid_purchase.iloc[hour]:.0f}', 
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            # 负值部分的标签（能量输出）
            # 负载消耗标签
            if abs(load_consume.iloc[hour]) > 50:
                ax.text(hour, load_consume.iloc[hour]/2, f'{abs(load_consume.iloc[hour]):.0f}', 
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            # 电池充电标签
            if abs(bat_charge.iloc[hour]) > 50:
                y_pos = load_consume.iloc[hour] + bat_charge.iloc[hour]/2
                ax.text(hour, y_pos, f'{abs(bat_charge.iloc[hour]):.0f}', 
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            # 电网吸纳标签
            if abs(grid_export.iloc[hour]) > 50:
                y_pos = load_consume.iloc[hour] + bat_charge.iloc[hour] + grid_export.iloc[hour]/2
                ax.text(hour, y_pos, f'{abs(grid_export.iloc[hour]):.0f}', 
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            
            # 净值标签（在顶部）
            net_energy = (pv_gen.iloc[hour] + bat_discharge.iloc[hour] + grid_purchase.iloc[hour] + 
                         load_consume.iloc[hour] + bat_charge.iloc[hour] + grid_export.iloc[hour])
            
            energy_in = pv_gen.iloc[hour] + bat_discharge.iloc[hour] + grid_purchase.iloc[hour]
            energy_out = abs(load_consume.iloc[hour] + bat_charge.iloc[hour] + grid_export.iloc[hour])
            y_pos = max(energy_in, energy_out) + 150
            
            color = 'green' if net_energy >= 0 else 'red'
            ax.text(hour, y_pos, f'净:{net_energy:.0f}', ha='center', va='bottom', 
                   color=color, fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, 'energy_waterfall.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("能量平衡瀑布图已保存为 vis/energy_waterfall.png")

    def plot_comprehensive_energy_dashboard(self, df):
        """创建综合能量仪表板"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. 能量流向概览 (占据上方2x4空间)
        ax_main = fig.add_subplot(gs[0:2, :])
        
        hours = df['hour']
        
        # 绘制能量流向堆叠面积图
        ax_main.fill_between(hours, 0, df['pv_generation'], alpha=0.7, color='orange', label='光伏发电')
        ax_main.fill_between(hours, df['pv_generation'], 
                           df['pv_generation'] + df['bat_discharge'], 
                           alpha=0.7, color='lightgreen', label='电池放电')
        
        # 负载需求线
        ax_main.plot(hours, df['load_demand'], color='blue', linewidth=3, label='负载需求')
        
        # 电池充电（负值显示）
        ax_main.fill_between(hours, 0, -df['bat_charge'], alpha=0.7, color='green', label='电池充电')
        
        ax_main.set_title('24小时综合能量流向仪表板', fontsize=18, fontweight='bold')
        ax_main.set_ylabel('功率 (W)', fontsize=12)
        ax_main.legend(loc='upper right')
        ax_main.grid(True, alpha=0.3)
        
        # 2. SOC变化趋势 (左下)
        ax_soc = fig.add_subplot(gs[2, :2])
        ax_soc.plot(hours, df['soc'], color='green', linewidth=3, marker='o', markersize=4)
        ax_soc.fill_between(hours, df['soc'], alpha=0.3, color='green')
        ax_soc.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% SOC')
        ax_soc.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90% SOC')
        ax_soc.set_title('电池SOC变化', fontsize=14, fontweight='bold')
        ax_soc.set_ylabel('SOC (%)')
        ax_soc.legend()
        ax_soc.grid(True, alpha=0.3)
        
        # 3. 电价与收益 (右下)
        ax_price = fig.add_subplot(gs[2, 2:])
        ax_price_twin = ax_price.twinx()
        
        # 电价
        ax_price.step(hours, df['buy_price'], where='mid', color='red', linewidth=2, label='购电价')
        ax_price.step(hours, df['sell_price'], where='mid', color='orange', linewidth=2, label='售电价')
        
        # 累计收益
        ax_price_twin.plot(hours, df['cumulative_profit'], color='green', linewidth=3, label='累计收益')
        
        ax_price.set_title('电价与收益变化', fontsize=14, fontweight='bold')
        ax_price.set_ylabel('电价 (元/kWh)')
        ax_price_twin.set_ylabel('累计收益 (元)')
        ax_price.legend(loc='upper left')
        ax_price_twin.legend(loc='upper right')
        ax_price.grid(True, alpha=0.3)
        
        # 4. 关键指标汇总 (底部)
        ax_summary = fig.add_subplot(gs[3, :])
        
        # 计算关键指标
        total_pv = df['pv_generation'].sum() / 1000  # kWh
        total_load = df['load_demand'].sum() / 1000
        total_charge = df['bat_charge'].sum() / 1000
        total_discharge = df['bat_discharge'].sum() / 1000
        self_consumption = (df['pv_to_load'].sum()) / max(df['pv_generation'].sum(), 1) * 100
        
        indicators = [
            f'光伏总发电: {total_pv:.2f} kWh',
            f'负载总需求: {total_load:.2f} kWh', 
            f'电池充电: {total_charge:.2f} kWh',
            f'电池放电: {total_discharge:.2f} kWh',
            f'自发自用率: {self_consumption:.1f}%',
            f'总收益: {df["cumulative_profit"].iloc[-1]:.3f} 元'
        ]
        
        # 绘制指标条
        y_pos = 0.5
        for i, indicator in enumerate(indicators):
            color = plt.cm.Set3(i)
            ax_summary.barh(y_pos - i*0.15, 1, height=0.1, color=color, alpha=0.7)
            ax_summary.text(0.5, y_pos - i*0.15, indicator, ha='center', va='center', 
                          fontsize=12, fontweight='bold')
        
        ax_summary.set_xlim(0, 1)
        ax_summary.set_ylim(-1, 1)
        ax_summary.set_title('关键指标汇总', fontsize=14, fontweight='bold')
        ax_summary.axis('off')
        
        plt.savefig(os.path.join(VIS_DIR, 'comprehensive_energy_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("综合能量仪表板已保存为 vis/comprehensive_energy_dashboard.png")


def test(info):
    try:
        start_time = time.time()
        gateway_id = info["gateway_id"]
        date = info["date"]
        _rated_cap = info["rated_cap"]
        _soc_min = info["soc_min"]
        _curr_soc = info["curr_soc"]
        _rated_power = info["rated_power"]
        pv_pred = info["pv_pred"]
        assert(float("nan") not in pv_pred)
        pv = info["pv"]
        assert(float("nan") not in pv)
        load_pred = info["load_pred"]
        assert(float("nan") not in load_pred)
        load = info["load"]
        assert(float("nan") not in load)
        buy_prices = info["buy_prices"]
        assert(float("nan") not in buy_prices)
        sell_prices = info["sell_prices"]
        assert(float("nan") not in sell_prices)
        load_priority = info["load_priority"]
        solar_priority = info["solar_priority"]
        grid_charge_max = info["grid_charge_max"]
        grid_discharge_max = info["grid_discharge_max"]
        code = info["code"]

        initial_soc = _curr_soc

        ems = ThreeStateEnergyManagementSystem(
            battery_capacity=_rated_cap, 
            max_power=_rated_power,
            charge_efficiency=0.9, 
            discharge_efficiency=0.9, 
            grid_charge_max=_rated_power, 
            grid_discharge_max=_rated_power, 
            soc_min=0.2, 
            soc_max=0.95
        )

        # 线性规划-真实值
        """
        VIS_DIR = 'vis/gateway_id:{}-date:{}-lp-grid-constrain-pred'.format(gateway_id, date)
        if not os.path.exists(VIS_DIR):
            os.makedirs(VIS_DIR)

        if not os.path.exists(os.path.join(VIS_DIR, "2.png")) or not os.path.exists(os.path.join(VIS_DIR, "res.json")):
            result = ems.optimize_complete_model(
                pv_pred, 
                load_pred, 
                buy_prices, 
                sell_prices, 
                initial_soc=initial_soc)
            json_result = convert_to_json_serializable(result)
            json_result['input_data'] = {
                'pv_generation': pv_pred,
                'load_demand': load_pred,
                'real_pv': pv,
                'real_load': load,
                'buy_prices': buy_prices,
                'sell_prices': sell_prices,
                'code': code,
                "load_priority": load_priority,
                "solar_priority": solar_priority,
                'initial_soc': initial_soc,
                'battery_config': {
                    'capacity': ems.E_bat,
                    'max_power': ems.P_bat_max,
                    'charge_efficiency': ems.eta_c,
                    'discharge_efficiency': ems.eta_d,
                    'soc_min': ems.Soc_min,
                    'soc_max': ems.Soc_max,
                    'grid_charge_max': ems.grid_charge_max,
                    'grid_discharge_max': ems.grid_discharge_max
                }
            }
            with open(os.path.join(VIS_DIR, 'res.json'), 'w', encoding='utf-8') as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
        
            df = ems.create_hourly_analysis(
                result, 
                pv_pred, 
                load_pred, 
                buy_prices, 
                sell_prices,
                pv, load
            )
            report_table = ems.create_enhanced_report_table(df, VIS_DIR)
            """

        # rul-真实值
        
        VIS_DIR = 'vis/gateway_id:{}-date:{}-hier-mpc-grid-constrain-pv100-load100-rule-pred'.format(gateway_id, date)
        if not os.path.exists(VIS_DIR):
            os.makedirs(VIS_DIR)

        if not os.path.exists(os.path.join(VIS_DIR, "2.png")) or not os.path.exists(os.path.join(VIS_DIR, "res.json")):
            result = ems.rule_based(
                code,
                solar_priority, 
                load_priority,
                grid_charge_max,
                grid_discharge_max,
                pv, 
                load, 
                buy_prices, 
                sell_prices,
                use_optimizeed_gh=False, 
                initial_soc=initial_soc
            )
            json_result = convert_to_json_serializable(result)
            json_result['input_data'] = {
                'pv_generation': pv_pred,
                'load_demand': load_pred,
                'real_pv': pv,
                'real_load': load,
                'buy_prices': buy_prices,
                'sell_prices': sell_prices,
                'code': code,
                "load_priority": load_priority,
                "solar_priority": solar_priority,
                'initial_soc': initial_soc,
                'battery_config': {
                    'capacity': ems.E_bat,
                    'max_power': ems.P_bat_max,
                    'charge_efficiency': ems.eta_c,
                    'discharge_efficiency': ems.eta_d,
                    'soc_min': ems.Soc_min,
                    'soc_max': ems.Soc_max,
                    'grid_charge_max': ems.grid_charge_max,
                    'grid_discharge_max': ems.grid_discharge_max
                }
            }
            with open(os.path.join(VIS_DIR, 'res.json'), 'w', encoding='utf-8') as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            """
            df = ems.create_hourly_analysis(
                result, 
                pv_pred, 
                load_pred, 
                buy_prices, 
                sell_prices,
                pv, load
            )
            report_table = ems.create_enhanced_report_table(df, VIS_DIR)
            """
        # dp-真实值
        """
        VIS_DIR = 'vis/gateway_id:{}-date:{}-dp'.format(gateway_id, date)
        if not os.path.exists(VIS_DIR):
            os.makedirs(VIS_DIR)
        
        if not os.path.exists(os.path.join(VIS_DIR, "2.png")) or not os.path.exists(os.path.join(VIS_DIR, "res.json")):
            result = ems.dynamic_programming(
                pv, 
                load, 
                buy_prices, 
                sell_prices, 
                initial_soc=initial_soc
            )
            json_result = convert_to_json_serializable(result)
            json_result['input_data'] = {
                'pv_generation': pv_pred,
                'load_demand': load_pred,
                'real_pv': pv,
                'real_load': load,
                'buy_prices': buy_prices,
                'sell_prices': sell_prices,
                'code': code,
                "load_priority": load_priority,
                "solar_priority": solar_priority,
                'initial_soc': initial_soc,
                'battery_config': {
                    'capacity': ems.E_bat,
                    'max_power': ems.P_bat_max,
                    'charge_efficiency': ems.eta_c,
                    'discharge_efficiency': ems.eta_d,
                    'soc_min': ems.Soc_min,
                    'soc_max': ems.Soc_max,
                    'grid_charge_max': ems.grid_charge_max,
                    'grid_discharge_max': ems.grid_discharge_max
                }
            }
            with open(os.path.join(VIS_DIR, 'res.json'), 'w', encoding='utf-8') as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            
            df = ems.create_hourly_analysis(
                result, 
                pv_pred, 
                load_pred, 
                buy_prices, 
                sell_prices,
                pv, load
            )
            report_table = ems.create_enhanced_report_table(df, VIS_DIR)
        """
        end_time = time.time()
        print("process {}-{}, time cost: {}...".format(gateway_id, date, end_time-start_time))
    except Exception as e:
        print(e)
    
    
if __name__ == "__main__":
    import time
    with open("test_samples.json", "r") as f:
        info_list = json.load(f)
    
    info_list_new = []
    for info in tqdm.tqdm(info_list):
        try:
            gateway_id = info["gateway_id"]
            date = info["date"]
            with open("vis/gateway_id:{}-date:{}-hier-mpc-grid-constrain-pv100-load100/rule.json".format(gateway_id, date), "r", encoding='utf-8') as f:
                pred_rule = json.load(f)
            info["load_priority"] = pred_rule["rule_based_format"]["load_priority"]
            info["solar_priority"] = pred_rule["rule_based_format"]["solar_priority"]
            info_list_new.append(info)
        except Exception as e:
            print(e)

    print(len(info_list_new))
    """
    start_t = time.time()
    for info in tqdm.tqdm(info_list):
        if info["gateway_id"] == "02cc7831705452271ac803b50e3f1562" and info["date"] == "2024-09-17":
            print(info["gateway_id"], info["date"])
            test(info)
    print("time: ", time.time() - start_t)
    """ 

    cpu_count = os.cpu_count()
    n=0
    futures = []
    start_t = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(cpu_count*0.95)) as executor:
        #for info in info_list:
        #    func = partial(test, info)
        #    executor.submit(func)
        results = list(executor.map(test, info_list_new))
    print("time: ", time.time() - start_t)
    
