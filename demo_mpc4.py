import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import traceback
import time
import tqdm
from utils import *

# 继承原有的ThreeStateEnergyManagementSystem
import demo
from demo import ThreeStateEnergyManagementSystem, convert_to_json_serializable, COLOR_THEME

class HierarchicalMPCEnergyManagementSystem(ThreeStateEnergyManagementSystem):
    """
    分层MPC能源管理系统
    
    分层策略：
    1. 上层：全局规划层 - 基于预测数据进行24小时全局优化，生成SOC目标轨迹
    2. 下层：MPC跟踪层 - 使用短时域MPC跟踪目标SOC轨迹，同时处理实时扰动
    
    核心思想：
    - 避免滚动窗口MPC的短视问题
    - 保持实时响应能力
    - 结合全局最优性与局部适应性
    """
    
    def __init__(self, 
                 prediction_horizon=4,     # MPC预测时域（小时）
                 control_horizon=1,        # MPC控制时域（小时）
                 soc_tracking_weight=10.0, # SOC跟踪权重
                 cost_weight=1.0,          # 成本权重
                 planning_frequency=24,    # 全局规划频率（小时）
                 **kwargs):
        """
        初始化分层MPC系统
        
        Args:
            prediction_horizon: MPC预测时域长度
            control_horizon: MPC控制时域长度  
            soc_tracking_weight: SOC轨迹跟踪权重
            cost_weight: 电力成本权重
            planning_frequency: 全局规划更新频率
            **kwargs: 继承参数
        """
        super().__init__(**kwargs)
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon
        self.soc_tracking_weight = soc_tracking_weight
        self.cost_weight = cost_weight
        self.planning_frequency = planning_frequency
        
        # 分层MPC状态
        self.global_plan = None
        self.target_soc_trajectory = None
        self.execution_history = []
        self.planning_history = []
        
        # print(f"🏗️ 分层MPC系统初始化完成")
        # print(f"   全局规划频率: {planning_frequency}小时")
        # print(f"   MPC预测时域: {prediction_horizon}小时")
        # print(f"   SOC跟踪权重: {soc_tracking_weight}")

    def hierarchical_mpc_optimize(self, pv_generation, load_demand, buy_prices, sell_prices,
                                 real_pv=None, real_load=None, initial_soc=0.5):
        """
        分层MPC主优化函数
        
        分两个层次执行：
        1. 全局规划层：制定24小时SOC目标轨迹
        2. MPC跟踪层：逐小时跟踪目标轨迹
        """
        
        T = len(pv_generation)
        # print(f"🎯 开始分层MPC优化 - {T}小时窗口")
        
        # === 第一层：全局规划层 ===
        # print("📋 Layer 1: 全局规划层 - 制定SOC目标轨迹")
        
        try:
            self.global_plan = self._global_planning_layer(
                pv_generation, load_demand, buy_prices, sell_prices, initial_soc
            )
            
            if self.global_plan['status'] != 'optimal':
                raise Exception(f"全局规划失败: {self.global_plan.get('message', '未知错误')}")
            
            # 提取目标SOC轨迹
            self.target_soc_trajectory = self.global_plan['battery']['SOC']
            
            # print(f"✅ 全局规划成功，预期成本: {self.global_plan['total_cost']:.3f}元")
            # print(f"   SOC轨迹: {self.target_soc_trajectory[0]:.1%} → {self.target_soc_trajectory[-1]:.1%}")
            
        except Exception as e:
            # print(f"❌ 全局规划失败: {e}")
            # 失败时使用线性SOC轨迹
            self.target_soc_trajectory = np.linspace(initial_soc, initial_soc, T+1)
            
        # === 第二层：MPC跟踪层 ===
        # print("⚡ Layer 2: MPC跟踪层 - 逐小时跟踪控制")
        
        # 初始化实际执行结果
        result = self._initialize_execution_result(T, initial_soc)
        
        current_soc = initial_soc
        total_reoptimizations = 0
        
        for t in range(T):
            # print(f"\n--- 执行时刻 {t} ---")
            
            # 获取实际数据
            actual_pv = real_pv[t] if real_pv and t < len(real_pv) else pv_generation[t]
            actual_load = real_load[t] if real_load and t < len(real_load) else load_demand[t]
            
            # 计算预测误差
            pv_error = abs(actual_pv - pv_generation[t]) / max(pv_generation[t], 1)
            load_error = abs(actual_load - load_demand[t]) / max(load_demand[t], 1)
            
            # print(f"  预测vs实际: PV {pv_generation[t]:.0f}→{actual_pv:.0f}W (误差{pv_error:.1%})")
            # print(f"  预测vs实际: Load {load_demand[t]:.0f}→{actual_load:.0f}W (误差{load_error:.1%})")
            # print(f"  目标SOC: {self.target_soc_trajectory[t]:.1%} → {self.target_soc_trajectory[t+1]:.1%}")
            
            # MPC跟踪控制
            step_result = self._mpc_tracking_layer(
                t, actual_pv, actual_load, current_soc,
                pv_generation, load_demand, buy_prices, sell_prices
            )
            
            # 更新实际执行结果
            self._update_execution_result(result, t, step_result)
            
            # 更新SOC
            current_soc = step_result['new_soc']
            
            # 记录执行历史
            self.execution_history.append({
                'time': t,
                'planned_pv': pv_generation[t],
                'actual_pv': actual_pv,
                'planned_load': load_demand[t],
                'actual_load': actual_load,
                'pv_error': pv_error,
                'load_error': load_error,
                'target_soc': self.target_soc_trajectory[t+1],
                'actual_soc': current_soc,
                'soc_tracking_error': abs(current_soc - self.target_soc_trajectory[t+1])
            })
            
            # print(f"  执行结果: SOC={current_soc:.1%}, 跟踪误差={abs(current_soc - self.target_soc_trajectory[t+1]):.1%}")
            # print(f"  成本: {step_result['step_cost']:.3f}元")
        
        # === 汇总最终结果 ===
        final_result = self._finalize_hierarchical_result(result, total_reoptimizations)
        
        # print(f"\n✅ 分层MPC执行完成！")
        # print(f"   实际总成本: {final_result['total_cost']:.3f}元")
        # print(f"   全局计划成本: {self.global_plan['total_cost']:.3f}元")
        # print(f"   平均SOC跟踪误差: {self._calculate_average_soc_tracking_error():.1%}")
        
        return final_result

    def _global_planning_layer(self, pv_generation, load_demand, buy_prices, sell_prices, initial_soc):
        """
        全局规划层：执行24小时全局优化
        
        目标：基于预测数据获得全局最优的SOC轨迹和功率分配策略
        这一层不考虑实时扰动，专注于全局最优性
        """
        
        # print("  执行24小时全局优化...")
        
        # 使用继承的完整模型进行全局优化
        try:
            global_result = self.optimize_complete_model(
                pv_generation, load_demand, buy_prices, sell_prices, initial_soc
            )
            
            # 存储规划历史
            self.planning_history.append({
                'timestamp': 0,  # 简化处理
                'method': 'global_optimization',
                'prediction_pv': pv_generation.copy(),
                'prediction_load': load_demand.copy(),
                'target_soc': global_result['battery']['SOC'] if global_result['status'] == 'optimal' else None,
                'expected_cost': global_result['total_cost'] if global_result['status'] == 'optimal' else None
            })
            
            return global_result
            
        except Exception as e:
            # print(f"  全局优化异常: {e}")
            return {'status': 'failed', 'message': str(e)}

    def _mpc_tracking_layer(self, current_time, actual_pv, actual_load, current_soc,
                           pv_prediction, load_prediction, buy_prices, sell_prices):
        """
        MPC跟踪层：短时域MPC跟踪目标SOC轨迹
        
        目标：
        1. 跟踪全局规划的SOC轨迹
        2. 处理实时扰动（实际vs预测的差异）
        3. 在跟踪性能和经济性之间平衡
        """
        # print("MPC tracking layer")
        # 确定预测窗口
        remaining_time = len(pv_prediction) - current_time
        horizon = min(self.prediction_horizon, remaining_time)
        
        if horizon <= 0:
            # 边界情况处理
            return self._emergency_control(actual_pv, actual_load, current_soc, 
                                         buy_prices[current_time], sell_prices[current_time])
        
        # 提取预测窗口数据
        pv_pred = np.zeros(horizon)
        load_pred = np.zeros(horizon) 
        buy_prices_pred = np.zeros(horizon)
        sell_prices_pred = np.zeros(horizon)
        target_soc_pred = np.zeros(horizon + 1)
        
        # 第一个时刻使用实际数据，后续使用预测数据
        pv_pred[0] = actual_pv
        load_pred[0] = actual_load
        buy_prices_pred[0] = buy_prices[current_time]
        sell_prices_pred[0] = sell_prices[current_time]
        target_soc_pred[0] = current_soc
        
        for i in range(1, horizon):
            if current_time + i < len(pv_prediction):
                pv_pred[i] = pv_prediction[current_time + i]
                load_pred[i] = load_prediction[current_time + i] 
                buy_prices_pred[i] = buy_prices[current_time + i]
                sell_prices_pred[i] = sell_prices[current_time + i]
        
        for i in range(horizon + 1):
            if current_time + i < len(self.target_soc_trajectory):
                target_soc_pred[i] = self.target_soc_trajectory[current_time + i]
            else:
                target_soc_pred[i] = target_soc_pred[i-1] if i > 0 else current_soc
        
        # 执行MPC优化
        mpc_result = self._solve_mpc_tracking_problem(
            horizon, pv_pred, load_pred, buy_prices_pred, sell_prices_pred,
            current_soc, target_soc_pred
        )
        # print("-----------------")
        if mpc_result['status'] != 'optimal':
            # print(f"  ⚠️ MPC求解失败，使用应急控制")
            return self._emergency_control(actual_pv, actual_load, current_soc,
                                         buy_prices[current_time], sell_prices[current_time])
        
        # 提取第一步的控制动作
        power_allocation = mpc_result['power_allocation'][0]
        
        # 计算成本
        step_cost = (power_allocation['P_purchase'] * buy_prices[current_time] - 
                    power_allocation['P_sell'] * sell_prices[current_time]) / 1000
        
        # 更新SOC
        energy_change = (power_allocation['P_bat_ch'] * self.eta_c - 
                        power_allocation['P_bat_dis'] / self.eta_d) / self.E_bat
        new_soc = max(self.Soc_min, min(self.Soc_max, current_soc + energy_change))
        
        return {
            'power_allocation': power_allocation,
            'step_cost': step_cost,
            'new_soc': new_soc,
            'mpc_status': mpc_result['status'],
            'soc_tracking_error': abs(new_soc - target_soc_pred[1]),
            'predicted_trajectory': mpc_result.get('soc_trajectory', [])
        }

    def _solve_mpc_tracking_problem(self, horizon, pv_pred, load_pred, buy_prices_pred, sell_prices_pred,
                                   current_soc, target_soc_pred):
        """
        求解MPC跟踪问题
        
        目标函数：成本最小化 + SOC轨迹跟踪
        """
        # print("MPC tracking layer solver")
        # === 变量定义 ===
        # 功率流变量
        P_pv_load = cp.Variable(horizon, nonneg=True)
        P_pv_bat = cp.Variable(horizon, nonneg=True)
        P_pv_grid = cp.Variable(horizon, nonneg=True)
        P_grid_load = cp.Variable(horizon, nonneg=True) 
        P_grid_bat = cp.Variable(horizon, nonneg=True)
        P_bat_load = cp.Variable(horizon, nonneg=True)
        P_bat_grid = cp.Variable(horizon, nonneg=True)
        
        # 电池变量
        P_bat_ch = cp.Variable(horizon, nonneg=True)
        P_bat_dis = cp.Variable(horizon, nonneg=True)
        Soc = cp.Variable(horizon + 1, nonneg=True)
        
        # 三态变量
        x_charge = cp.Variable(horizon, boolean=True)
        x_discharge = cp.Variable(horizon, boolean=True) 
        x_idle = cp.Variable(horizon, boolean=True)

        # === 电网三态二进制变量 ===
        g_charge = cp.Variable(horizon, boolean=True)    # 电网接收能量（售电给电网）
        g_discharge = cp.Variable(horizon, boolean=True) # 电网释放能量（向电网购电）
        g_idle = cp.Variable(horizon, boolean=True)      # 电网待机（能量平衡）
        
        # 辅助变量
        P_purchase = cp.Variable(horizon, nonneg=True)
        P_sell = cp.Variable(horizon, nonneg=True)
        
        # === 约束条件 ===
        constraints = []
        
        # 初始SOC
        constraints.append(Soc[0] == current_soc)
        
        for t in range(horizon):
            # 三态互斥约束
            constraints.append(x_charge[t] + x_discharge[t] + x_idle[t] == 1)
            constraints.append(g_charge[t] + g_discharge[t] + g_idle[t] == 1)
            
            # 电池功率约束
            constraints.append(P_bat_ch[t] <= self.P_bat_max * x_charge[t])
            constraints.append(P_bat_dis[t] <= self.P_bat_max * x_discharge[t])
            constraints.append(P_bat_ch[t] <= self.P_bat_max * (1 - x_idle[t]))
            constraints.append(P_bat_dis[t] <= self.P_bat_max * (1 - x_idle[t]))
            
            # 电网-电池功率约束
            constraints.append(P_grid_bat[t] <= self.grid_charge_max * x_charge[t])
            constraints.append(P_bat_grid[t] <= self.grid_discharge_max * x_discharge[t])
            
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
            
            # 功率平衡约束
            constraints.append(P_pv_load[t] + P_pv_bat[t] + P_pv_grid[t] == pv_pred[t])
            constraints.append(P_pv_load[t] + P_bat_load[t] + P_grid_load[t] == load_pred[t])
            constraints.append(P_pv_bat[t] + P_grid_bat[t] == P_bat_ch[t])
            constraints.append(P_bat_load[t] + P_bat_grid[t] == P_bat_dis[t])
            
            # 电网净功率
            constraints.append(P_purchase[t] - P_sell[t] == 
                             (P_grid_load[t] + P_grid_bat[t]) - (P_pv_grid[t] + P_bat_grid[t]))
            constraints.append(P_sell[t] == P_pv_grid[t] + P_bat_grid[t])
            constraints.append(P_purchase[t] == P_grid_load[t] + P_grid_bat[t])
            
            # SOC动态约束
            constraints.append(Soc[t+1] == Soc[t] + 
                             (P_bat_ch[t] * self.eta_c - P_bat_dis[t] / self.eta_d) / self.E_bat)
            
            # SOC边界约束
            constraints.append(Soc[t+1] >= self.Soc_min)
            constraints.append(Soc[t+1] <= self.Soc_max)
        
        # === 目标函数：成本最小化 + SOC跟踪 ===
        # 电力成本
        cost_term = cp.sum(cp.multiply(buy_prices_pred, P_purchase) - 
                          cp.multiply(sell_prices_pred, P_sell)) / 1000
        
        # SOC跟踪误差
        soc_tracking_term = cp.sum_squares(Soc - target_soc_pred)
        
        # 组合目标函数
        objective = cp.Minimize(self.cost_weight * cost_term + 
                               self.soc_tracking_weight * soc_tracking_term)
        
        # === 求解问题 ===
        problem = cp.Problem(objective, constraints)
        
        """
        try:
            problem.solve(solver=cp.GUROBI, verbose=False)
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                problem.solve(solver=cp.MOSEK, verbose=False)
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                problem.solve(solver=cp.ECOS, verbose=False)
        except:
            try:
                problem.solve(solver=cp.ECOS, verbose=False)
            except:
                problem.solve(verbose=False)
        """
        """
        solver_used = None
        try:
            problem.solve(solver=cp.GUROBI, verbose=False)
            solver_used = "GUROBI"
        except Exception as e:
            # print(e)
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

        problem.solve(solver=cp.GUROBI, verbose=False, TimeLimit=15, MIPGap=0.01)

        if problem.status in ['optimal', 'optimal_inaccurate']:
            # 构建功率分配结果
            power_allocations = []
            for t in range(horizon):
                power_allocations.append({
                    'P_pv_load': float(P_pv_load.value[t]),
                    'P_pv_bat': float(P_pv_bat.value[t]),
                    'P_pv_grid': float(P_pv_grid.value[t]),
                    'P_grid_load': float(P_grid_load.value[t]),
                    'P_grid_bat': float(P_grid_bat.value[t]),
                    'P_bat_load': float(P_bat_load.value[t]),
                    'P_bat_grid': float(P_bat_grid.value[t]),
                    'P_bat_ch': float(P_bat_ch.value[t]),
                    'P_bat_dis': float(P_bat_dis.value[t]),
                    'P_purchase': float(P_purchase.value[t]),
                    'P_sell': float(P_sell.value[t]),
                    'x_charge': int(round(x_charge.value[t])),
                    'x_discharge': int(round(x_discharge.value[t])),
                    'x_idle': int(round(x_idle.value[t]))
                })
            
            return {
                'status': 'optimal',
                'power_allocation': power_allocations,
                'soc_trajectory': Soc.value.tolist(),
                'objective_value': problem.value,
                'cost_component': float(cost_term.value) if cost_term.value is not None else 0,
                'tracking_component': float(soc_tracking_term.value) if soc_tracking_term.value is not None else 0
            }
        else:
            return {'status': 'failed', 'message': f'MPC求解失败: {problem.status}'}

    def _emergency_control(self, actual_pv, actual_load, current_soc, buy_price, sell_price):
        """
        应急控制：当MPC求解失败时的备选方案
        使用简单的规则控制
        """
        
        # print("  执行应急控制策略")
        
        # 计算功率差额
        power_deficit = actual_load - actual_pv
        
        # 初始化功率分配
        allocation = {
            'P_pv_load': min(actual_pv, actual_load),
            'P_pv_bat': 0, 'P_pv_grid': 0,
            'P_grid_load': 0, 'P_grid_bat': 0,
            'P_bat_load': 0, 'P_bat_grid': 0,
            'P_bat_ch': 0, 'P_bat_dis': 0,
            'P_purchase': 0, 'P_sell': 0,
            'x_charge': 0, 'x_discharge': 0, 'x_idle': 1
        }
        
        remaining_pv = actual_pv - allocation['P_pv_load']
        remaining_load = actual_load - allocation['P_pv_load']
        
        if remaining_load > 0:
            # 需要额外供电
            if current_soc > self.Soc_min + 0.05:  # SOC足够
                # 电池放电
                discharge_power = min(remaining_load, self.P_bat_max, 
                                    (current_soc - self.Soc_min) * self.E_bat * self.eta_d)
                allocation['P_bat_dis'] = discharge_power
                allocation['P_bat_load'] = discharge_power
                allocation['x_discharge'] = 1
                allocation['x_idle'] = 0
                remaining_load -= discharge_power
            
            if remaining_load > 0:
                # 电网供电
                allocation['P_grid_load'] = remaining_load
                allocation['P_purchase'] = remaining_load
        
        elif remaining_pv > 0:
            # 有多余光伏
            if current_soc < self.Soc_max - 0.05:  # SOC有空间
                # 光伏充电
                charge_power = min(remaining_pv, self.P_bat_max,
                                 (self.Soc_max - current_soc) * self.E_bat / self.eta_c)
                allocation['P_pv_bat'] = charge_power
                allocation['P_bat_ch'] = charge_power
                allocation['x_charge'] = 1
                allocation['x_idle'] = 0
                remaining_pv -= charge_power
            
            if remaining_pv > 0:
                # 光伏售电
                allocation['P_pv_grid'] = remaining_pv
                allocation['P_sell'] = remaining_pv
        
        # 计算成本和新SOC
        step_cost = (allocation['P_purchase'] * buy_price - allocation['P_sell'] * sell_price) / 1000
        energy_change = (allocation['P_bat_ch'] * self.eta_c - allocation['P_bat_dis'] / self.eta_d) / self.E_bat
        new_soc = max(self.Soc_min, min(self.Soc_max, current_soc + energy_change))
        
        return {
            'power_allocation': allocation,
            'step_cost': step_cost,
            'new_soc': new_soc,
            'mpc_status': 'emergency_control',
            'soc_tracking_error': 0,  # 应急控制不考虑跟踪
            'predicted_trajectory': []
        }

    def _initialize_execution_result(self, T, initial_soc):
        """初始化执行结果结构"""
        
        return {
            'status': 'optimal',
            'total_cost': 0.0,
            'money_spend': 0.0,
            'money_earn': 0.0,
            'power_flows': {
                'P_pv_load': np.zeros(T),
                'P_pv_bat': np.zeros(T),
                'P_pv_grid': np.zeros(T),
                'P_grid_load': np.zeros(T),
                'P_grid_bat': np.zeros(T),
                'P_bat_load': np.zeros(T),
                'P_bat_grid': np.zeros(T),
                'P_purchase': np.zeros(T),
                'P_sell': np.zeros(T),
                'P_net_grid': np.zeros(T)
            },
            'battery': {
                'P_bat_ch': np.zeros(T),
                'P_bat_dis': np.zeros(T),
                'x_charge': np.zeros(T, dtype=int),
                'x_discharge': np.zeros(T, dtype=int),
                'x_idle': np.zeros(T, dtype=int),
                'SOC': self._initialize_soc_array(T, initial_soc)
            }
        }

    def _initialize_soc_array(self, T, initial_soc):
        """初始化SOC数组，第一个元素设置为初始SOC值"""
        soc_array = np.zeros(T+1)
        soc_array[0] = initial_soc
        return soc_array

    def _update_execution_result(self, result, t, step_result):
        """更新执行结果"""
        
        allocation = step_result['power_allocation']
        
        # 更新功率流
        for key in result['power_flows']:
            if key in allocation:
                result['power_flows'][key][t] = allocation[key]
        
        # 更新电池数据
        result['battery']['P_bat_ch'][t] = allocation['P_bat_ch']
        result['battery']['P_bat_dis'][t] = allocation['P_bat_dis']
        result['battery']['SOC'][t+1] = step_result['new_soc']
        result['battery']['x_charge'][t] = allocation['x_charge']
        result['battery']['x_discharge'][t] = allocation['x_discharge']
        result['battery']['x_idle'][t] = allocation['x_idle']
        
        # 更新成本
        result['total_cost'] += step_result['step_cost']
        if step_result['step_cost'] > 0:
            result['money_spend'] += step_result['step_cost']
        else:
            result['money_earn'] += -step_result['step_cost']

    def _finalize_hierarchical_result(self, result, total_reoptimizations):
        """完成分层MPC结果"""
        
        # 计算净电网功率
        result['power_flows']['P_net_grid'] = (
            result['power_flows']['P_purchase'] - result['power_flows']['P_sell']
        )
        
        # 添加分层MPC信息
        result['mpc_info'] = {
            'solver_type': 'hierarchical_mpc',
            'prediction_horizon': self.prediction_horizon,
            'control_horizon': self.control_horizon,
            'planning_frequency': self.planning_frequency,
            'soc_tracking_weight': self.soc_tracking_weight,
            'cost_weight': self.cost_weight,
            'total_reoptimizations': total_reoptimizations,
            'average_soc_tracking_error': self._calculate_average_soc_tracking_error(),
            'global_plan_cost': self.global_plan['total_cost'] if self.global_plan else None,
            'hierarchical_gap': abs(result['total_cost'] - self.global_plan['total_cost']) if self.global_plan else None
        }
        
        return result

    def _calculate_average_soc_tracking_error(self):
        """计算平均SOC跟踪误差"""
        if not self.execution_history:
            return 0.0
        
        errors = [h.get('soc_tracking_error', 0) for h in self.execution_history]
        return np.mean(errors) if errors else 0.0

    # === 可视化和分析方法 ===
    
    def plot_hierarchical_analysis(self, result, pv_generation, load_demand, buy_prices, sell_prices, VIS_DIR):
        """绘制分层MPC特有的分析图表"""
        
        try:
            # 1. SOC跟踪性能分析
            self._plot_soc_tracking_analysis(VIS_DIR)
            
            # 2. 分层决策对比分析
            self._plot_hierarchical_decision_analysis(VIS_DIR)
            
            # 3. 成本分解分析
            self._plot_cost_decomposition_analysis(result, VIS_DIR)
            
        except Exception as e:
            # # print(f"⚠️ 分层MPC图表生成失败: {e}")
            import traceback
            traceback.print_exc()

    def _plot_soc_tracking_analysis(self, VIS_DIR):
        """绘制SOC跟踪性能分析图"""
        
        if not self.execution_history:
            # # print("⚠️ 无执行历史数据，跳过SOC跟踪分析图")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 添加初始时刻(t=-1)的数据点，显示初始SOC状态
        times = [-1] + [h['time'] for h in self.execution_history]
        
        # 获取初始SOC值
        initial_soc = self.target_soc_trajectory[0] if self.target_soc_trajectory is not None and len(self.target_soc_trajectory) > 0 else 0.05
        
        # 添加初始SOC到轨迹数据
        target_soc = [initial_soc * 100] + [h['target_soc'] * 100 for h in self.execution_history]
        actual_soc = [initial_soc * 100] + [h['actual_soc'] * 100 for h in self.execution_history]
        tracking_errors = [0.0] + [h['soc_tracking_error'] * 100 for h in self.execution_history]
        
        # 子图1：SOC轨迹跟踪
        ax1.plot(times, target_soc, 'b--', linewidth=2, label='目标SOC轨迹（全局规划）', marker='o', markersize=4)
        ax1.plot(times, actual_soc, 'r-', linewidth=2, label='实际SOC轨迹（MPC跟踪）', marker='s', markersize=4)
        ax1.fill_between(times, target_soc, actual_soc, alpha=0.3, color='gray', label='跟踪误差区域')
        
        ax1.set_xlabel('时间 (小时)')
        ax1.set_ylabel('SOC (%)')
        ax1.set_title('分层MPC - SOC轨迹跟踪性能', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-1.5, max(times) + 0.5)
        ax1.set_ylim(0, 110)
        
        # 子图2：跟踪误差分析
        ax2.bar(times, tracking_errors, alpha=0.7, color='orange', label='SOC跟踪误差')
        ax2.axhline(y=np.mean(tracking_errors), color='red', linestyle='--', 
                   label=f'平均误差 ({np.mean(tracking_errors):.1f}%)')
        
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('跟踪误差 (%)')
        ax2.set_title('SOC跟踪误差分布', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-1.5, max(times) + 0.5)
        
        plt.tight_layout()
        
        # 保存图表
        """
        import demo
        vis_dir = getattr(demo, 'VIS_DIR', 'vis')
        global VIS_DIR
        # print(VIS_DIR)
        """
        plt.savefig(os.path.join(VIS_DIR, 'hierarchical_mpc_soc_tracking.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # print("  ✅ SOC跟踪分析图已生成")

    def _plot_hierarchical_decision_analysis(self, VIS_DIR):
        """绘制分层决策分析图"""
        
        if not self.global_plan:
            # print("⚠️ 无全局规划数据，跳过分层决策分析图")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        times = list(range(len(self.global_plan['battery']['P_bat_ch'])))
        
        # 子图1：充电功率对比
        global_charge = self.global_plan['battery']['P_bat_ch']
        actual_charge = [h.get('power_allocation', {}).get('P_bat_ch', 0) for h in self.execution_history]
        
        ax1.plot(times, global_charge, 'b-', linewidth=2, label='全局规划充电', marker='o', markersize=4)
        if len(actual_charge) == len(times):
            ax1.plot(times, actual_charge, 'r-', linewidth=2, label='MPC实际充电', marker='s', markersize=4)
        
        ax1.set_xlabel('时间 (小时)')
        ax1.set_ylabel('充电功率 (W)')
        ax1.set_title('充电决策对比 - 全局规划 vs MPC跟踪', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2：放电功率对比
        global_discharge = self.global_plan['battery']['P_bat_dis']
        actual_discharge = [h.get('power_allocation', {}).get('P_bat_dis', 0) for h in self.execution_history]
        
        ax2.plot(times, global_discharge, 'b-', linewidth=2, label='全局规划放电', marker='o', markersize=4)
        if len(actual_discharge) == len(times):
            ax2.plot(times, actual_discharge, 'r-', linewidth=2, label='MPC实际放电', marker='s', markersize=4)
        
        ax2.set_xlabel('时间 (小时)')
        ax2.set_ylabel('放电功率 (W)')
        ax2.set_title('放电决策对比', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 子图3：电网交互对比
        global_net_grid = np.array(self.global_plan['power_flows']['P_net_grid'])
        actual_net_grid = [h.get('power_allocation', {}).get('P_purchase', 0) - 
                          h.get('power_allocation', {}).get('P_sell', 0) for h in self.execution_history]
        
        ax3.plot(times, global_net_grid, 'b-', linewidth=2, label='全局规划电网交互', marker='o', markersize=4)
        if len(actual_net_grid) == len(times):
            ax3.plot(times, actual_net_grid, 'r-', linewidth=2, label='MPC实际电网交互', marker='s', markersize=4)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xlabel('时间 (小时)')
        ax3.set_ylabel('净电网功率 (W)')
        ax3.set_title('电网交互决策对比', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 子图4：决策偏差统计
        if len(actual_charge) == len(times) and len(actual_discharge) == len(times):
            charge_deviation = np.array(actual_charge) - np.array(global_charge)
            discharge_deviation = np.array(actual_discharge) - np.array(global_discharge)
            
            ax4.plot(times, charge_deviation, 'g-', linewidth=2, label='充电偏差', marker='^', markersize=4)
            ax4.plot(times, discharge_deviation, 'orange', linewidth=2, label='放电偏差', marker='v', markersize=4)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            ax4.set_xlabel('时间 (小时)')
            ax4.set_ylabel('功率偏差 (W)')
            ax4.set_title('MPC vs 全局规划的决策偏差', fontsize=12, fontweight='bold')  
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        """
        import demo
        vis_dir = getattr(demo, 'VIS_DIR', 'vis')
        global VIS_DIR
        """
        plt.savefig(os.path.join(VIS_DIR, 'hierarchical_mpc_decision_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # print("  ✅ 分层决策分析图已生成")

    def _plot_cost_decomposition_analysis(self, result, VIS_DIR):
        """绘制成本分解分析图"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 子图1：成本对比
        categories = ['全局规划', '分层MPC实际', 'MPC跟踪成本', '分层额外成本']
        global_cost = self.global_plan['total_cost'] if self.global_plan else 0
        actual_cost = result['total_cost']
        hierarchical_gap = abs(actual_cost - global_cost)
        
        costs = [global_cost, actual_cost, 0, hierarchical_gap]  # MPC跟踪成本暂时设为0
        colors = ['blue', 'red', 'green', 'orange']
        
        bars = ax1.bar(categories, costs, color=colors, alpha=0.7)
        ax1.set_ylabel('成本 (元)')
        ax1.set_title('分层MPC成本分解分析', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上添加数值标签
        for bar, cost in zip(bars, costs):
            if cost > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{cost:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 子图2：性能指标雷达图风格的条形图
        metrics = ['成本效率', 'SOC跟踪', '实时性', '鲁棒性']
        # 计算性能指标（归一化到0-1）
        cost_efficiency = max(0, 1 - hierarchical_gap / max(global_cost, 0.001))
        soc_tracking = max(0, 1 - self._calculate_average_soc_tracking_error() / 0.1)  # 假设10%为最差
        realtime_performance = 0.8  # 假设MPC实时性较好
        robustness = 0.9  # 假设分层MPC鲁棒性较好
        
        values = [cost_efficiency, soc_tracking, realtime_performance, robustness]
        
        bars2 = ax2.barh(metrics, values, color=['red', 'green', 'blue', 'orange'], alpha=0.7)
        ax2.set_xlabel('性能评分 (0-1)')
        ax2.set_title('分层MPC性能评估', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim(0, 1)
        
        # 在条形上添加数值标签
        for bar, value in zip(bars2, values):
            ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{value:.2f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        """
        import demo
        vis_dir = getattr(demo, 'VIS_DIR', 'vis')
        global VIS_DIR
        """
        plt.savefig(os.path.join(VIS_DIR, 'hierarchical_mpc_cost_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # print("  ✅ 成本分解分析图已生成")

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

        VIS_DIR = 'vis/gateway_id:{}-date:{}-hier-mpc-grid-constrain-pv100-load100'.format(gateway_id, date)
        if not os.path.exists(VIS_DIR):
            os.makedirs(VIS_DIR)

        initial_soc = _curr_soc

        hierarchical_mpc_ems = HierarchicalMPCEnergyManagementSystem(
            battery_capacity=_rated_cap,
            max_power=_rated_power,
            charge_efficiency=0.9,
            discharge_efficiency=0.9,
            grid_charge_max=max(grid_charge_max),
            grid_discharge_max=max(grid_discharge_max),
            soc_min=0.2,
            soc_max=0.95,
            prediction_horizon=6,        # MPC预测时域4小时
            control_horizon=1,           # MPC控制时域1小时
            soc_tracking_weight=1,    # SOC跟踪权重
            cost_weight=1.0,             # 成本权重
            planning_frequency=24        # 全局规划频率24小时
        )

        result = hierarchical_mpc_ems.hierarchical_mpc_optimize(
            pv_pred, load_pred, buy_prices, sell_prices,
            real_pv=pv, real_load=load,
            initial_soc=_curr_soc
        )

        # === 保存结果 ===
        json_result = convert_to_json_serializable(result)
        json_result['input_data'] = {
            'pv_generation': pv_pred,
            'load_demand': load_pred,
            'real_pv': pv,
            'real_load': load,
            'buy_prices': buy_prices,
            'sell_prices': sell_prices,
            'initial_soc': _curr_soc,
            'execution_history': convert_to_json_serializable(hierarchical_mpc_ems.execution_history),
            'planning_history': convert_to_json_serializable(hierarchical_mpc_ems.planning_history),
            'battery_config': {
                'capacity': hierarchical_mpc_ems.E_bat,
                'max_power': hierarchical_mpc_ems.P_bat_max,
                'charge_efficiency': hierarchical_mpc_ems.eta_c,
                'discharge_efficiency': hierarchical_mpc_ems.eta_d,
                'soc_min': hierarchical_mpc_ems.Soc_min,
                'soc_max': hierarchical_mpc_ems.Soc_max,
                'grid_charge_max': hierarchical_mpc_ems.grid_charge_max,
                'grid_discharge_max': hierarchical_mpc_ems.grid_discharge_max
            }
        }
        
        df = hierarchical_mpc_ems.create_hourly_analysis(
            result, pv_pred, load_pred, buy_prices, sell_prices, pv, load
        )
        
        # 添加小时级分析到JSON结果
        json_result['hourly_analysis'] = convert_to_json_serializable(df.to_dict('records'))
        
        # 保存完整结果
        with open(os.path.join(VIS_DIR, 'res.json'), 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        """
        report_table = hierarchical_mpc_ems.create_enhanced_report_table(df, VIS_DIR)
        hierarchical_mpc_ems.plot_hierarchical_analysis(result, pv_pred, load_pred, buy_prices, sell_prices, VIS_DIR)
        """
        end_time = time.time()
        print("process {}-{}, time cost: {}...".format(gateway_id, date, end_time-start_time))
        return 1
    except Exception as e:
        """
        if VIS_DIR and os.path.exists(VIS_DIR):
            try:
                import shutil
                shutil.rmtree(VIS_DIR)  # 比os.system更可靠
            except:
                pass
        # print(f"Task failed for {gateway_id}-{date}: {e}")
        raise  # 重新抛出异常，让executor.map能捕获到
        """
        return None

def test_with_logging(info):
    try:
        return test(info)
    except Exception as e:
        gateway_id = info.get("gateway_id", "unknown")
        date = info.get("date", "unknown")
        # print(f"❌ Failed: {gateway_id}-{date}: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import time
    with open("test_samples.json", "r") as f:
        info_list = json.load(f)

    pv_noise_rate = 1
    load_noise_rate = 1
    np.random.seed(123)

    info_list_new = []
    for info in tqdm.tqdm(info_list):
        try:
            rate = (1 - pv_noise_rate) + 2 * pv_noise_rate * np.random.rand(24)
            info_pv = info["pv"] * rate
            info["pv"] = info_pv.tolist()
            rate = (1 - load_noise_rate) + 2 * load_noise_rate * np.random.rand(24)
            info_load = info["load"] * rate
            info["load"] = info_load.tolist()
            info_list_new.append(info)
        except Exception as e:
            print(e)
    """
    start_t = time.time()
    for info in info_list[:2000]:
        #if info["gateway_id"] == "bcd09f16beb2a284565d5c5f1fa22a8f" and info["date"]=="2024-08-12":
        test(info)
    """
    # print("time: ", time.time() - start_t)

    successful = 0
    failed = 0
    cpu_count = os.cpu_count()

    with concurrent.futures.ProcessPoolExecutor(max_workers=int(cpu_count*0.95)) as executor:  # 减少并发数
        results = list(executor.map(test, info_list_new))

    