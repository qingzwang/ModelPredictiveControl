import json, os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class PowerToPriorityConverter:
    """
    将功率流向的具体数字转化为优先级规则 - 改进版
    确保优先级分配的确定性和唯一性
    """
    
    def __init__(self, threshold_ratio=0.05):
        """
        初始化转换器
        
        Args:
            threshold_ratio: 判断功率流向显著性的阈值比例
        """
        self.threshold_ratio = threshold_ratio
        
    def analyze_power_flows(self, result_data: Dict) -> Dict:
        """
        分析功率流向数据，提取每小时的决策模式
        """
        power_flows = result_data['power_flows']
        T = len(power_flows['P_pv_load'])
        
        hourly_analysis = []
        
        for t in range(T):
            # 获取当前小时的所有功率流向
            pv_load = power_flows['P_pv_load'][t]
            pv_bat = power_flows['P_pv_bat'][t]
            pv_grid = power_flows['P_pv_grid'][t]
            grid_load = power_flows['P_grid_load'][t]
            grid_bat = power_flows['P_grid_bat'][t]
            bat_load = power_flows['P_bat_load'][t]
            bat_grid = power_flows['P_bat_grid'][t]
            
            # 使用功率平衡关系计算实际发电量和负载需求
            pv_gen = pv_load + pv_bat + pv_grid
            load_demand = pv_load + grid_load + bat_load
            
            # 分析优先级模式
            solar_priority, load_priority = self._extract_priorities_deterministic(
                pv_gen, load_demand, pv_load, pv_bat, pv_grid, 
                grid_load, grid_bat, bat_load, bat_grid
            )
            
            # 确定主导策略
            main_strategy = self._determine_main_strategy(
                pv_load, pv_bat, pv_grid, grid_load, grid_bat, bat_load, bat_grid
            )
            
            hourly_analysis.append({
                'hour': t,
                'pv_generation': pv_gen,
                'load_demand': load_demand,
                'solar_priority': solar_priority,
                'load_priority': load_priority,
                'main_strategy': main_strategy,
                'power_flows': {
                    'pv_to_load': pv_load,
                    'pv_to_bat': pv_bat,
                    'pv_to_grid': pv_grid,
                    'grid_to_load': grid_load,
                    'grid_to_bat': grid_bat,
                    'bat_to_load': bat_load,
                    'bat_to_grid': bat_grid
                }
            })
        
        return {
            'hourly_analysis': hourly_analysis,
            'summary': self._generate_priority_summary(hourly_analysis)
        }
    
    def _extract_priorities_deterministic(self, pv_gen: float, load_demand: float,
                                        pv_load: float, pv_bat: float, pv_grid: float,
                                        grid_load: float, grid_bat: float, 
                                        bat_load: float, bat_grid: float) -> Tuple[List[int], List[int]]:
        """
        使用确定性规则从功率流向提取优先级，确保不会有相同优先级
        
        Returns:
            (solar_priority, load_priority)
            solar_priority: [电网, 负载, 电池] 的优先级 [1,2,3]
            load_priority: [电网, 光伏, 电池] 的优先级 [1,2,3]
        """
        
        # 分析光伏发电的去向优先级
        solar_priority = self._infer_solar_priority_deterministic(
            pv_gen, pv_load, pv_bat, pv_grid
        )
        
        # 分析负载需求的来源优先级
        load_priority = self._infer_load_priority_deterministic(
            load_demand, pv_load, grid_load, bat_load
        )
        
        return solar_priority, load_priority
    
    def _infer_solar_priority_deterministic(self, pv_gen: float, pv_load: float, 
                                          pv_bat: float, pv_grid: float) -> List[int]:
        """
        确定性地推导光伏发电优先级
        
        规则逻辑：
        1. 首先看哪个方向有显著功率流（>5%的光伏发电量）
        2. 然后按照预设的决策树规则分配优先级
        3. 确保每个组件都有唯一的优先级 [1,2,3]
        
        Returns:
            [电网, 负载, 电池] 的优先级
        """
        if pv_gen < 10:  # 光伏发电量很小
            return [1, 3, 2]  # 默认：负载(3) > 电池(2) > 电网(1)
        
        # 计算有效功率阈值
        threshold = max(10, pv_gen * self.threshold_ratio)
        
        # 判断各方向是否有显著功率流
        to_load = pv_load > threshold
        to_bat = pv_bat > threshold  
        to_grid = pv_grid > threshold
        
        # 根据功率流模式确定性地分配优先级
        if to_load and to_bat and to_grid:
            # 三个方向都有流向：按功率大小排序，但有确定性的平局处理
            flows = [
                ('grid', pv_grid, 0),    # 电网
                ('load', pv_load, 1),    # 负载  
                ('battery', pv_bat, 2)   # 电池
            ]
            # 按功率排序，功率相同时按索引排序（确定性）
            flows.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            
            priorities = [0, 0, 0]
            for rank, (name, power, idx) in enumerate(flows):
                priorities[idx] = 3 - rank
            
        elif to_load and to_bat:
            # 负载和电池有流向，电网没有 → 负载优先
            return [1, 3, 2]  # 负载(3) > 电池(2) > 电网(1)
            
        elif to_load and to_grid:
            # 负载和电网有流向，电池没有 → 看哪个更大
            if pv_load >= pv_grid:
                return [2, 3, 1]  # 负载(3) > 电网(2) > 电池(1)
            else:
                return [3, 2, 1]  # 电网(3) > 负载(2) > 电池(1)
                
        elif to_bat and to_grid:
            # 电池和电网有流向，负载没有 → 售电模式
            return [3, 1, 2]  # 电网(3) > 电池(2) > 负载(1)
            
        elif to_load:
            # 只有负载有流向 → 自发自用优先
            return [1, 3, 2]  # 负载(3) > 电池(2) > 电网(1)
            
        elif to_bat:
            # 只有电池有流向 → 储能优先
            return [1, 2, 3]  # 电池(3) > 负载(2) > 电网(1)
            
        elif to_grid:
            # 只有电网有流向 → 售电优先
            return [3, 2, 1]  # 电网(3) > 负载(2) > 电池(1)
            
        else:
            # 没有显著流向 → 使用默认优先级
            return [1, 3, 2]  # 负载(3) > 电池(2) > 电网(1)
        
        return priorities
    
    def _infer_load_priority_deterministic(self, load_demand: float, pv_load: float,
                                         grid_load: float, bat_load: float) -> List[int]:
        """
        确定性地推导负载需求优先级
        
        Returns:
            [电网, 光伏, 电池] 的优先级
        """
        if load_demand < 10:  # 负载需求很小
            return [2, 3, 1]  # 默认：光伏(3) > 电网(2) > 电池(1)
        
        # 计算有效功率阈值
        threshold = max(10, load_demand * self.threshold_ratio)
        
        # 判断各来源是否有显著功率流
        from_grid = grid_load > threshold
        from_pv = pv_load > threshold
        from_bat = bat_load > threshold
        
        # 根据功率流模式确定性地分配优先级
        if from_grid and from_pv and from_bat:
            # 三个来源都有：按功率大小排序，确定性处理平局
            flows = [
                ('grid', grid_load, 0),  # 电网
                ('pv', pv_load, 1),      # 光伏
                ('battery', bat_load, 2) # 电池
            ]
            flows.sort(key=lambda x: (x[1], -x[2]), reverse=True)
            
            priorities = [0, 0, 0]
            for rank, (name, power, idx) in enumerate(flows):
                priorities[idx] = 3 - rank
                
        elif from_grid and from_pv:
            # 电网和光伏供电 → 优先光伏
            return [2, 3, 1]  # 光伏(3) > 电网(2) > 电池(1)
            
        elif from_grid and from_bat:
            # 电网和电池供电 → 优先电池
            return [2, 1, 3]  # 电池(3) > 电网(2) > 光伏(1)
            
        elif from_pv and from_bat:
            # 光伏和电池供电 → 优先光伏
            return [1, 3, 2]  # 光伏(3) > 电池(2) > 电网(1)
            
        elif from_grid:
            # 只有电网供电 → 电网优先
            return [3, 2, 1]  # 电网(3) > 光伏(2) > 电池(1)
            
        elif from_pv:
            # 只有光伏供电 → 光伏优先
            return [1, 3, 2]  # 光伏(3) > 电网(1) > 电池(2)
            
        elif from_bat:
            # 只有电池供电 → 电池优先
            return [1, 2, 3]  # 电池(3) > 光伏(2) > 电网(1)
            
        else:
            # 没有显著流向 → 使用默认优先级
            return [2, 3, 1]  # 光伏(3) > 电网(2) > 电池(1)
        
        return priorities
    
    def _determine_main_strategy(self, pv_load: float, pv_bat: float, pv_grid: float,
                               grid_load: float, grid_bat: float, 
                               bat_load: float, bat_grid: float) -> str:
        """
        确定主导策略类型
        """
        threshold = 50  # W
        
        # 电池相关策略
        if bat_load > threshold:
            return "电池供负载"
        elif bat_grid > threshold:
            return "电池售电"
        elif pv_bat > threshold:
            return "光伏充电"
        elif grid_bat > threshold:
            return "电网充电"
        
        # 光伏相关策略
        elif pv_grid > threshold:
            return "光伏售电"
        elif pv_load > threshold:
            return "光伏供负载"
        
        # 电网相关策略
        elif grid_load > threshold:
            return "电网供负载"
        
        else:
            return "功率平衡"
    
    def _generate_priority_summary(self, hourly_analysis: List[Dict]) -> Dict:
        """
        生成优先级规则总结
        """
        from collections import Counter
        
        # 统计各种优先级模式的出现频次
        solar_priority_counts = Counter([str(hour_data['solar_priority']) for hour_data in hourly_analysis])
        load_priority_counts = Counter([str(hour_data['load_priority']) for hour_data in hourly_analysis])
        strategy_counts = Counter([hour_data['main_strategy'] for hour_data in hourly_analysis])
        
        # 找出最常见的优先级模式
        most_common_solar = solar_priority_counts.most_common(1)[0]
        most_common_load = load_priority_counts.most_common(1)[0]
        most_common_strategy = strategy_counts.most_common(1)[0]
        
        return {
            'solar_priority_distribution': dict(solar_priority_counts),
            'load_priority_distribution': dict(load_priority_counts),
            'strategy_distribution': dict(strategy_counts),
            'recommended_solar_priority': eval(most_common_solar[0]),
            'recommended_load_priority': eval(most_common_load[0]),
            'dominant_strategy': most_common_strategy[0],
            'priority_explanations': {
                'solar_priority': "[電網, 負載, 電池]優先級 (1=最低, 2=中等, 3=最高)",
                'load_priority': "[電網, 光伏, 電池]優先級 (1=最低, 2=中等, 3=最高)"
            }
        }
    
    def convert_to_rule_based_format(self, analysis_result: Dict) -> Dict:
        """
        将分析结果转换为规则基础系统可用的格式
        """
        summary = analysis_result['summary']
        hourly_analysis = analysis_result['hourly_analysis']
        
        # 生成每小时的优先级设置
        solar_priorities = []
        load_priorities = []
        
        for hour_data in hourly_analysis:
            solar_priorities.append(hour_data['solar_priority'])
            load_priorities.append(hour_data['load_priority'])
        
        return {
            'solar_priority': solar_priorities,
            'load_priority': load_priorities,
            'recommended_global_solar_priority': summary['recommended_solar_priority'],
            'recommended_global_load_priority': summary['recommended_load_priority'],
            'strategy_analysis': summary['strategy_distribution'],
            'usage_instructions': {
                'dynamic_priorities': "使用solar_priority和load_priority列表，每小时不同优先级",
                'static_priorities': "使用recommended_global_*作为全天统一优先级",
                'priority_format': "[电网, 负载/光伏, 电池]，1=最低优先级，2=中等优先级，3=最高优先级",
                'uniqueness_guarantee': "每个组件都有唯一的优先级，不存在相同优先级"
            }
        }


def analyze_optimization_result(json_file_path: str) -> Dict:
    """
    主函数：分析优化结果并提取确定性的唯一优先级规则
    """
    print("Precessing {}...".format(json_file_path))
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        converter = PowerToPriorityConverter(threshold_ratio=0.05)
        analysis_result = converter.analyze_power_flows(result_data)
        rule_format = converter.convert_to_rule_based_format(analysis_result)
        
        p1, p2, p3 = json_file_path.split("/")
        with open(os.path.join(p1, p2, "rule.json"), "w", encoding='utf-8') as f:
            json.dump(
                {
                    'detailed_analysis': analysis_result,
                    'rule_based_format': rule_format
                },
                f
            )
        
        return {
            'detailed_analysis': analysis_result,
            'rule_based_format': rule_format
        }
    except Exception as e:
        print("{}, {}".format(json_file_path, e))


def validate_priority_uniqueness(priorities_list: List[List[int]]) -> bool:
    """
    验证优先级列表中每个优先级设置都是唯一的
    """
    for priorities in priorities_list:
        if len(set(priorities)) != len(priorities):
            return False
    return True


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    import concurrent.futures
    import os
    path = os.listdir('vis')
    path = [os.path.join("vis", p, "res.json") for p in path if p.endswith("hier-mpc-grid-constrain-pv100-load100")]    
    cpu_count = os.cpu_count()
    n=0
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(cpu_count*0.9)) as executor:
        #for info in info_list:
        #    func = partial(test, info)
        #    executor.submit(func)
        results = list(executor.map(analyze_optimization_result, path))