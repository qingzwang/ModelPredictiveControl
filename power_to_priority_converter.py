import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class PowerToPriorityConverter:
    """
    将功率流向的具体数字转化为优先级规则
    用于从优化结果反推决策逻辑
    """
    
    def __init__(self, threshold_ratio=0.1):
        """
        初始化转换器
        
        Args:
            threshold_ratio: 判断功率流向显著性的阈值比例
        """
        self.threshold_ratio = threshold_ratio
        
    def analyze_power_flows(self, result_data: Dict) -> Dict:
        """
        分析功率流向数据，提取每小时的决策模式
        
        Args:
            result_data: 包含power_flows的结果数据
            
        Returns:
            每小时的决策分析结果
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
            
            # 获取输入数据
            pv_gen = result_data['input_data']['real_pv'][t]
            load_demand = result_data['input_data']['real_load'][t]
            
            # 分析优先级模式
            solar_priority, load_priority = self._extract_priorities(
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
    
    def _extract_priorities(self, pv_gen: float, load_demand: float,
                          pv_load: float, pv_bat: float, pv_grid: float,
                          grid_load: float, grid_bat: float, 
                          bat_load: float, bat_grid: float) -> Tuple[List[int], List[int]]:
        """
        从功率流向提取光伏发电和负载需求的优先级
        
        Returns:
            (solar_priority, load_priority)
            solar_priority: [电网, 负载, 电池] 的优先级
            load_priority: [电网, 光伏, 电池] 的优先级
        """
        
        # 分析光伏发电的去向优先级
        solar_flows = {
            'grid': pv_grid,    # 光伏→电网
            'load': pv_load,    # 光伏→负载
            'battery': pv_bat   # 光伏→电池
        }
        
        # 分析负载需求的来源优先级
        load_sources = {
            'grid': grid_load,  # 电网→负载
            'pv': pv_load,      # 光伏→负载 (已在solar_flows中计算)
            'battery': bat_load # 电池→负载
        }
        
        # 根据实际功率流向推导优先级
        solar_priority = self._infer_solar_priority(solar_flows, pv_gen)
        load_priority = self._infer_load_priority(load_sources, load_demand)
        
        return solar_priority, load_priority
    
    def _infer_solar_priority(self, solar_flows: Dict[str, float], pv_gen: float) -> List[int]:
        """
        从光伏功率分配推导光伏发电优先级
        
        Returns:
            [电网, 负载, 电池] 的优先级 (数字越大优先级越高)
        """
        if pv_gen < 1:  # 光伏发电量很小，无法判断优先级
            return [1, 2, 0]  # 默认：负载>电网>电池
        
        # 计算各去向的功率占比
        total_flow = sum(solar_flows.values())
        if total_flow < 1:
            return [1, 2, 0]
        
        ratios = {k: v/total_flow for k, v in solar_flows.items()}
        
        # 根据功率占比推导优先级
        # 如果某个方向功率占比很高，说明它的优先级高
        priority_mapping = {'grid': 0, 'load': 1, 'battery': 2}  # 对应[电网, 负载, 电池]
        priorities = [0, 0, 0]
        
        # 按占比排序，占比最高的优先级最高
        sorted_flows = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (destination, ratio) in enumerate(sorted_flows):
            if ratio > self.threshold_ratio:  # 只有显著的功率流向才赋予高优先级
                priorities[priority_mapping[destination]] = 3 - rank
        
        return priorities
    
    def _infer_load_priority(self, load_sources: Dict[str, float], load_demand: float) -> List[int]:
        """
        从负载供电来源推导负载需求优先级
        
        Returns:
            [电网, 光伏, 电池] 的优先级 (数字越大优先级越高)
        """
        if load_demand < 1:  # 负载需求很小，无法判断优先级
            return [1, 3, 2]  # 默认：光伏>电池>电网
        
        # 计算各来源的功率占比
        total_supply = sum(load_sources.values())
        if total_supply < 1:
            return [1, 3, 2]
        
        ratios = {k: v/total_supply for k, v in load_sources.items()}
        
        # 根据功率占比推导优先级
        priority_mapping = {'grid': 0, 'pv': 1, 'battery': 2}  # 对应[电网, 光伏, 电池]
        priorities = [0, 0, 0]
        
        # 按占比排序，占比最高的优先级最高
        sorted_sources = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (source, ratio) in enumerate(sorted_sources):
            if ratio > self.threshold_ratio:  # 只有显著的功率流向才赋予高优先级
                priorities[priority_mapping[source]] = 3 - rank
        
        return priorities
    
    def _determine_main_strategy(self, pv_load: float, pv_bat: float, pv_grid: float,
                               grid_load: float, grid_bat: float, 
                               bat_load: float, bat_grid: float) -> str:
        """
        确定主导策略类型
        """
        # 定义功率阈值
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
        # 统计各种优先级模式的出现频次
        solar_priority_counts = {}
        load_priority_counts = {}
        strategy_counts = {}
        
        for hour_data in hourly_analysis:
            # 统计光伏优先级模式
            solar_key = str(hour_data['solar_priority'])
            solar_priority_counts[solar_key] = solar_priority_counts.get(solar_key, 0) + 1
            
            # 统计负载优先级模式
            load_key = str(hour_data['load_priority'])
            load_priority_counts[load_key] = load_priority_counts.get(load_key, 0) + 1
            
            # 统计策略类型
            strategy = hour_data['main_strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # 找出最常见的优先级模式
        most_common_solar = max(solar_priority_counts.items(), key=lambda x: x[1])
        most_common_load = max(load_priority_counts.items(), key=lambda x: x[1])
        most_common_strategy = max(strategy_counts.items(), key=lambda x: x[1])
        
        return {
            'solar_priority_distribution': solar_priority_counts,
            'load_priority_distribution': load_priority_counts,
            'strategy_distribution': strategy_counts,
            'recommended_solar_priority': eval(most_common_solar[0]),
            'recommended_load_priority': eval(most_common_load[0]),
            'dominant_strategy': most_common_strategy[0],
            'priority_explanations': {
                'solar_priority': "[電網, 負載, 電池]優先級 (數字越大優先級越高)",
                'load_priority': "[電網, 光伏, 電池]優先級 (數字越大優先級越高)"
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
            'solar_priority': solar_priorities,  # 24小时的光伏优先级
            'load_priority': load_priorities,    # 24小时的负载优先级
            'recommended_global_solar_priority': summary['recommended_solar_priority'],
            'recommended_global_load_priority': summary['recommended_load_priority'],
            'strategy_analysis': summary['strategy_distribution'],
            'usage_instructions': {
                'dynamic_priorities': "使用solar_priority和load_priority列表，每小时不同优先级",
                'static_priorities': "使用recommended_global_*作为全天统一优先级",
                'priority_format': "[电网, 负载/光伏, 电池]，数字越大优先级越高"
            }
        }


def analyze_optimization_result(json_file_path: str) -> Dict:
    """
    主函数：分析优化结果并提取优先级规则
    
    Args:
        json_file_path: JSON结果文件路径
        
    Returns:
        优先级分析结果
    """
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        result_data = json.load(f)
    
    # 创建转换器
    converter = PowerToPriorityConverter(threshold_ratio=0.1)
    
    # 分析功率流向
    analysis_result = converter.analyze_power_flows(result_data)
    
    # 转换为规则基础格式
    rule_format = converter.convert_to_rule_based_format(analysis_result)
    
    return {
        'detailed_analysis': analysis_result,
        'rule_based_format': rule_format
    }


if __name__ == "__main__":
    # 示例用法
    json_path = "vis/gateway_id:ebbb9e1a343b092537857cc17021670e-date:2024-12-29-hier-mpc-pcs/res.json"
    
    try:
        result = analyze_optimization_result(json_path)
        
        print("="*60)
        print("🔍 功率流向转优先级分析结果")
        print("="*60)
        
        # 显示推荐的全局优先级
        rule_format = result['rule_based_format']
        print(f"\n📊 推荐的全局优先级设置:")
        print(f"光伏发电优先级 [电网, 负载, 电池]: {rule_format['recommended_global_solar_priority']}")
        print(f"负载需求优先级 [电网, 光伏, 电池]: {rule_format['recommended_global_load_priority']}")
        
        # 显示策略分布
        print(f"\n📈 主导策略分析:")
        for strategy, count in rule_format['strategy_analysis'].items():
            print(f"  {strategy}: {count}小时")
        
        # 显示使用说明
        print(f"\n📝 使用说明:")
        for key, instruction in rule_format['usage_instructions'].items():
            print(f"  {key}: {instruction}")
        
        # 保存分析结果
        output_path = json_path.replace('.json', '_priority_analysis.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 详细分析结果已保存至: {output_path}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")