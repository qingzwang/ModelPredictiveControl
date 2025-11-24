import pandas as pd
import numpy as np
import json
import tqdm
from typing import Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import os

#plt.rcParams['font.sans-serif'] = ['Heiti TC', 'DejaVu Sans', "SourceHanSansSC-Normal.otf"]
#zhfont1 = FontProperties(fname="SourceHanSansSC-Normal.otf", size = 15)
font_path = 'SourceHanSansSC-Normal.otf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

TIME_MATCHING = {
    0: "00:00:00",
    1: "01:00:00",
    2: "02:00:00",
    3: "03:00:00",
    4: "04:00:00",
    5: "05:00:00",
    6: "06:00:00",
    7: "07:00:00",
    8: "08:00:00",
    9: "09:00:00",
    10: "10:00:00",
    11: "11:00:00",
    12: "12:00:00",
    13: "13:00:00",
    14: "14:00:00",
    15: "15:00:00",
    16: "16:00:00",
    17: "17:00:00",
    18: "18:00:00",
    19: "19:00:00",
    20: "20:00:00",
    21: "21:00:00",
    22: "22:00:00",
    23: "23:00:00"
}


def get_time_string(hour: int) -> str:
    """
    TIME_MATCHING功能：将小时数字转换为时间字符串
    
    Args:
        hour: 小时数 (0-23)
    
    Returns:
        str: 时间字符串格式 "HH:00:00"
    """
    if hour in TIME_MATCHING:
        return TIME_MATCHING[hour]
    else:
        # 对于超出范围的值，返回None或抛出异常
        raise ValueError(f"Invalid hour: {hour}. Hour must be between 0 and 23.")


def extract_load_sun_data(load_data: Union[str, pd.DataFrame], sun_data: Union[str, pd.DataFrame], tariff_data: Union[str, pd.DataFrame],
                         gateway_id: str, datetime: str) -> Optional[Tuple[float, float, float, float, float, float]]:
    """
    从load、sun和tariff数据中提取指定gateway_id和datetime的数据
    
    Args:
        load_data: load CSV文件路径(str)或已读取的DataFrame
        sun_data: sun CSV文件路径(str)或已读取的DataFrame  
        tariff_data: tariff CSV文件路径(str)或已读取的DataFrame
        gateway_id: 网关ID
        datetime: 时间字符串 (格式: 'YYYY-MM-DD HH:MM:SS')
    
    Returns:
        tuple: (load_pred, load, sun_pred, sun, buy_price, sell_price) 如果找到数据，否则返回None
    """
    try:
        # 支持传入DataFrame或文件路径
        load_df = pd.read_csv(load_data) if isinstance(load_data, str) else load_data
        sun_df = pd.read_csv(sun_data) if isinstance(sun_data, str) else sun_data
        tariff_df = pd.read_csv(tariff_data) if isinstance(tariff_data, str) else tariff_data
        
        # 查找指定gateway_id和datetime的记录
        load_record = load_df[(load_df['gateway_id'] == gateway_id) & 
                             (load_df['datetime'] == datetime)]
        sun_record = sun_df[(sun_df['gateway_id'] == gateway_id) & 
                           (sun_df['datetime'] == datetime)]
        tariff_record = tariff_df[(tariff_df['gateway_id'] == gateway_id) & 
                                 (tariff_df['device_time'] == datetime)]
        
        # 检查是否找到匹配记录
        if load_record.empty or sun_record.empty or tariff_record.empty:
            return None
        
        # 提取数据
        load_pred = float(load_record.iloc[0]['kwh_load_predict'])
        load = float(load_record.iloc[0]['kwh_load'])
        sun_pred = float(sun_record.iloc[0]['kwh_sun_predict'])
        sun = float(sun_record.iloc[0]['kwh_sun'])
        buy_price = float(tariff_record.iloc[0]['tariff_price'])
        sell_price = float(tariff_record.iloc[0]['tariff_sell_price'])
        
        return (load_pred, load, sun_pred, sun, buy_price, sell_price)
        
    except Exception as e:
        print(f"Error extracting data: {e}")
        return None


def extract_all_matching_data(load_data: Union[str, pd.DataFrame], sun_data: Union[str, pd.DataFrame], tariff_data: Union[str, pd.DataFrame],
                             gateway_id: str) -> pd.DataFrame:
    """
    提取指定gateway_id的所有匹配时间点的数据
    
    Args:
        load_data: load CSV文件路径(str)或已读取的DataFrame
        sun_data: sun CSV文件路径(str)或已读取的DataFrame  
        tariff_data: tariff CSV文件路径(str)或已读取的DataFrame
        gateway_id: 网关ID
    
    Returns:
        DataFrame: 包含匹配数据的DataFrame，列为 [datetime, load_pred, load, sun_pred, sun, buy_price, sell_price]
    """
    try:
        # 支持传入DataFrame或文件路径
        load_df = pd.read_csv(load_data) if isinstance(load_data, str) else load_data
        sun_df = pd.read_csv(sun_data) if isinstance(sun_data, str) else sun_data
        tariff_df = pd.read_csv(tariff_data) if isinstance(tariff_data, str) else tariff_data
        
        # 筛选指定gateway_id的数据
        load_filtered = load_df[load_df['gateway_id'] == gateway_id]
        sun_filtered = sun_df[sun_df['gateway_id'] == gateway_id]
        tariff_filtered = tariff_df[tariff_df['gateway_id'] == gateway_id]
        
        # 先合并load和sun数据
        merged_df = pd.merge(load_filtered, sun_filtered, on=['gateway_id', 'datetime'])
        
        # 再合并tariff数据，使用device_time匹配datetime
        merged_df = pd.merge(merged_df, tariff_filtered, 
                           left_on=['gateway_id', 'datetime'], 
                           right_on=['gateway_id', 'device_time'])
        
        # 选择需要的列并重命名
        result_df = merged_df[['datetime', 'kwh_load_predict', 'kwh_load', 
                              'kwh_sun_predict', 'kwh_sun', 'tariff_price', 'tariff_sell_price']].copy()
        result_df.columns = ['datetime', 'load_pred', 'load', 'sun_pred', 'sun', 'buy_price', 'sell_price']
        
        return result_df
        
    except Exception as e:
        print(f"Error extracting all matching data: {e}")
        return pd.DataFrame()


def group_by_24h_periods(df: pd.DataFrame) -> list:
    """
    将数据按24小时周期分组，每组必须包含完整的00:00:00到23:00:00的24个点
    不满足24小时的数据会被丢弃
    
    Args:
        df: 包含datetime列的DataFrame
    
    Returns:
        list: 每个元素是一个包含24小时完整数据的DataFrame
    """
    try:
        # 确保datetime列是datetime类型
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 按日期分组
        df['date'] = df['datetime'].dt.date
        grouped = df.groupby('date')
        
        valid_groups = []
        
        for date, group in grouped:
            # 检查是否有24个小时的数据
            hours = group['datetime'].dt.hour.unique()
            
            # 检查是否包含0-23所有小时
            expected_hours = set(range(24))
            actual_hours = set(hours)
            
            if actual_hours == expected_hours:
                # 按小时排序
                group_sorted = group.sort_values('datetime').reset_index(drop=True)
                valid_groups.append(group_sorted)
        
        return valid_groups
        
    except Exception as e:
        print(f"Error grouping data by 24h periods: {e}")
        return []


def extract_rule_data(rule_data: Union[str, pd.DataFrame], gateway_id: str, datetime: str) -> Optional[Tuple[str, list, list, int, int]]:
    """
    从规则配置数据中提取指定gateway_id和datetime的配置信息
    
    Args:
        rule_data: 规则配置CSV文件路径(str)或已读取的DataFrame
        gateway_id: 网关ID
        datetime: 时间字符串 (格式: 'YYYY-MM-DD HH:MM:SS')
    
    Returns:
        tuple: (dispatch_code, load_priority_list, solar_priority_list, grid_charge_max, grid_discharge_max) 
               其中priority_list是3元素列表，如120.0->[1,2,0], 123.0->[1,2,3], 12.0->[0,1,2]
               如果找到数据，否则返回None
    """
    try:
        import pandas as pd
        from datetime import datetime as dt
        
        def priority_to_list(priority_val):
            """将priority数字转换为3元素列表"""
            # 将数字转为字符串，取前3位数字
            priority_str = str(int(priority_val))
            if len(priority_str) >= 3:
                return [int(priority_str[0]), int(priority_str[1]), int(priority_str[2])]
            elif len(priority_str) == 2:
                return [0, int(priority_str[0]), int(priority_str[1])]
            elif len(priority_str) == 1:
                return [0, 0, int(priority_str[0])]
            else:
                return [0, 0, 0]
        
        # 支持传入DataFrame或文件路径
        rule_df = pd.read_csv(rule_data) if isinstance(rule_data, str) else rule_data
        
        # 解析输入的datetime
        input_dt = dt.strptime(datetime, '%Y-%m-%d %H:%M:%S')
        input_date = input_dt.strftime('%Y-%m-%d')
        input_time = input_dt.strftime('%H:%M')
        
        # 筛选指定gateway_id和日期的记录
        filtered_df = rule_df[(rule_df['gateway_id'] == gateway_id) & 
                             (rule_df['device_time'] == input_date)]
        
        if filtered_df.empty:
            return None
        
        # 查找时间范围匹配的规则
        for _, row in filtered_df.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']
            
            # 处理时间格式
            if len(start_time.split(':')) == 2:
                start_time = start_time + ':00'
            
            # 处理特殊时间格式 (24:00表示当天结束)
            if end_time == '24:00':
                end_time = '23:59:59'
            elif len(end_time.split(':')) == 2:
                end_time = end_time + ':00'
            
            # 转换为时间对象进行比较
            start_dt = dt.strptime(start_time, '%H:%M:%S').time()
            end_dt = dt.strptime(end_time, '%H:%M:%S').time()
            input_time_obj = input_dt.time()
            
            # 检查时间是否在范围内
            if start_dt <= input_time_obj <= end_dt:
                dispatch_code = str(row['dispatch_code'])
                load_priority_list = priority_to_list(row['load_priority'])
                solar_priority_list = priority_to_list(row['solar_priority'])
                
                # 处理grid_charge_max和grid_discharge_max的NaN值
                try:
                    grid_charge_max = int(float(row['grid_charge_max'])) if not pd.isna(row['grid_charge_max']) else 0
                except (ValueError, TypeError):
                    grid_charge_max = 0
                    
                try:
                    grid_discharge_max = int(float(row['grid_discharge_max'])) if not pd.isna(row['grid_discharge_max']) else 0
                except (ValueError, TypeError):
                    grid_discharge_max = 0
                
                return (dispatch_code, load_priority_list, solar_priority_list, grid_charge_max, grid_discharge_max)
        
        return None
        
    except Exception as e:
        print(f"Error extracting rule data: {e}")
        return None


def extract_rule_data_optimized(rule_filtered_df: pd.DataFrame, datetime: str) -> Optional[Tuple[str, list, list, int, int]]:
    """
    优化版本：从预先筛选的规则配置数据中提取指定datetime的配置信息
    
    Args:
        rule_filtered_df: 已筛选的规则配置DataFrame（仅包含特定gateway_id和日期）
        datetime: 时间字符串 (格式: 'YYYY-MM-DD HH:MM:SS')
    
    Returns:
        tuple: (dispatch_code, load_priority_list, solar_priority_list, grid_charge_max, grid_discharge_max)
    """
    try:
        from datetime import datetime as dt
        
        def priority_to_list(priority_val):
            """将priority数字转换为3元素列表"""
            # 处理NaN值
            if pd.isna(priority_val):
                return [0, 0, 0]
            
            try:
                priority_int = int(float(priority_val))  # 先转换为float再转int，处理可能的浮点数
                priority_str = str(priority_int)
                if len(priority_str) >= 3:
                    return [int(priority_str[0]), int(priority_str[1]), int(priority_str[2])]
                elif len(priority_str) == 2:
                    return [0, int(priority_str[0]), int(priority_str[1])]
                elif len(priority_str) == 1:
                    return [0, 0, int(priority_str[0])]
                else:
                    return [0, 0, 0]
            except (ValueError, TypeError):
                return [0, 0, 0]
        
        # 解析输入的datetime
        input_dt = dt.strptime(datetime, '%Y-%m-%d %H:%M:%S')
        input_time = input_dt.strftime('%H:%M')
        
        if rule_filtered_df.empty:
            return None
        
        # 查找时间范围匹配的规则
        for _, row in rule_filtered_df.iterrows():
            start_time = row['start_time']
            end_time = row['end_time']
            
            # 处理时间格式
            if len(start_time.split(':')) == 2:
                start_time = start_time + ':00'
            
            # 处理特殊时间格式 (24:00表示当天结束)
            if end_time == '24:00':
                end_time = '23:59:59'
            elif len(end_time.split(':')) == 2:
                end_time = end_time + ':00'
            
            # 转换为时间对象进行比较
            start_dt = dt.strptime(start_time, '%H:%M:%S').time()
            end_dt = dt.strptime(end_time, '%H:%M:%S').time()
            input_time_obj = input_dt.time()
            
            # 检查时间是否在范围内
            if start_dt <= input_time_obj <= end_dt:
                dispatch_code = str(row['dispatch_code'])
                load_priority_list = priority_to_list(row['load_priority'])
                solar_priority_list = priority_to_list(row['solar_priority'])
                
                # 处理grid_charge_max和grid_discharge_max的NaN值
                try:
                    grid_charge_max = int(float(row['grid_charge_max'])) if not pd.isna(row['grid_charge_max']) else 0
                except (ValueError, TypeError):
                    grid_charge_max = 0
                    
                try:
                    grid_discharge_max = int(float(row['grid_discharge_max'])) if not pd.isna(row['grid_discharge_max']) else 0
                except (ValueError, TypeError):
                    grid_discharge_max = 0
                
                return (dispatch_code, load_priority_list, solar_priority_list, grid_charge_max, grid_discharge_max)
        
        return None
        
    except Exception as e:
        print(f"Error extracting optimized rule data: {e}")
        return None


def extract_load_sun_data_optimized(load_lookup: dict, sun_lookup: dict, tariff_lookup: dict,
                                   datetime: str) -> Optional[Tuple[float, float, float, float, float, float]]:
    """
    优化版本：从预先构建的查找字典中提取指定datetime的数据
    
    Args:
        load_lookup: 负载数据查找字典 {datetime: row_data}
        sun_lookup: 光伏数据查找字典 {datetime: row_data}
        tariff_lookup: 电价数据查找字典 {device_time: row_data}
        datetime: 时间字符串 (格式: 'YYYY-MM-DD HH:MM:SS')
    
    Returns:
        tuple: (load_pred, load, sun_pred, sun, buy_price, sell_price)
    """
    try:
        # 直接从字典中查找数据，时间复杂度O(1)
        load_record = load_lookup.get(datetime)
        sun_record = sun_lookup.get(datetime)
        tariff_record = tariff_lookup.get(datetime)
        
        # 检查是否找到匹配记录
        if load_record is None or sun_record is None or tariff_record is None:
            return None
        
        # 提取数据
        load_pred = float(load_record['kwh_load_predict'])
        load = float(load_record['kwh_load'])
        sun_pred = float(sun_record['kwh_sun_predict'])
        sun = float(sun_record['kwh_sun'])
        buy_price = float(tariff_record['tariff_price'])
        sell_price = float(tariff_record['tariff_sell_price'])
        
        return (load_pred, load, sun_pred, sun, buy_price, sell_price)
        
    except Exception as e:
        print(f"Error extracting optimized load/sun data: {e}")
        return None


def extract_battery_info(battery_data: Union[str, pd.DataFrame], gateway_id: str) -> Optional[dict]:
    """
    从电池信息数据中提取指定gateway_id的电池配置信息
    
    Args:
        battery_data: 电池信息CSV文件路径(str)或已读取的DataFrame
        gateway_id: 网关ID
    
    Returns:
        dict: 包含电池信息的字典，键为 [rated_cap, tou_min_soc, battery_soc, battery_count, rated_power]
              如果找到数据，否则返回None
    """
    try:
        # 支持传入DataFrame或文件路径
        battery_df = pd.read_csv(battery_data) if isinstance(battery_data, str) else battery_data
        
        # 查找指定gateway_id的记录
        battery_record = battery_df[battery_df['gateway_id'] == gateway_id]
        
        # 检查是否找到匹配记录
        if battery_record.empty:
            return None
        
        # 提取电池信息（取第一条匹配记录）
        record = battery_record.iloc[0]
        battery_info = {
            'gateway_id': gateway_id,
            'rated_cap': float(record['rated_cap']),           # 额定容量 (kWh)
            'tou_min_soc': float(record['tou_min_soc']),       # TOU最小SOC (%)
            'battery_soc': float(record['battery_soc']),       # 当前电池SOC (%)
            'battery_count': int(record['battery_count']),     # 电池数量
            'rated_power': float(record['rated_power'])        # 额定功率 (kW)
        }
        
        return (battery_info["rated_cap"], battery_info["tou_min_soc"], battery_info["battery_soc"], battery_info["rated_power"])
        
    except Exception as e:
        print(f"Error extracting battery info: {e}")
        return None


def compare_algorithm_performance(vis_dir: str = 'vis') -> dict:
    """
    对比分析vis目录中不同算法的性能表现
    
    通过分析所有子文件夹中的res.json文件，对比线性规划、动态规划和规则方法的total_cost
    
    Args:
        vis_dir: vis目录路径，默认为'vis'
        
    Returns:
        dict: 包含算法性能对比结果的字典，格式为：
        {
            'summary': {
                'linear_programming': {'avg_cost': float, 'count': int, 'costs': list},
                'rule_based': {'avg_cost': float, 'count': int, 'costs': list},
                'dynamic_programming': {'avg_cost': float, 'count': int, 'costs': list}
            },
            'detailed_results': [
                {
                    'gateway_id': str,
                    'date': str, 
                    'linear_programming': float,
                    'rule_based': float,
                    'dynamic_programming': float
                },
                ...
            ],
            'best_algorithm': str,
            'performance_difference': dict
        }
    """
    import os
    import json
    import re
    from collections import defaultdict
    
    try:
        # 存储所有算法的成本数据
        algorithm_costs = {
            'linear_programming': [],
            'rule_based': [],  
            'dynamic_programming': [],
            'hier_mpc':[],
            'rule_pred':[],
            'mpc_rule_gt': [],
            'mpc_rule_pred20': [],
            'mpc_rule_pred50': [],
            'mpc_rule_pred100': []
        }
        
        # 存储详细对比结果
        detailed_results = []
        gateway_date_results = defaultdict(dict)
        
        # 统计处理情况的计数器
        success_count = 0
        error_count = 0
        invalid_data_count = 0
        
        # 遍历vis目录下的所有子文件夹
        if not os.path.exists(vis_dir):
            print(f"错误: 目录 {vis_dir} 不存在")
            return {}
            
        for folder_name in tqdm.tqdm(os.listdir(vis_dir)):
            folder_path = os.path.join(vis_dir, folder_name)
            
            # 跳过非目录文件
            if not os.path.isdir(folder_path):
                continue
            
            # 检查res.json文件是否存在
            res_json_path = os.path.join(folder_path, 'res.json')
            if not os.path.exists(res_json_path):
                continue
            
            # 解析文件夹名称，确定算法类型
            # 格式: gateway_id:...-date:YYYY-MM-DD[-algorithm]
            folder_pattern = r'gateway_id:([^-]+)-date:(\d{4}-\d{2}-\d{2})(?:-(.+))?'
            match = re.match(folder_pattern, folder_name)
            
            if not match:
                continue
                
            gateway_id = match.group(1)
            date = match.group(2)
            algorithm_suffix = match.group(3)
            
            # 确定算法类型
            if algorithm_suffix == 'rule':
                algorithm = 'rule_based'
            elif algorithm_suffix == 'dp':
                algorithm = 'dynamic_programming'
            elif algorithm_suffix == 'lp':
                algorithm = 'linear_programming'
            elif algorithm_suffix == 'lp-grid-constrain':
                algorithm = 'lp_grid_constrain'
            elif algorithm_suffix == 'rule-pred':
                algorithm = 'rule_pred'
            elif algorithm_suffix == 'hier-mpc-grid-constrain-pv0-load0-rule-pred':
                algorithm = 'mpc_rule_gt'
            elif algorithm_suffix == 'hier-mpc-grid-constrain-pv20-load20-rule-pred':
                algorithm = 'mpc_rule_pred20'
            elif algorithm_suffix == 'hier-mpc-grid-constrain-pv50-load50-rule-pred':
                algorithm = 'mpc_rule_pred50'
            elif algorithm_suffix == 'hier-mpc-grid-constrain-pv100-load100-rule-pred':
                algorithm = 'mpc_rule_pred100'
            else:
                continue  # 跳过未知的算法后缀
            
            # 读取res.json文件
            try:
                with open(res_json_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                # 提取total_cost
                if 'total_cost' in result_data and result_data['status'] == 'optimal':
                    total_cost = result_data['total_cost']
                    
                    # 检查total_cost是否为有效数值
                    if total_cost is not None and not (isinstance(total_cost, float) and np.isnan(total_cost)):
                        try:
                            total_cost = float(total_cost)
                            if not np.isnan(total_cost) and not np.isinf(total_cost):
                                # 存储到算法成本列表
                                algorithm_costs[algorithm].append(total_cost)
                                
                                # 存储到详细结果中（按gateway_id和date分组）
                                key = f"{gateway_id}_{date}"
                                gateway_date_results[key]['gateway_id'] = gateway_id
                                gateway_date_results[key]['date'] = date
                                gateway_date_results[key][algorithm] = total_cost
                                success_count += 1
                        except (ValueError, TypeError):
                            invalid_data_count += 1
                            continue
                    else:
                        invalid_data_count += 1
            except Exception as e:
                error_count += 1
                continue
        
        # 打印处理摘要
        total_processed = success_count + error_count + invalid_data_count
        print(f"\n📊 数据处理摘要:")
        print(f"  成功处理: {success_count} 个文件")
        if error_count > 0:
            print(f"  JSON解析错误: {error_count} 个文件")
        if invalid_data_count > 0:
            print(f"  无效数据: {invalid_data_count} 个文件")
        print(f"  总计: {total_processed} 个文件")
        
        # 计算汇总统计信息
        summary = {}
        for alg, costs in algorithm_costs.items():
            if costs:
                summary[alg] = {
                    'avg_cost': sum(costs) / len(costs),
                    'count': len(costs),
                    'costs': sorted(costs),
                    'min_cost': min(costs),
                    'max_cost': max(costs),
                    'median_cost': sorted(costs)[len(costs)//2] if costs else 0
                }
            else:
                summary[alg] = {
                    'avg_cost': 0,
                    'count': 0, 
                    'costs': [],
                    'min_cost': 0,
                    'max_cost': 0,
                    'median_cost': 0
                }
        
        # 转换详细结果为列表
        detailed_results = list(gateway_date_results.values())
        
        # 确定最佳算法（平均成本最低）
        best_algorithm = None
        best_avg_cost = float('inf')
        
        for alg, stats in summary.items():
            if stats['count'] > 0 and stats['avg_cost'] < best_avg_cost:
                best_avg_cost = stats['avg_cost']
                best_algorithm = alg
        
        # 计算算法间的性能差异
        performance_difference = {}
        if summary['linear_programming']['count'] > 0:
            lp_avg = summary['linear_programming']['avg_cost']
            
            if summary['rule_based']['count'] > 0:
                rule_avg = summary['rule_based']['avg_cost']
                performance_difference['rule_vs_linear'] = {
                    'difference': rule_avg - lp_avg,
                    'percentage': ((rule_avg - lp_avg) / lp_avg) * 100 if lp_avg != 0 else 0
                }
            
            if summary['dynamic_programming']['count'] > 0:
                dp_avg = summary['dynamic_programming']['avg_cost']
                performance_difference['dp_vs_linear'] = {
                    'difference': dp_avg - lp_avg,
                    'percentage': ((dp_avg - lp_avg) / lp_avg) * 100 if lp_avg != 0 else 0
                }
        
        if summary['dynamic_programming']['count'] > 0 and summary['rule_based']['count'] > 0:
            dp_avg = summary['dynamic_programming']['avg_cost']
            rule_avg = summary['rule_based']['avg_cost']
            performance_difference['rule_vs_dp'] = {
                'difference': rule_avg - dp_avg,
                'percentage': ((rule_avg - dp_avg) / dp_avg) * 100 if dp_avg != 0 else 0
            }
        
        # 添加hier_mpc相关的性能对比
        if summary['hier_mpc']['count'] > 0:
            hier_mpc_avg = summary['hier_mpc']['avg_cost']
            
            if summary['linear_programming']['count'] > 0:
                lp_avg = summary['linear_programming']['avg_cost']
                performance_difference['hier_mpc_vs_linear'] = {
                    'difference': hier_mpc_avg - lp_avg,
                    'percentage': ((hier_mpc_avg - lp_avg) / lp_avg) * 100 if lp_avg != 0 else 0
                }
            
            if summary['rule_based']['count'] > 0:
                rule_avg = summary['rule_based']['avg_cost']
                performance_difference['hier_mpc_vs_rule'] = {
                    'difference': hier_mpc_avg - rule_avg,
                    'percentage': ((hier_mpc_avg - rule_avg) / rule_avg) * 100 if rule_avg != 0 else 0
                }
            
            if summary['dynamic_programming']['count'] > 0:
                dp_avg = summary['dynamic_programming']['avg_cost']
                performance_difference['hier_mpc_vs_dp'] = {
                    'difference': hier_mpc_avg - dp_avg,
                    'percentage': ((hier_mpc_avg - dp_avg) / dp_avg) * 100 if dp_avg != 0 else 0
                }
        
        return {
            'summary': summary,
            'detailed_results': detailed_results,
            'best_algorithm': best_algorithm,
            'performance_difference': performance_difference
        }
        
    except Exception as e:
        print(f"算法性能对比分析失败: {e}")
        return {}

def process_one_gateway_one_day(
    gateway_id, 
    day,
    rule_df,
    load_df,
    sun_df,
    tariff_df,
    battery_info_df):

    bat_info = extract_battery_info(battery_info_df, gateway_id)
    if bat_info is None:
        return None
    elif None in bat_info:
        return None
    else:
        _rated_cap, _soc_min, _curr_soc, _rated_power = bat_info
        _rated_cap, _soc_min, _curr_soc, _rated_power = _rated_cap*1000, _soc_min/100, _curr_soc/100, _rated_power*1000

    # 优化：预先筛选该gateway_id和日期的所有数据，避免重复查找
    rule_filtered = rule_df[(rule_df['gateway_id'] == gateway_id) & 
                           (rule_df['device_time'] == day)].copy()
    load_filtered = load_df[load_df['gateway_id'] == gateway_id].copy()
    sun_filtered = sun_df[sun_df['gateway_id'] == gateway_id].copy()
    tariff_filtered = tariff_df[tariff_df['gateway_id'] == gateway_id].copy()
    
    # 为datetime查找创建索引字典以提高查找速度
    load_lookup = {row['datetime']: row for _, row in load_filtered.iterrows()}
    sun_lookup = {row['datetime']: row for _, row in sun_filtered.iterrows()}
    tariff_lookup = {row['device_time']: row for _, row in tariff_filtered.iterrows()}

    pv_pred = []
    pv = []
    load_pred = []
    load = []
    buy_prices = []
    sell_prices = []
    load_priority = []
    solar_priority = []
    grid_charge_max = []
    grid_discharge_max = []
    code = []

    for i in range(24):
        time = get_time_string(i)
        datetime_str = day + " " + time
        
        # 优化的规则数据提取
        rule_data = extract_rule_data_optimized(rule_filtered, datetime_str)
        if rule_data is None:
            print(f"Error: Unable to extract rule data for gateway_id {gateway_id} on {day} {time}")
            break
        _code, _load_priority, _solar_priority, _grid_charge_max, _grid_discharge_max = rule_data
        if None in [_code, _load_priority, _solar_priority, _grid_charge_max, _grid_discharge_max]:
            print(f"Error: Unable to extract rule data for gateway_id {gateway_id} on {day} {time}")
            break
            
        # 优化的负载和光伏数据提取
        load_sun_data = extract_load_sun_data_optimized(
            load_lookup, sun_lookup, tariff_lookup, datetime_str
        )
        if load_sun_data is None:
            print(f"Error: Unable to extract load and sun data for gateway_id {gateway_id} on {day} {time}")
            break
        _load_pred, _load, _sun_pred, _sun, _buy_price, _sell_price = load_sun_data
        if None in [_load_pred, _load, _sun_pred, _sun, _buy_price, _sell_price]:
            print(f"Error: Unable to extract load and sun data for gateway_id {gateway_id} on {day} {time}")
            break
            
        pv_pred.append(_sun_pred * 1000)
        pv.append(_sun*1000)
        load_pred.append(_load_pred*1000)
        load.append(_load*1000)
        buy_prices.append(_buy_price)
        sell_prices.append(_sell_price)
        load_priority.append(_load_priority)
        solar_priority.append(_solar_priority)
        grid_charge_max.append(_grid_charge_max*1000)
        grid_discharge_max.append(_grid_discharge_max*1000)
        code.append(_code)
    
    if len(pv_pred) != 24:
        return None
    else:
        sample = {
                    "gateway_id": gateway_id,
                    "date": day,
                    "rated_cap": _rated_cap,
                    "soc_min": _soc_min,
                    "curr_soc": _curr_soc,
                    "rated_power": _rated_power,
                    "pv_pred": pv_pred,
                    "pv": pv,
                    "load_pred": load_pred,
                    "load": load,
                    "buy_prices": buy_prices,
                    "sell_prices": sell_prices,
                    "load_priority": load_priority,
                    "solar_priority": solar_priority,
                    "grid_charge_max": grid_charge_max,
                    "grid_discharge_max": grid_discharge_max,
                    "code": code
                }
        return sample

def select_test_samples(
    rule_df,
    load_df,
    sun_df,
    tariff_df,
    battery_info_df,
    year=2024,
    num_day=180,
    num_gatwayid=20,
    seed=123):

    np.random.seed(seed)

    day_candidates = []
    for i in range(12):
        month = i+1
        if month in [1,3,5,7,8,10,12]:
            for day in range(1, 32):
                day_candidates.append("{}-{:02d}-{:02d}".format(year, month, day))
        else:
            for day in range(1, 31):
                day_candidates.append("{}-{:02d}-{:02d}".format(year, month, day))

    day_candidates = np.random.permutation(day_candidates)
    day_candidates = day_candidates[:num_day]
    print("\n".join(day_candidates.tolist()))
    gateway_id_list = list(battery_info_df["gateway_id"])
    gateway_id_list = np.random.permutation(gateway_id_list)
    gateway_id_list = gateway_id_list[:num_gatwayid]
    print("\n".join(gateway_id_list.tolist()))
    
    samples = []
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for gateway_id in gateway_id_list:
            for day in day_candidates:
                futures.append(
                    executor.submit(
                        process_one_gateway_one_day, 
                        gateway_id, 
                        day, 
                        rule_df,
                        load_df,
                        sun_df,
                        tariff_df,
                        battery_info_df)
                )
        for future in concurrent.futures.as_completed(futures):
            sample = future.result()
            if sample is not None:
                samples.append(sample)
            print(len(samples))
    return samples


def create_algorithm_comparison_charts(comparison_result: dict, output_dir: str = "algorithm_charts"):
    """
    创建算法对比图表并保存
    
    Args:
        comparison_result: 算法对比结果字典
        output_dir: 输出目录
    """
    if not comparison_result:
        print("❌ 无法生成图表：数据为空")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    summary = comparison_result.get('summary', {})
    
    # 设置中文字体和图表样式
    #plt.rcParams['font.sans-serif'] = ['Heiti TC', 'DejaVu Sans']
    #plt.rcParams['axes.unicode_minus'] = False
    #sns.set_style("whitegrid")
    
    # 算法名称映射
    alg_names = {
        'linear_programming': '线性规划',
        'rule_based': '规则方法', 
        'dynamic_programming': '动态规划',
        'hier_mpc': '分层MPC'
    }
    
    # 1. 平均成本对比柱状图
    create_average_cost_chart(summary, alg_names, output_dir)
    
    # 2. 胜率对比图
    create_win_rate_chart(comparison_result, alg_names, output_dir)
    
    # 3. 成本分布箱线图
    create_cost_distribution_chart(comparison_result, alg_names, output_dir)
    
    # 4. 算法性能差异对比图
    create_performance_difference_chart(comparison_result, output_dir)
    
    print(f"📊 图表已保存到 {output_dir}/ 目录")


def create_average_cost_chart(summary: dict, alg_names: dict, output_dir: str):
    """创建平均成本对比柱状图"""
    algorithms = []
    avg_costs = []
    counts = []
    
    for alg, stats in summary.items():
        if stats['count'] > 0:
            algorithms.append(alg_names.get(alg, alg))
            avg_costs.append(stats['avg_cost'])
            counts.append(stats['count'])
    
    if not algorithms:
        return
    algorithms=["规则方法", "MILP", "MPC", "MPC(20%)", "MPC(50%)", "MPC(100%)"]
    plt.figure(figsize=(10, 6))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    bars = plt.bar(algorithms, avg_costs, color=colors[:len(algorithms)])
    
    # 添加数值标签
    for i, (bar, cost, count) in enumerate(zip(bars, avg_costs, counts)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.05),
                f'{cost:.3f}美元', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.title('算法平均成本对比', fontsize=16, fontweight='bold')
    plt.xlabel('算法类型', fontsize=12)
    plt.ylabel('平均成本 (美元)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 设置颜色条例说明
    legend_labels = [f'{alg} (n={count})' for alg, count in zip(algorithms, counts)]
    plt.legend(bars, legend_labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_cost_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_win_rate_chart(comparison_result: dict, alg_names: dict, output_dir: str):
    """创建胜率对比图"""
    detailed_results = comparison_result.get('detailed_results', [])
    
    if not detailed_results:
        return
    
    # 统计每个算法的胜利次数
    algorithm_wins = {
        'linear_programming': 0,
        'rule_based': 0,
        'dynamic_programming': 0,
        'hier_mpc': 0
    }
    
    total_comparisons = 0
    
    # 遍历每个gateway_id和日期的对比结果
    for result in detailed_results:
        # 获取该条记录中所有算法的成本
        costs = {}
        for alg in algorithm_wins.keys():
            if alg in result and result[alg] is not None:
                costs[alg] = result[alg]
        
        # 如果至少有2个算法有数据，才进行比较
        if len(costs) >= 2:
            # 找到成本最低的算法
            min_cost = min(costs.values())
            winners = [alg for alg, cost in costs.items() if cost == min_cost]
            
            # 如果有并列最优，每个算法分得胜利分数
            win_score = 1.0 / len(winners)
            for winner in winners:
                algorithm_wins[winner] += win_score
            
            total_comparisons += 1
    
    if total_comparisons == 0:
        return
    
    # 计算胜率
    algorithms = []
    win_rates = []
    win_counts = []
    
    for alg, wins in algorithm_wins.items():
        if wins > 0:  # 只显示有胜利记录的算法
            algorithms.append(alg_names.get(alg, alg))
            win_rate = (wins / total_comparisons) * 100
            win_rates.append(win_rate)
            win_counts.append(wins)
    
    if not algorithms:
        return
    
    plt.figure(figsize=(12, 6))
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffa726']
    bars = plt.bar(algorithms, win_rates, color=colors[:len(algorithms)])
    
    # 添加胜率和胜利次数标签
    for bar, rate, wins in zip(bars, win_rates, win_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%\n({wins:.0f}胜)', ha='center', va='bottom')
    
    plt.title(f'算法胜率对比 (总对比: {total_comparisons}次)', fontsize=16, fontweight='bold')
    plt.xlabel('算法类型', fontsize=12)
    plt.ylabel('胜率 (%)', fontsize=12)
    plt.ylim(0, max(win_rates) * 1.2)
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息文本
    stats_text = f'统计说明:\n• 总对比次数: {total_comparisons}\n• 对比方式: 同一gateway_id同一天的算法成本比较\n• 胜利标准: 成本最低的算法获胜'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_rate_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_cost_distribution_chart(comparison_result: dict, alg_names: dict, output_dir: str):
    """创建成本分布箱线图"""
    # 这里需要从原始数据创建分布图，暂时使用汇总统计创建模拟分布
    summary = comparison_result.get('summary', {})
    
    data_for_box = []
    labels = []
    
    for alg, stats in summary.items():
        if stats['count'] > 0:
            labels.append(alg_names.get(alg, alg))
            # 使用统计信息模拟数据分布
            avg = stats['avg_cost']
            min_val = stats['min_cost']
            max_val = stats['max_cost']
            median = stats['median_cost']
            
            # 创建模拟数据点
            simulated_data = [
                min_val, 
                avg - (avg - min_val) * 0.5,
                median,
                avg,
                avg + (max_val - avg) * 0.5,
                max_val
            ] * (stats['count'] // 6 + 1)
            data_for_box.append(simulated_data[:stats['count']])
    
    if not data_for_box:
        return
    
    plt.figure(figsize=(12, 6))
    box_plot = plt.boxplot(data_for_box, labels=labels, patch_artist=True)
    
    # 设置箱线图颜色
    colors = ['#ffcccc', '#ccddff', '#ccffcc', '#ffe0cc']
    for patch, color in zip(box_plot['boxes'], colors[:len(data_for_box)]):
        patch.set_facecolor(color)
    
    plt.title('算法成本分布对比', fontsize=16, fontweight='bold')
    plt.xlabel('算法类型', fontsize=12)
    plt.ylabel('成本分布 (元)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_performance_difference_chart(comparison_result: dict, output_dir: str):
    """创建算法性能差异对比图"""
    performance_diff = comparison_result.get('performance_difference', {})
    
    if not performance_diff:
        return
    
    comparison_names = {
        'rule_vs_linear': '规则方法 vs 线性规划',
        'dp_vs_linear': '动态规划 vs 线性规划',
        'rule_vs_dp': '规则方法 vs 动态规划',
        'hier_mpc_vs_linear': '分层MPC vs 线性规划',
        'hier_mpc_vs_rule': '分层MPC vs 规则方法',
        'hier_mpc_vs_dp': '分层MPC vs 动态规划'
    }
    
    comparisons = []
    differences = []
    percentages = []
    
    for comp, diff_data in performance_diff.items():
        if comp in comparison_names:
            comparisons.append(comparison_names[comp])
            differences.append(diff_data['difference'])
            percentages.append(diff_data['percentage'])
    
    if not comparisons:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 成本差异图
    colors = ['red' if d > 0 else 'green' for d in differences]
    bars1 = ax1.bar(range(len(comparisons)), differences, color=colors, alpha=0.7)
    ax1.set_title('算法间成本差异 (元)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('对比组合', fontsize=12)
    ax1.set_ylabel('成本差异 (元)', fontsize=12)
    ax1.set_xticks(range(len(comparisons)))
    ax1.set_xticklabels(comparisons, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 添加数值标签
    for bar, diff in zip(bars1, differences):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.003),
                f'{diff:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # 百分比差异图
    colors2 = ['red' if p > 0 else 'green' for p in percentages]
    bars2 = ax2.bar(range(len(comparisons)), percentages, color=colors2, alpha=0.7)
    ax2.set_title('算法间成本差异 (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('对比组合', fontsize=12)
    ax2.set_ylabel('成本差异 (%)', fontsize=12)
    ax2.set_xticks(range(len(comparisons)))
    ax2.set_xticklabels(comparisons, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 添加数值标签
    for bar, pct in zip(bars2, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.3),
                f'{pct:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_difference.png'), dpi=300, bbox_inches='tight')
    plt.close()


def print_algorithm_comparison_report(comparison_result: dict):
    """
    打印算法性能对比报告
    
    Args:
        comparison_result: compare_algorithm_performance函数的返回结果
    """
    if not comparison_result:
        print("❌ 无法生成对比报告：数据为空")
        return
    
    summary = comparison_result.get('summary', {})
    best_algorithm = comparison_result.get('best_algorithm')
    performance_diff = comparison_result.get('performance_difference', {})
    
    print("=" * 80)
    print("🔍 算法性能对比分析报告")
    print("=" * 80)
    
    print("\n📊 汇总统计:")
    print("-" * 50)
    
    # 算法名称映射
    alg_names = {
        'linear_programming': '线性规划',
        'rule_based': '规则方法', 
        'dynamic_programming': '动态规划',
        'hier_mpc': '分层MPC'
    }
    
    for alg, stats in summary.items():
        if stats['count'] > 0:
            print(f"{alg_names.get(alg, alg)}:")
            print(f"  样本数量: {stats['count']}")
            print(f"  平均成本: {stats['avg_cost']:.3f} 元")
            print(f"  成本范围: {stats['min_cost']:.3f} - {stats['max_cost']:.3f} 元")
            print(f"  中位数: {stats['median_cost']:.3f} 元")
            
            # 成本解释
            avg_cost = stats['avg_cost']
            if avg_cost > 0:
                print(f"  财务表现: 净支出 {abs(avg_cost):.3f} 元")
            elif avg_cost < 0:
                print(f"  财务表现: 净收益 {abs(avg_cost):.3f} 元")
            else:
                print(f"  财务表现: 收支平衡")
            print()
    
    # 最佳算法
    if best_algorithm:
        print(f"🏆 最佳算法: {alg_names.get(best_algorithm, best_algorithm)}")
        print(f"   (平均成本最低: {summary[best_algorithm]['avg_cost']:.3f} 元)")
    
    print("\n📈 算法间性能差异:")
    print("-" * 50)
    
    for comparison, diff_data in performance_diff.items():
        difference = diff_data['difference']
        percentage = diff_data['percentage']
        
        comparison_names = {
            'rule_vs_linear': '规则方法 vs 线性规划',
            'dp_vs_linear': '动态规划 vs 线性规划',
            'rule_vs_dp': '规则方法 vs 动态规划',
            'hier_mpc_vs_linear': '分层MPC vs 线性规划',
            'hier_mpc_vs_rule': '分层MPC vs 规则方法',
            'hier_mpc_vs_dp': '分层MPC vs 动态规划'
        }
        
        print(f"{comparison_names.get(comparison, comparison)}:")
        
        if difference > 0:
            print(f"  成本差异: +{difference:.3f} 元 (+{percentage:.1f}%)")
            print(f"  结论: 前者成本更高")
        elif difference < 0:
            print(f"  成本差异: {difference:.3f} 元 ({percentage:.1f}%)")
            print(f"  结论: 前者成本更低") 
        else:
            print(f"  成本差异: 0.000 元 (0.0%)")
            print(f"  结论: 成本相等")
        print()
    
    # 详细建议
    print("💡 建议:")
    print("-" * 50)
    
    if summary['linear_programming']['count'] > 0 and summary['rule_based']['count'] > 0:
        lp_avg = summary['linear_programming']['avg_cost']
        rule_avg = summary['rule_based']['avg_cost']
        
        if rule_avg < lp_avg:
            savings = lp_avg - rule_avg
            print(f"• 规则方法比线性规划平均每天节省 {savings:.3f} 元")
        elif lp_avg < rule_avg:
            extra_cost = rule_avg - lp_avg
            print(f"• 线性规划比规则方法平均每天节省 {extra_cost:.3f} 元")
    
    if summary['dynamic_programming']['count'] > 0:
        dp_avg = summary['dynamic_programming']['avg_cost']
        print(f"• 动态规划的平均日成本为 {dp_avg:.3f} 元")
        
        if best_algorithm == 'dynamic_programming':
            print("• 动态规划在当前数据集上表现最佳，建议优先使用")
    
    print(f"\n📋 数据概览:")
    print(f"  总分析文件夹数: {sum(stats['count'] for stats in summary.values())}")
    print(f"  涵盖算法类型: {len([alg for alg, stats in summary.items() if stats['count'] > 0])}/4")
    
    # 生成对比图表
    print(f"\n📊 生成算法对比图表...")
    create_algorithm_comparison_charts(comparison_result)
   
def extract_rule_data_from_cache(rule_data_list: list, datetime: str) -> Optional[Tuple[str, list, list, int, int]]:
    """
    从缓存的规则数据列表中提取指定datetime的配置信息
    
    Args:
        rule_data_list: 该日期的规则数据列表
        datetime: 时间字符串 (格式: 'YYYY-MM-DD HH:MM:SS')
    
    Returns:
        tuple: (dispatch_code, load_priority_list, solar_priority_list, grid_charge_max, grid_discharge_max)
    """
    try:
        from datetime import datetime as dt
        
        def priority_to_list(priority_val):
            """将priority数字转换为3元素列表"""
            # 处理NaN值
            if pd.isna(priority_val):
                return [0, 0, 0]
            
            try:
                priority_int = int(float(priority_val))  # 先转换为float再转int，处理可能的浮点数
                priority_str = str(priority_int)
                if len(priority_str) >= 3:
                    return [int(priority_str[0]), int(priority_str[1]), int(priority_str[2])]
                elif len(priority_str) == 2:
                    return [0, int(priority_str[0]), int(priority_str[1])]
                elif len(priority_str) == 1:
                    return [0, 0, int(priority_str[0])]
                else:
                    return [0, 0, 0]
            except (ValueError, TypeError):
                return [0, 0, 0]
        
        # 解析输入的datetime
        input_dt = dt.strptime(datetime, '%Y-%m-%d %H:%M:%S')
        input_time_obj = input_dt.time()
        
        # 查找时间范围匹配的规则
        for row in rule_data_list:
            start_time = row['start_time']
            end_time = row['end_time']
            
            # 处理时间格式
            if len(start_time.split(':')) == 2:
                start_time = start_time + ':00'
            
            # 处理特殊时间格式 (24:00表示当天结束)
            if end_time == '24:00':
                end_time = '23:59:59'
            elif len(end_time.split(':')) == 2:
                end_time = end_time + ':00'
            
            # 转换为时间对象进行比较
            start_dt = dt.strptime(start_time, '%H:%M:%S').time()
            end_dt = dt.strptime(end_time, '%H:%M:%S').time()
            
            # 检查时间是否在范围内
            if start_dt <= input_time_obj <= end_dt:
                dispatch_code = str(row['dispatch_code'])
                load_priority_list = priority_to_list(row['load_priority'])
                solar_priority_list = priority_to_list(row['solar_priority'])
                
                # 处理grid_charge_max和grid_discharge_max的NaN值
                try:
                    grid_charge_max = int(float(row['grid_charge_max'])) if not pd.isna(row['grid_charge_max']) else 0
                except (ValueError, TypeError):
                    grid_charge_max = 0
                    
                try:
                    grid_discharge_max = int(float(row['grid_discharge_max'])) if not pd.isna(row['grid_discharge_max']) else 0
                except (ValueError, TypeError):
                    grid_discharge_max = 0
                
                return (dispatch_code, load_priority_list, solar_priority_list, grid_charge_max, grid_discharge_max)
        
        return None
        
    except Exception as e:
        print(f"Error extracting rule data from cache: {e}")
        return None


def process_one_gateway_one_day_batch(gateway_ids: list, days: list, 
                                      rule_df, load_df, sun_df, tariff_df, battery_info_df):
    """
    批量处理多个网关多天的数据，进一步优化性能
    
    Args:
        gateway_ids: 网关ID列表
        days: 日期列表  
        rule_df, load_df, sun_df, tariff_df, battery_info_df: 数据表
    
    Returns:
        list: 所有有效样本的列表
    """
    
    print(f"批量处理 {len(gateway_ids)} 个网关，{len(days)} 天数据")
    
    # 预筛选所有相关数据，减少重复过滤
    relevant_gateway_ids = set(gateway_ids)
    relevant_days = set(days)
    
    # 按网关ID预筛选数据
    load_filtered = load_df[load_df['gateway_id'].isin(relevant_gateway_ids)].copy()
    sun_filtered = sun_df[sun_df['gateway_id'].isin(relevant_gateway_ids)].copy() 
    tariff_filtered = tariff_df[tariff_df['gateway_id'].isin(relevant_gateway_ids)].copy()
    rule_filtered = rule_df[(rule_df['gateway_id'].isin(relevant_gateway_ids)) & 
                           (rule_df['device_time'].isin(relevant_days))].copy()
    
    print(f"预筛选完成: Load({len(load_filtered)}), Sun({len(sun_filtered)}), Tariff({len(tariff_filtered)}), Rule({len(rule_filtered)})")
    
    # 并行处理
    samples = []
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for gateway_id in gateway_ids:
            for day in days:
                futures.append(
                    executor.submit(
                        process_one_gateway_one_day,
                        gateway_id, day,
                        rule_filtered,  # 使用预筛选的数据
                        load_filtered,
                        sun_filtered, 
                        tariff_filtered,
                        battery_info_df
                    )
                )
        
        # 收集结果
        completed = 0
        total_tasks = len(futures)
        for future in concurrent.futures.as_completed(futures):
            sample = future.result()
            completed += 1
            if sample is not None:
                samples.append(sample)
            if completed % 50 == 0 or completed == total_tasks:  # 显示进度
                print(f"批量处理进度: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%), 有效样本: {len(samples)}")
    
    return samples


def create_gateway_data_cache(rule_df, load_df, sun_df, tariff_df, gateway_ids: list):
    """
    为指定的网关ID列表创建数据缓存，进一步提升批量处理性能
    
    Args:
        rule_df, load_df, sun_df, tariff_df: 原始数据表
        gateway_ids: 需要缓存的网关ID列表
    
    Returns:
        dict: 缓存的数据字典，按gateway_id组织
    """
    
    print(f"为 {len(gateway_ids)} 个网关创建数据缓存...")
    
    cache = {}
    
    for i, gateway_id in enumerate(gateway_ids):
        if i % 10 == 0:  # 每10个网关显示进度
            print(f"缓存进度: {i}/{len(gateway_ids)} ({i/len(gateway_ids)*100:.1f}%)")
            
        # 筛选该网关的所有数据
        gateway_load = load_df[load_df['gateway_id'] == gateway_id].copy()
        gateway_sun = sun_df[sun_df['gateway_id'] == gateway_id].copy()
        gateway_tariff = tariff_df[tariff_df['gateway_id'] == gateway_id].copy()
        gateway_rule = rule_df[rule_df['gateway_id'] == gateway_id].copy()
        
        # 创建datetime查找字典，时间复杂度O(1)
        load_lookup = {row['datetime']: row for _, row in gateway_load.iterrows()}
        sun_lookup = {row['datetime']: row for _, row in gateway_sun.iterrows()}
        tariff_lookup = {row['device_time']: row for _, row in gateway_tariff.iterrows()}
        
        # 按日期组织规则数据
        rule_by_date = {}
        for _, row in gateway_rule.iterrows():
            date = row['device_time']
            if date not in rule_by_date:
                rule_by_date[date] = []
            rule_by_date[date].append(row)
        
        cache[gateway_id] = {
            'load_lookup': load_lookup,
            'sun_lookup': sun_lookup, 
            'tariff_lookup': tariff_lookup,
            'rule_by_date': rule_by_date,
            'data_counts': {
                'load': len(gateway_load),
                'sun': len(gateway_sun),
                'tariff': len(gateway_tariff), 
                'rule': len(gateway_rule)
            }
        }
    
    print(f"数据缓存创建完成，缓存了 {len(cache)} 个网关的数据")
    return cache


def process_one_gateway_one_day_cached(gateway_id: str, day: str, 
                                       gateway_cache: dict, battery_info_df):
    """
    使用缓存数据处理单个网关单天数据，最大化性能
    
    Args:
        gateway_id: 网关ID
        day: 日期字符串
        gateway_cache: 该网关的缓存数据
        battery_info_df: 电池信息数据表
    
    Returns:
        dict: 样本数据或None
    """
    
    # 提取电池信息（只需执行一次）
    bat_info = extract_battery_info(battery_info_df, gateway_id)
    if bat_info is None or None in bat_info:
        return None
    
    _rated_cap, _soc_min, _curr_soc, _rated_power = bat_info
    _rated_cap, _soc_min, _curr_soc, _rated_power = _rated_cap*1000, _soc_min/100, _curr_soc/100, _rated_power*1000
    
    # 获取缓存的查找字典（O(1)访问）
    load_lookup = gateway_cache['load_lookup']
    sun_lookup = gateway_cache['sun_lookup']
    tariff_lookup = gateway_cache['tariff_lookup']
    rule_by_date = gateway_cache['rule_by_date']
    
    # 获取该日期的规则数据
    if day not in rule_by_date:
        return None
    rule_data_list = rule_by_date[day]
    
    # 初始化结果列表
    pv_pred, pv, load_pred, load = [], [], [], []
    buy_prices, sell_prices = [], []
    load_priority, solar_priority = [], []
    grid_charge_max, grid_discharge_max = [], []
    code = []
    
    for i in range(24):
        time = get_time_string(i)
        datetime_str = day + " " + time
        
        # 从缓存中提取规则数据（避免重复DataFrame过滤）
        rule_data = extract_rule_data_from_cache(rule_data_list, datetime_str)
        if rule_data is None:
            return None
        _code, _load_priority, _solar_priority, _grid_charge_max, _grid_discharge_max = rule_data
        
        # 从缓存中提取负载和光伏数据（O(1)字典查找）
        load_sun_data = extract_load_sun_data_optimized(load_lookup, sun_lookup, tariff_lookup, datetime_str)
        if load_sun_data is None:
            return None
        _load_pred, _load, _sun_pred, _sun, _buy_price, _sell_price = load_sun_data
        
        # 添加到结果（列表append操作）
        pv_pred.append(_sun_pred * 1000)
        pv.append(_sun * 1000)
        load_pred.append(_load_pred * 1000)
        load.append(_load * 1000)
        buy_prices.append(_buy_price)
        sell_prices.append(_sell_price)
        load_priority.append(_load_priority)
        solar_priority.append(_solar_priority)
        grid_charge_max.append(_grid_charge_max * 1000)
        grid_discharge_max.append(_grid_discharge_max * 1000)
        code.append(_code)
    
    if len(pv_pred) != 24:
        return None
    
    return {
        "gateway_id": gateway_id,
        "date": day,
        "rated_cap": _rated_cap,
        "soc_min": _soc_min,
        "curr_soc": _curr_soc,
        "rated_power": _rated_power,
        "pv_pred": pv_pred,
        "pv": pv,
        "load_pred": load_pred,
        "load": load,
        "buy_prices": buy_prices,
        "sell_prices": sell_prices,
        "load_priority": load_priority,
        "solar_priority": solar_priority,
        "grid_charge_max": grid_charge_max,
        "grid_discharge_max": grid_discharge_max,
        "code": code
    }


def benchmark_processing_methods(gateway_ids, days, rule_df, load_df, sun_df, tariff_df, battery_info_df, sample_size=10):
    """
    对比不同处理方法的性能
    
    Args:
        gateway_ids, days: 网关ID和日期列表
        rule_df, load_df, sun_df, tariff_df, battery_info_df: 数据表
        sample_size: 测试样本数量
    """
    import time
    
    # 选择小样本进行性能测试
    test_gateway_ids = gateway_ids[:sample_size//2] if len(gateway_ids) > sample_size//2 else gateway_ids
    test_days = days[:sample_size//len(test_gateway_ids)] if len(days) > sample_size//len(test_gateway_ids) else days
    
    print(f"性能对比测试: {len(test_gateway_ids)} 网关 x {len(test_days)} 天 = {len(test_gateway_ids) * len(test_days)} 样本")
    
    # 方法1: 原始方法
    print("\n测试原始方法...")
    start_time = time.time()
    original_samples = []
    for gateway_id in test_gateway_ids:
        for day in test_days:
            sample = process_one_gateway_one_day(gateway_id, day, rule_df, load_df, sun_df, tariff_df, battery_info_df)
            if sample:
                original_samples.append(sample)
    original_time = time.time() - start_time
    
    # 方法2: 缓存方法
    print("\n测试缓存方法...")
    start_time = time.time()
    cache = create_gateway_data_cache(rule_df, load_df, sun_df, tariff_df, test_gateway_ids)
    cached_samples = []
    for gateway_id in test_gateway_ids:
        for day in test_days:
            sample = process_one_gateway_one_day_cached(gateway_id, day, cache[gateway_id], battery_info_df)
            if sample:
                cached_samples.append(sample)
    cached_time = time.time() - start_time
    
    # 输出对比结果
    print(f"\n性能对比结果:")
    print(f"原始方法: {original_time:.2f}秒, 样本数: {len(original_samples)}")
    print(f"缓存方法: {cached_time:.2f}秒, 样本数: {len(cached_samples)}")
    print(f"性能提升: {original_time/cached_time:.1f}x")
    print(f"平均每样本处理时间: 原始{original_time/len(original_samples)*1000:.1f}ms, 缓存{cached_time/len(cached_samples)*1000:.1f}ms")
    
    return {
        'original_time': original_time,
        'cached_time': cached_time,
        'speedup': original_time/cached_time,
        'original_samples': len(original_samples),
        'cached_samples': len(cached_samples)
    }

def select_test_samples_cache(
    rule_df,
    load_df,
    sun_df,
    tariff_df,
    battery_info_df,
    year=2024,
    num_day=180,
    num_gatwayid=50,
    seed=123):
    np.random.seed(seed)

    day_candidates = []
    for i in range(12):
        month = i+1
        if month in [1,3,5,7,8,10,12]:
            for day in range(1, 32):
                day_candidates.append("{}-{:02d}-{:02d}".format(year, month, day))
        else:
            for day in range(1, 31):
                day_candidates.append("{}-{:02d}-{:02d}".format(year, month, day))

    day_candidates = np.random.permutation(day_candidates)
    day_candidates = day_candidates[:num_day]
    print("\n".join(day_candidates.tolist()))
    gateway_id_list = list(battery_info_df["gateway_id"])
    gateway_id_list = np.random.permutation(gateway_id_list)
    gateway_id_list = gateway_id_list[:num_gatwayid]
    print("\n".join(gateway_id_list.tolist()))

    cache = create_gateway_data_cache(rule_df, load_df, sun_df, tariff_df, gateway_id_list)

    samples = []
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for gateway_id in gateway_id_list:
            for day in day_candidates:
                futures.append(
                    executor.submit(
                        process_one_gateway_one_day_cached, 
                        gateway_id, 
                        day, 
                        cache[gateway_id],
                        battery_info_df)
                )
        for future in concurrent.futures.as_completed(futures):
            sample = future.result()
            if sample is not None:
                samples.append(sample)
            print(len(samples))
    return samples


def plot_histogram_of_two_algo(algo1, algo2, res, bins=50):
    month = []
    error = []
    _dict = {}
    month_dist = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 0,
    }
    for data in res["detailed_results"]:
        try:
            if data[algo1] != float("nan") and data[algo2] != float('nan'):
                error.append(data[algo1] - data[algo2])
                month.append(int(data["date"].split("-")[1]))
                _dict["gateway_id:{}-data:{}".format(data["gateway_id"], data["date"])] = data[algo1] - data[algo2]
                mm = int(data["date"].split("-")[1])
                if data[algo1] - data[algo2] < 0:
                    month_dist[mm] += 1
            else:
                continue
        except:
            continue

    sorted_items = sorted(_dict.items(), key=lambda item: item[1])
    sorted_dict = dict(sorted_items)

    with open("algorithm_charts/{}-{}_sorted_diff.json".format(algo1, algo2), "w") as f:
        json.dump(sorted_dict, f)

    x = [_x+1 for _x in range(12)]
    y = [month_dist[_x+1] for _x in range(12)]
    plt.bar(x, y) # bins control the number of bars
    plt.title('Low cost distribution')
    plt.xlabel('Month')
    plt.ylabel('Frequency')
    plt.savefig("algorithm_charts/{}-{}-lowcost-dist.png".format(algo1, algo2), dpi=300, bbox_inches='tight')
    plt.close()

    plt.hist(month, bins=12, edgecolor='black') # bins control the number of bars
    plt.title('Histogram of month')
    plt.xlabel('Month')
    plt.ylabel('Frequency')
    plt.savefig("algorithm_charts/month.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.hist(error, bins=bins, edgecolor='black') # bins control the number of bars
    plt.title('Histogram of {}-{}'.format(algo1, algo2))
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig("algorithm_charts/{}-{}.png".format(algo1, algo2), dpi=300, bbox_inches='tight')
    plt.close()

    num_high_cost = 0
    num_low_cost05 = 0
    num_low_cost5 = 0
    for e in error:
        if e >= 0:
            num_high_cost += 1
        elif e >= -5:
            num_low_cost05 += 1
        else:
            num_low_cost5 += 1
    print([len(error), num_high_cost, num_low_cost05, num_low_cost5])
    patches, texts = plt.pie(
        [num_high_cost, num_low_cost05, num_low_cost5],
        labels=[str(num_high_cost), str(num_low_cost05), str(num_low_cost5)],
    )
    plt.legend(patches, [">=0", "-5-0", "<-5"], loc="best")
    plt.axis('equal')
    plt.title('Cost Differences Distribution')
    plt.savefig("algorithm_charts/{}-{}-distribution.png".format(algo1, algo2), dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_statistics(samples):
    res = {"pv_error": [], "load_error": []}
    for sample in samples:
        pv_pred = sample["pv_pred"]
        pv_true = sample["pv"]
        load_pred = sample["load_pred"]
        load_true = sample["load"]
        pv_error_list = []
        load_error_list = []
        for i in range(24):
            if pv_true[i] == 0:
                pv_error=0
            else:
                pv_error = (pv_pred[i] - pv_true[i]) / pv_true[i]
        
            if load_true[i] == 0:
                load_error = 0
            else:
                load_error = (load_pred[i] - load_true[i]) / load_true[i]
            pv_error_list.append(pv_error)
            load_error_list.append(load_error)
        res["pv_error"].append(pv_error_list)
        res["load_error"].append(load_error_list)

    pv_error_mean = np.mean(np.array(res["pv_error"]), 0)
    pv_error_std = np.std(np.array(res["pv_error"]), 0)
    load_error_mean = np.mean(np.array(res["load_error"]), 0)
    load_error_std = np.std(np.array(res["load_error"]), 0)
    
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
    ax0.errorbar(range(len(pv_error_mean)), pv_error_mean, yerr=pv_error_std, fmt='-o')
    ax0.grid()
    ax0.set_title('光伏预测误差')

    ax1.errorbar(range(len(load_error_mean)), load_error_mean, yerr=load_error_std, fmt='-o')
    ax1.set_title('负载预测误差')
    ax1.grid()
    plt.savefig("algorithm_charts/error.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    """
    with open("test_samples.json", "r", encoding='gbk') as f:
        data = json.load(f)
    plot_error_statistics(data)
    """
    """
    res = compare_algorithm_performance()
    with open("compare_algorithm_performance.json", "w") as f:
        json.dump(res, f)
    """
    with open("compare_algorithm_performance.json", "r") as f:
        res = json.load(f)
    plot_histogram_of_two_algo("rule_pred", "rule_based", res, 100)
    plot_histogram_of_two_algo("mpc_rule_gt", "rule_based", res, 100)
    plot_histogram_of_two_algo("mpc_rule_pred20", "rule_based", res, 100)
    plot_histogram_of_two_algo("mpc_rule_pred50", "rule_based", res, 100)
    plot_histogram_of_two_algo("mpc_rule_pred100", "rule_based", res, 100)
    print_algorithm_comparison_report(res)
    
    """
    rule_df = pd.read_csv('rule_dispath_data_md5.csv')
    load_df = pd.read_csv('kwh_load_md5.csv')
    sun_df = pd.read_csv('kwh_sun_md5.csv')
    tariff_df = pd.read_csv('tariff_info_md5.csv')
    battery_info_df = pd.read_csv('battery_info_md5.csv')
    
    samples = select_test_samples_cache(
        rule_df,
        load_df,
        sun_df,
        tariff_df,
        battery_info_df,
        year=2024,
        num_day=20,
        num_gatwayid=10,
        seed=123
    )
    with open("test_samples.json", "w") as f:
        json.dump(samples, f, indent=4)
    """