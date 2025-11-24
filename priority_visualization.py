import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from power_to_priority_converter import analyze_optimization_result

def visualize_priority_analysis(json_file_path: str):
    """
    可视化优先级分析结果
    """
    # 执行分析
    result = analyze_optimization_result(json_file_path)
    
    detailed_analysis = result['detailed_analysis']
    rule_format = result['rule_based_format']
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('功率流向转优先级分析可视化', fontsize=16, fontweight='bold')
    
    # 1. 24小时光伏优先级变化
    ax1 = axes[0, 0]
    solar_priorities = np.array(rule_format['solar_priority'])
    hours = range(24)
    
    ax1.plot(hours, solar_priorities[:, 0], 'o-', label='电网优先级', color='gray', linewidth=2)
    ax1.plot(hours, solar_priorities[:, 1], 's-', label='负载优先级', color='blue', linewidth=2)
    ax1.plot(hours, solar_priorities[:, 2], '^-', label='电池优先级', color='green', linewidth=2)
    ax1.set_title('光伏发电优先级变化', fontweight='bold')
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('优先级')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 3.5)
    
    # 2. 24小时负载优先级变化
    ax2 = axes[0, 1]
    load_priorities = np.array(rule_format['load_priority'])
    
    ax2.plot(hours, load_priorities[:, 0], 'o-', label='电网优先级', color='gray', linewidth=2)
    ax2.plot(hours, load_priorities[:, 1], 's-', label='光伏优先级', color='orange', linewidth=2)
    ax2.plot(hours, load_priorities[:, 2], '^-', label='电池优先级', color='green', linewidth=2)
    ax2.set_title('负载需求优先级变化', fontweight='bold')
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('优先级')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 3.5)
    
    # 3. 主导策略分布饼图
    ax3 = axes[0, 2]
    strategy_counts = rule_format['strategy_analysis']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    ax3.pie(strategy_counts.values(), labels=strategy_counts.keys(), autopct='%1.1f%%', 
           colors=colors[:len(strategy_counts)], startangle=90)
    ax3.set_title('主导策略分布', fontweight='bold')
    
    # 4. 功率流向热图 - 光伏去向
    ax4 = axes[1, 0]
    pv_flows = []
    for hour_data in detailed_analysis['hourly_analysis']:
        flows = hour_data['power_flows']
        pv_flows.append([flows['pv_to_grid'], flows['pv_to_load'], flows['pv_to_bat']])
    
    pv_flows = np.array(pv_flows).T
    im1 = ax4.imshow(pv_flows, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax4.set_title('光伏功率流向热图', fontweight='bold')
    ax4.set_ylabel('去向')
    ax4.set_xlabel('时间 (小时)')
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['电网', '负载', '电池'])
    plt.colorbar(im1, ax=ax4, label='功率 (W)')
    
    # 5. 功率流向热图 - 负载来源
    ax5 = axes[1, 1]
    load_flows = []
    for hour_data in detailed_analysis['hourly_analysis']:
        flows = hour_data['power_flows']
        load_flows.append([flows['grid_to_load'], flows['pv_to_load'], flows['bat_to_load']])
    
    load_flows = np.array(load_flows).T
    im2 = ax5.imshow(load_flows, aspect='auto', cmap='Blues', interpolation='nearest')
    ax5.set_title('负载供电来源热图', fontweight='bold')
    ax5.set_ylabel('来源')
    ax5.set_xlabel('时间 (小时)')
    ax5.set_yticks([0, 1, 2])
    ax5.set_yticklabels(['电网', '光伏', '电池'])
    plt.colorbar(im2, ax=ax5, label='功率 (W)')
    
    # 6. 优先级稳定性分析
    ax6 = axes[1, 2]
    
    # 计算优先级变化次数
    solar_changes = 0
    load_changes = 0
    
    for i in range(1, 24):
        if not np.array_equal(solar_priorities[i], solar_priorities[i-1]):
            solar_changes += 1
        if not np.array_equal(load_priorities[i], load_priorities[i-1]):
            load_changes += 1
    
    # 计算优先级模式多样性
    unique_solar = len(set([str(p) for p in solar_priorities]))
    unique_load = len(set([str(p) for p in load_priorities]))
    
    stability_metrics = ['光伏优先级\n变化次数', '负载优先级\n变化次数', 
                        '光伏优先级\n模式数', '负载优先级\n模式数']
    values = [solar_changes, load_changes, unique_solar, unique_load]
    colors_bar = ['orange', 'blue', 'lightcoral', 'lightblue']
    
    bars = ax6.bar(stability_metrics, values, color=colors_bar, alpha=0.7)
    ax6.set_title('优先级稳定性分析', fontweight='bold')
    ax6.set_ylabel('次数/模式数')
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(value), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    output_path = json_file_path.replace('.json', '_priority_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return result

def generate_priority_report(json_file_path: str):
    """
    生成优先级分析报告
    """
    result = analyze_optimization_result(json_file_path)
    detailed_analysis = result['detailed_analysis']
    rule_format = result['rule_based_format']
    
    print("="*80)
    print("📋 功率流向转优先级详细报告")
    print("="*80)
    
    # 1. 总体优先级推荐
    print("\n🎯 总体优先级推荐:")
    print(f"光伏发电优先级 [电网, 负载, 电池]: {rule_format['recommended_global_solar_priority']}")
    print(f"负载需求优先级 [电网, 光伏, 电池]: {rule_format['recommended_global_load_priority']}")
    
    # 2. 优先级解释
    print(f"\n📖 优先级含义:")
    print(f"  • 数字越大，优先级越高")
    print(f"  • 光伏发电优先级：光伏电力的分配顺序")
    print(f"  • 负载需求优先级：负载用电的来源顺序")
    
    # 3. 策略分析
    print(f"\n📊 运行策略统计:")
    total_hours = sum(rule_format['strategy_analysis'].values())
    for strategy, count in rule_format['strategy_analysis'].items():
        percentage = count / total_hours * 100
        print(f"  • {strategy}: {count}小时 ({percentage:.1f}%)")
    
    # 4. 时段分析
    print(f"\n⏰ 不同时段的优先级特征:")
    
    # 按时段分析优先级模式
    night_hours = list(range(0, 7)) + list(range(20, 24))  # 夜间
    day_hours = list(range(7, 20))  # 白天
    
    night_strategies = []
    day_strategies = []
    
    for hour_data in detailed_analysis['hourly_analysis']:
        hour = hour_data['hour']
        strategy = hour_data['main_strategy']
        
        if hour in night_hours:
            night_strategies.append(strategy)
        else:
            day_strategies.append(strategy)
    
    print(f"  夜间时段 (0-6, 20-23点):")
    night_counter = {}
    for s in night_strategies:
        night_counter[s] = night_counter.get(s, 0) + 1
    for strategy, count in night_counter.items():
        print(f"    - {strategy}: {count}小时")
    
    print(f"  白天时段 (7-19点):")
    day_counter = {}
    for s in day_strategies:
        day_counter[s] = day_counter.get(s, 0) + 1
    for strategy, count in day_counter.items():
        print(f"    - {strategy}: {count}小时")
    
    # 5. 关键洞察
    print(f"\n💡 关键洞察:")
    
    # 分析是否有明显的优先级模式
    solar_priorities = rule_format['solar_priority']
    load_priorities = rule_format['load_priority']
    
    # 统计最常见的优先级组合
    from collections import Counter
    solar_patterns = Counter([str(p) for p in solar_priorities])
    load_patterns = Counter([str(p) for p in load_priorities])
    
    most_common_solar = solar_patterns.most_common(1)[0]
    most_common_load = load_patterns.most_common(1)[0]
    
    print(f"  • 最常见光伏优先级模式: {most_common_solar[0]} (出现{most_common_solar[1]}次)")
    print(f"  • 最常见负载优先级模式: {most_common_load[0]} (出现{most_common_load[1]}次)")
    
    # 分析电池使用模式
    battery_active_hours = sum(1 for hour_data in detailed_analysis['hourly_analysis'] 
                              if '电池' in hour_data['main_strategy'])
    print(f"  • 电池活跃时间: {battery_active_hours}小时 ({battery_active_hours/24*100:.1f}%)")
    
    # 分析光伏利用模式
    pv_active_hours = sum(1 for hour_data in detailed_analysis['hourly_analysis'] 
                         if hour_data['pv_generation'] > 100)
    pv_sell_hours = sum(1 for hour_data in detailed_analysis['hourly_analysis'] 
                       if '光伏售电' in hour_data['main_strategy'])
    if pv_active_hours > 0:
        print(f"  • 光伏发电时段: {pv_active_hours}小时，其中售电{pv_sell_hours}小时")
    
    # 6. 使用建议
    print(f"\n🚀 使用建议:")
    print(f"  1. 对于静态配置，使用推荐的全局优先级")
    print(f"  2. 对于动态配置，使用24小时逐时优先级列表")
    print(f"  3. 重点关注{list(rule_format['strategy_analysis'].keys())[0]}策略的参数调优")
    print(f"  4. 可以根据季节/天气模式调整优先级权重")
    
    return result

if __name__ == "__main__":
    json_path = "vis/gateway_id:ebbb9e1a343b092537857cc17021670e-date:2024-12-29-hier-mpc-pcs/res.json"
    
    # 生成可视化
    print("正在生成优先级分析可视化...")
    result = visualize_priority_analysis(json_path)
    
    # 生成详细报告
    print("\n" + "="*80)
    generate_priority_report(json_path)