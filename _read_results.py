import json
with open(r'e:\E桌面\AgentVNE\eval_output\evaluation_results.json', 'r', encoding='utf-8') as f:
    d = json.load(f)
print(f'Total records: {len(d)}')
print()
for r in sorted(d, key=lambda x: (x['strategy'], x['arrival_rate'], x['mean_lifetime'])):
    s = r['strategy']
    a = r['arrival_rate']
    l = r['mean_lifetime']
    ac = r['acceptance_rate'] * 100
    cd = r.get('avg_comm_delay', 0)
    tm = r.get('avg_time_per_vn_ms', 0)
    av = r.get('accepted_vn', 0)
    tv = r.get('total_vn', 0)
    print(f'{s:10s}  ar={a:.1f}  lt={l:.0f}  accept={ac:.1f}%  delay={cd:.1f}  time={tm:.1f}ms  ({av}/{tv})')
