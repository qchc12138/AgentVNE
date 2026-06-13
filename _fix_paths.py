import sys
p = sys.argv[1]
c = open(p, encoding='utf-8').read()
c = c.replace("'workflow_topo', 'workflow1_topo.json'", "'Workflow_topo', 'workflow1_topo.json'")
c = c.replace("'workflow_topo', 'workflow1_noderank.json'", "'Workflow_topo', 'workflow1_noderank.json'")
open(p, 'w', encoding='utf-8').write(c)
print('Fixed', p)