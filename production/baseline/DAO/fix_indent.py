import sys

def process(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # We need to unindent lines 1175 to 1329 by 4 spaces
    # We delete line 1330
    # We unindent lines 1454 to 1498 by 4 spaces
    
    out_lines = []
    for i, line in enumerate(lines):
        line_num = i + 1
        if 1175 <= line_num <= 1329:
            if line.startswith('    '):
                out_lines.append(line[4:])
            else:
                out_lines.append(line)
        elif line_num == 1330:
            # delete conn.close()
            continue
        elif 1454 <= line_num <= 1498:
            if line.startswith('    '):
                out_lines.append(line[4:])
            else:
                out_lines.append(line)
        else:
            out_lines.append(line)
            
    with open(filepath, 'w') as f:
        f.writelines(out_lines)

process('dashboard_monitor_ultimate.py')
