"""
Generate an HTML animation for a case CSV file.
Parses CSV directly in Python and injects data into the case90.html template.

Usage:
  python generate_video.py case13.csv
  python generate_video.py case1.csv case46.csv case61.csv case74.csv case96.csv
"""

import sys
import os
import re
import csv

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.join(SCRIPT_DIR, '..')
CASES_DIR     = os.path.join(ROOT_DIR, 'cases')
TEMPLATE_FILE = os.path.join(ROOT_DIR, 'videos/case90.html')


def parse_csv(csv_path):
    """Extract DATA[agent][timestep]=[x,y,vx,vy] and GOALS[[gx,gy],...] from CSV."""
    data = [[] for _ in range(4)]   # 4 agents
    goals = None

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if goals is None:
                goals = []
                for i in range(4):
                    x  = float(row[f'obs_{i*6}'])
                    y  = float(row[f'obs_{i*6+1}'])
                    dx = float(row[f'obs_{i*6+4}'])
                    dy = float(row[f'obs_{i*6+5}'])
                    goals.append([round(x - dx, 5), round(y - dy, 5)])

            for i in range(4):
                x  = float(row[f'obs_{i*6}'])
                y  = float(row[f'obs_{i*6+1}'])
                vx = float(row[f'obs_{i*6+2}'])
                vy = float(row[f'obs_{i*6+3}'])
                data[i].append([round(x,7), round(y,7), round(vx,7), round(vy,7)])

    return data, goals


def to_js_array(data, goals):
    """Serialize DATA and GOALS to compact JS array strings."""
    def fmt(v):
        return str(v) if v != 0 else '0'

    agents = []
    for agent_steps in data:
        steps = '[' + ','.join(
            '[' + ','.join(fmt(v) for v in step) + ']'
            for step in agent_steps
        ) + ']'
        agents.append(steps)
    data_str = '[' + ','.join(agents) + ']'

    goals_str = '[' + ','.join('[' + ','.join(fmt(v) for v in g) + ']' for g in goals) + ']'

    return data_str, goals_str


def generate_html(csv_path):
    case_name   = os.path.splitext(os.path.basename(csv_path))[0]  # e.g. case61
    case_number = case_name.replace('case', '')                      # e.g. 61

    print(f'Processing {case_name}...')
    data, goals = parse_csv(csv_path)
    n_steps = len(data[0])

    data_str, goals_str = to_js_array(data, goals)

    with open(TEMPLATE_FILE, 'r') as f:
        html = f.read()

    # Replace DATA array
    html = re.sub(r'const DATA=\[.*?\];', f'const DATA={data_str};', html, flags=re.DOTALL)

    # Replace GOALS array
    html = re.sub(r'const GOALS=\[.*?\];', f'const GOALS={goals_str};', html, flags=re.DOTALL)

    # Replace slider max
    html = re.sub(r'(max=")(\d+)(")', lambda m: f'{m.group(1)}{n_steps - 1}{m.group(3)}', html)

    # Replace MAX_T constant
    html = re.sub(r'const MAX_T = \d+;', f'const MAX_T = {n_steps - 1};', html)

    # Replace timestep display in info text  e.g. "/ 71"
    html = re.sub(r'(/ )\d+(<)', f'\\g<1>{n_steps - 1}\\2', html)

    # Replace case number in title
    html = re.sub(r'Case \d+', f'Case {case_number}', html)

    out_path = os.path.join(ROOT_DIR, f'{case_name}.html')
    with open(out_path, 'w') as f:
        f.write(html)

    print(f'  Saved: {out_path}  ({n_steps} timesteps, goals={goals})')
    return out_path


def main():
    if len(sys.argv) < 2:
        print('Usage: python generate_video.py <csv_file> [csv_file2 ...]')
        print('Example: python generate_video.py case61.csv case74.csv case96.csv')
        sys.exit(1)

    for csv_arg in sys.argv[1:]:
        if os.path.isabs(csv_arg) or os.path.exists(csv_arg):
            csv_path = csv_arg
        else:
            csv_path = os.path.join(CASES_DIR, csv_arg)

        if not os.path.exists(csv_path):
            print(f'Error: file not found: {csv_path}')
            continue

        generate_html(csv_path)


if __name__ == '__main__':
    main()
