import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,numpy as np
from magnetic_control.runtime import load_fields
from magnetic_control.tracking import Target,track
from magnetic_control.export import save_result

def main():
    ap=argparse.ArgumentParser(description='Track waypoints with the trained magnetic inverse policy.')
    ap.add_argument('path');ap.add_argument('--mode',choices=['spatial','time'],default='spatial')
    ap.add_argument('--output',default='results/custom');a=ap.parse_args()
    c=load_fields();target=Target(np.loadtxt(a.path,delimiter=',',skiprows=1),c,a.mode)
    result=track(target,c);save_result(result,c,a.output)
    print(json.dumps(result['report'],indent=2))
    if not result['report']['accepted']:raise SystemExit('Target rejected: tracking tolerance, domain, duration or magnetic limits exceeded.')
if __name__=='__main__':main()
