"""Build a fixed, manually matchable CAD target; no hardware access."""
import hashlib
import json
from pathlib import Path

import numpy as np

from so101_kinematics import SO101Kinematics, ARM_JOINTS

ROOT = Path(__file__).resolve().parents[1]


def main():
    # An explicit CAD target, not an estimate of an already observed arm pose.
    # Starts from the level-member display and raises the upper arm 90 degrees,
    # compensating at the elbow so the forearm stays approximately level.
    degrees = [0., 93.1, -89.5, 1.5, 90.]
    radians = np.deg2rad(degrees)
    model = SO101Kinematics()
    target = {
        'version': 1,
        'kind': 'CAD_target_not_hardware_measurement',
        'name': 'upright-upper-arm-level-forearm',
        'model_sha256': hashlib.sha256((ROOT/'SO-101.zip').read_bytes()).hexdigest(),
        'joint_order': list(ARM_JOINTS),
        'q_cad_degrees': degrees,
        'q_cad_rad': radians.tolist(),
        'wrist_frame_pose': model.fk(radians).tolist(),
        'matching_method': 'manual visual match; uncertainty not quantified',
        'notes': [
            'Match the base/clamp orientation as well as the arm links.',
            'Upper arm is approximately upright and forearm approximately level.',
            'Colored moving jaw identifies wrist-roll orientation; avoid a 180-degree reversal.',
            'Gripper opening is not one of the five calibrated arm coordinates.',
        ],
    }
    path = ROOT/'calibration/reference_poses/upright-cad-target.json'
    path.write_text(json.dumps(target, indent=2)+'\n', encoding='utf-8')
    html = (ROOT/'visualizations/so101-kinematics.html').read_text(encoding='utf-8')
    js = (ROOT/'tools/so101_reference_view.js').read_text(encoding='utf-8')
    js = js.replace('/*REFERENCE_POSE_DEG*/', json.dumps([*degrees, 0.]))
    js += '''
root.querySelector('.so-toolbar span').textContent='Manual zero-offset reference · colored moving jaw';
root.querySelector('#so-reset').textContent='Target pose';
root.querySelectorAll('#so-sliders input').forEach(e=>e.disabled=true);
root.querySelector('details').hidden=true;
const jawIndex=data.findIndex(d=>d.name==='Body5_2');
nodes[jawIndex].children.forEach(m=>{if(m.isMesh)m.material.color.copy(accent)});
root.querySelector('#so-position').textContent='CAD target: base 0° · shoulder 93.1° · elbow −89.5° · wrist pitch 1.5° · roll 90°';
fitReferenceView([-.75,.65,.4]);
'''
    marker = '}catch(e){root.querySelector'
    assert html.count(marker) == 1
    html = html.replace(marker, js+'\n'+marker)
    output = ROOT/'visualizations/so101-upright-reference.html'
    output.write_text(html, encoding='utf-8')
    assert output.stat().st_size < 1_000_000
    print('CAD target:', path)
    print('Reference guide:', output)
    print('Target radians:', radians)


if __name__ == '__main__':
    main()
