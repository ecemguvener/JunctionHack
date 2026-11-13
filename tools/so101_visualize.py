"""Build a mesh viewer and check resolved AGL FK against the bundled URDF.

Run on Intel: uv run --with fast-simplification python tools/so101_visualize.py
No connection to robot hardware is made. Mesh decimation affects display only.
"""
import json
import re
import struct
import subprocess
import tempfile
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import fast_simplification
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def frame(position, rotation=None):
    m = np.eye(4)
    m[:3, 3] = position
    if rotation is not None:
        m[:3, :3] = rotation
    return m


def rot(axis, angle):
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    r = np.eye(3) + np.sin(angle) * k + (1-np.cos(angle)) * (k @ k)
    return frame([0, 0, 0], r)


def urdf_origin(node):
    if node is None:
        return np.eye(4)
    xyz = np.fromstring(node.get('xyz', '0 0 0'), sep=' ')
    r, p, y = np.fromstring(node.get('rpy', '0 0 0'), sep=' ')
    return frame(xyz) @ rot([0, 0, 1], y) @ rot([0, 1, 0], p) @ rot([1, 0, 0], r)


def main():
    archive = zipfile.ZipFile(ROOT / 'SO-101.zip')
    prefix = 'robotics_ws_model/'
    model = json.loads(archive.read(prefix+'kinematics.json'))
    urdf = ET.fromstring(archive.read(prefix+'robot.urdf'))
    agl = (ROOT/'SO-101.agl').read_text()
    token = r'(?:"[^"]+"|[\w]+)'
    declared = re.findall(rf'joint\s+({token})\s*:\s*revolute\s+({token})\s*->\s*({token})', agl)
    triples = {(a.strip('"'), b.strip('"'), c.strip('"')) for a,b,c in declared}
    assert triples == {(j['name'],j['parent'],j['child']) for j in model['joints']}
    part_lists=re.findall(rf'link\s+({token})\s*=\s*parts\s*\[([^\]]+)\]',agl)
    declared_parts={n.strip('"'):set(re.findall(r'"([^"]+)"',parts)) for n,parts in part_lists}
    resolved_parts={l['name']:{Path(m['file']).stem for m in l['meshes']} for l in model['links']}
    assert declared_parts==resolved_parts
    assert not model['couplings'] and not model['cuts']
    links = {l['name']:l for l in model['links']}
    homes = {n:frame(l['origin'], l.get('rotation')) for n,l in links.items()}
    names = [model['base']] + [j['child'] for j in model['joints']]
    assert len(set(names)) == len(links)
    incoming = {j['child']:j for j in model['joints']}
    uj = {j.get('name'):j for j in urdf.findall('joint')}
    urdf_links={l.get('name'):l for l in urdf.findall('link')}
    rows = []
    for name in names:
        if name == model['base']:
            rows.append(dict(name=name,parent=-1,joint=None,h=np.eye(4),axis=np.array([0,0,1.])))
            continue
        j=incoming[name]
        parent=names.index(j['parent'])
        assert parent < names.index(name)
        h=np.linalg.inv(homes[j['parent']])@homes[name]
        axis=homes[name][:3,:3].T@np.array(j['coordinates'][0]['axis'])
        axis/=np.linalg.norm(axis)
        rows.append(dict(name=name,parent=parent,joint=j['name'],h=h,axis=axis))

    # Independent URDF local frames vs resolved AGL global frames, using C++ FK.
    max_pos=max_rot=0.
    rng=np.random.default_rng(101)
    with tempfile.TemporaryDirectory(prefix='so101-fk-') as tmp:
        executable=Path(tmp)/'fk'
        subprocess.run(['c++','-O2','-std=c++17',str(ROOT/'tools/so101_fk.cpp'),'-o',str(executable)],check=True)
        for case in range(21):
            q=np.zeros(len(rows)) if case==0 else rng.uniform(-1.2,1.2,len(rows))
            q[0]=0  # Fixed base is not a coordinate.
            lines=[str(len(rows))]
            expected=[]
            for i,row in enumerate(rows):
                values=[row['parent'],*row['h'].ravel(),*row['axis'],q[i]]
                lines.append(' '.join(map(str,values)))
                if row['parent']==-1:
                    expected.append(np.eye(4))
                else:
                    joint=uj[row['joint']]
                    axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
                    expected.append(expected[row['parent']]@urdf_origin(joint.find('origin'))@rot(axis,q[i]))
            output=subprocess.check_output([str(executable)],input='\n'.join(lines)+'\n',text=True)
            actual=np.fromstring(output,sep=' ').reshape(-1,4,4)
            expected=np.array(expected)
            max_pos=max(max_pos,float(np.linalg.norm(actual[:,:3,3]-expected[:,:3,3],axis=1).max()))
            max_rot=max(max_rot,float(np.abs(actual[:,:3,:3]-expected[:,:3,:3]).max()))
    assert max_pos<1e-7 and max_rot<1e-7,(max_pos,max_rot)
    print(f'FK PASS: 21 poses, {len(rows)} frames; max position {max_pos:.3g} m, rotation entry {max_rot:.3g}')

    data=[]
    triangles=0
    for row in rows:
        name=row['name']
        local_inv=np.linalg.inv(homes[name])
        visuals=urdf_links[name].findall('visual')
        assert len(visuals)==len(links[name]['meshes'])
        for visual in visuals:
            assert np.max(np.abs(urdf_origin(visual.find('origin'))-local_inv))<1e-7
            assert np.allclose(np.fromstring(visual.find('geometry/mesh').get('scale'),sep=' '),.001)
        meshes=[]
        for mesh in links[name]['meshes']:
            raw=archive.read(prefix+mesh['file'])
            count=struct.unpack_from('<I',raw,80)[0]
            dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')])
            vertices=np.frombuffer(raw,dtype=dtype,count=count,offset=84)['v'].reshape(-1,3)
            vertices,idx=np.unique(vertices,axis=0,return_inverse=True)
            faces=idx.reshape(-1,3)
            if len(faces)>350:
                vertices,faces=fast_simplification.simplify(vertices.astype(float),faces.astype(np.int32),1-350/len(faces))
            vertices=vertices*mesh['scale']
            vertices=vertices@local_inv[:3,:3].T+local_inv[:3,3]
            # Quantize display geometry to 0.01 mm; retain full precision FK.
            meshes.append(dict(v=np.rint(vertices*1e5).astype(int).ravel().tolist(),f=faces.ravel().tolist()))
            triangles+=len(faces)
        data.append(dict(name=name,parent=row['parent'],joint=row['joint'],h=row['h'].ravel().tolist(),axis=row['axis'].tolist(),meshes=meshes))
    template=(ROOT/'tools/so101_viewer.html').read_text()
    html=template.replace('/*MODEL_DATA*/',json.dumps(data,separators=(',',':')))
    html='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head><body>'+html+'</body></html>'
    out=ROOT/'visualizations/so101-kinematics.html'
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    assert out.stat().st_size<1_000_000
    report=dict(poses=21,frames=len(rows),joints=len(rows)-1,max_position_error_m=max_pos,max_rotation_entry_error=max_rot,display_triangles=triangles)
    (out.parent/'so101-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'{out}: {out.stat().st_size} bytes, {triangles} display triangles')


if __name__=='__main__':
    main()
