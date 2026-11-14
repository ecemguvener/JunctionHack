// Injected into the viewer's existing scope after its interaction smoke check.
const referenceAngles=/*REFERENCE_POSE_DEG*/;
const referenceNames=['Revolute_1','Revolute_2','Revolute_3','Revolute_2_5','Revolute_4','Revolute_5'];
root.querySelector('.so-toolbar span').textContent='Follower reference fit · provisional CAD angles';
root.querySelector('#so-reset').textContent='Reference pose';
root.querySelectorAll('#so-sliders input').forEach(e=>{e.min=-270;e.max=270;e.step=.1});
root.querySelector('#so-extra').min=-270;root.querySelector('#so-extra').max=270;
function fitReferenceView(direction=[-1,.12,.22]){
 scene.updateMatrixWorld(true);
 const box=new THREE.Box3();meshes.forEach(m=>box.union(new THREE.Box3().setFromObject(m)));
 box.getCenter(center);control.target.copy(center);
 const size=box.getSize(new THREE.Vector3());
 const distance=Math.max(size.z,size.length()*.5,size.length()/Math.max(camera.aspect,1))*.72/Math.tan(camera.fov*Math.PI/360);
 camera.position.copy(center).add(new THREE.Vector3(...direction).normalize().multiplyScalar(distance));
 grid.position.z=box.min.z-.002;control.update();render();
}
function referencePose(){
 q.fill(0);referenceNames.forEach((name,i)=>q[byJoint.get(name)]=referenceAngles[i]);
 syncExtra();update();fitReferenceView();
}
root.querySelector('#so-reset').onclick=referencePose;
root.querySelector('#so-front').onclick=()=>fitReferenceView([-1,.02,.12]);
root.querySelector('#so-perspective').onclick=()=>fitReferenceView([-.75,.65,.4]);
new ResizeObserver(()=>fitReferenceView()).observe(stage);
referencePose();
console.info(`Reference fit loaded: wrist roll ${referenceAngles[4]} CAD degrees; display orientation includes the user correction. Other reference angles are visual hypotheses, not confirmed actuator alignment.`);
