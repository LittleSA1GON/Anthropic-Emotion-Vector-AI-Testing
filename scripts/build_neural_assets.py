"""Blender: --background --python scripts/build_neural_assets.py -- --out assets/neural.

Original procedural scientific illustration. Geometry never represents actual
weights, neurons or measured tensors. The runtime overlays separate measurements.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Vector


def main():
    args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='assets/neural');parser.add_argument('--skip-render',action='store_true')
    options=parser.parse_args(args);out=Path(options.out).resolve();out.mkdir(parents=True,exist_ok=True)
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene;scene.frame_start=1;scene.frame_end=121;scene.render.fps=24
    scene.world.use_nodes=True
    background=scene.world.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value=(.008,.012,.025,1);background.inputs['Strength'].default_value=.3
    scene['purpose']='Illustrative neural network; not measured parameters or activations'
    scene['states']='Dormant: frames 1-60. Active / aggregation / steering motifs: frames 61-120.'
    def material(name,color,glow=0):
        mat=bpy.data.materials.new(name);mat.diffuse_color=(*color,1);mat.use_nodes=True
        bsdf=mat.node_tree.nodes.get('Principled BSDF');bsdf.inputs['Base Color'].default_value=(*color,1)
        bsdf.inputs['Metallic'].default_value=.35;bsdf.inputs['Roughness'].default_value=.36
        bsdf.inputs['Emission Color'].default_value=(*color,1);bsdf.inputs['Emission Strength'].default_value=glow
        return mat
    cyan=material('Neutral / cyan',(.12,.54,.72),.25);violet=material('Dormant / violet',(.24,.2,.43),.12)
    amber=material('Emotion / amber',(.78,.42,.12),.3);mint=material('Direction / mint',(.16,.65,.49),.4)
    wire=material('Structural connections',(.035,.12,.2),.05)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1,radius=.085)
    prototype=bpy.context.object;mesh=prototype.data;mesh.name='Shared low-poly node';bpy.data.objects.remove(prototype,do_unlink=True)
    mesh.materials.append(cyan)
    points=[];links=[];objects=[]
    for layer in range(6):
        for slot in range(8):
            angle=2*math.pi*slot/8
            coord=(math.cos(angle)*1.05,math.sin(angle)*1.05,(layer-2.5)*.78)
            obj=bpy.data.objects.new(f'Layer_{layer:02d}_sample_{slot:02d}',mesh);scene.collection.objects.link(obj);obj.location=coord
            obj['illustrative_layer']=layer;obj['sample_slot']=slot
            obj.material_slots[0].link='OBJECT';obj.material_slots[0].material=cyan if layer%2==0 else violet
            points.append({'layer':layer,'slot':slot,'position':list(coord)});objects.append(obj)
            if layer<5:
                links.extend([[layer*8+slot,(layer+1)*8+slot],[layer*8+slot,(layer+1)*8+(slot+1)%8]])
            for frame in [1,31,61,91,121]:
                phase=2*math.pi*(frame-1)/120
                obj.location=(coord[0]+.035*math.sin(phase+slot),coord[1]+.035*math.cos(phase+slot),coord[2])
                obj.keyframe_insert(data_path='location',frame=frame)
                strength=.8+.32*(.5-.5*math.cos(phase))
                obj.scale=(strength,)*3;obj.keyframe_insert(data_path='scale',frame=frame)
    curve=bpy.data.curves.new('Sparse structural links','CURVE');curve.dimensions='3D';curve.bevel_depth=.008;curve.bevel_resolution=0
    for a,b in links:
        spline=curve.splines.new('POLY');spline.points.add(1)
        for point,position in zip(spline.points,[points[a]['position'],points[b]['position']]):point.co=(*position,1)
    obj=bpy.data.objects.new('Sampled structural connections',curve);scene.collection.objects.link(obj);curve.materials.append(wire)
    # Two aggregated streams and a separate direction motif, never model weights.
    for name,location,mat in [('Neutral mean',(-2.25,0,0),cyan),('Emotion mean',(2.25,0,0),amber),('Unit contrast',(0,0,3),mint)]:
        bpy.ops.mesh.primitive_cube_add(size=.28,location=location);obj=bpy.context.object;obj.name=name;obj.data.materials.append(mat)
        obj['purpose']='Illustrative aggregation / vector motif'
    for i in range(6):
        obj=bpy.data.objects.new(f'Stream particle {i}',mesh);scene.collection.objects.link(obj)
        obj.material_slots[0].link='OBJECT';obj.material_slots[0].material=amber if i%2 else cyan
        for frame in [1,31,61,91,121]:
            phase=((frame-1)/120+i/6)%1
            obj.location=((2.25 if i%2 else -2.25)*(1-phase),0,phase*3)
            obj.scale=(.65,)*3;obj.keyframe_insert(data_path='location',frame=frame)
    bpy.ops.object.camera_add(location=(7,-9,6));camera=bpy.context.object;camera.name='Slow orbit camera';camera.data.lens=32;scene.camera=camera
    for frame,angle in [(1,-.05),(61,.05),(121,-.05)]:
        camera.location=(8*math.cos(angle+.8),-8*math.sin(angle+.8),5.5)
        camera.rotation_euler=(Vector((0,0,.2))-camera.location).to_track_quat('-Z','Y').to_euler()
        camera.keyframe_insert(data_path='location',frame=frame);camera.keyframe_insert(data_path='rotation_euler',frame=frame)
    for name,location,power in [('Key',(3,-4,6),850),('Fill',(-4,1,3),600)]:
        bpy.ops.object.light_add(type='AREA',location=location);lamp=bpy.context.object;lamp.name=name;lamp.data.energy=power;lamp.data.shape='DISK';lamp.data.size=5
        lamp.rotation_euler=(-lamp.location).to_track_quat('-Z','Y').to_euler()
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'neural_lab.blend'),compress=True)
    bpy.ops.export_scene.gltf(filepath=str(out/'neural_lab.glb'),export_format='GLB',export_animations=True,export_lights=False,export_cameras=False)
    manifest={'schema':'mm-emotion.blender-neural-asset.v1','generator':'scripts/build_neural_assets.py',
        'blender_version':bpy.app.version_string,'license':'CC0-1.0; original procedural geometry',
        'purpose':'Illustrative geometry only; not measured parameters, neurons, layers, or activations',
        'illustrative_layers':6,'nodes_per_layer':8,'nodes':points,'links':links,
        'frames':121,'loop_period_frames':120,'fps':24,'states':{'dormant':[1,60],'active':[61,120]},
        'runtime':'CPU Canvas perspective projection of this Blender-authored node template; actual architecture overrides the displayed layer count',
        'files':['neural_lab.blend','neural_lab.glb','neural-preview.png']}
    (out/'neural_scene.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    if not options.skip_render:
        scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=12
        scene.render.resolution_x=960;scene.render.resolution_y=540;scene.render.resolution_percentage=100
        scene.render.image_settings.file_format='PNG';scene.render.filepath=str(out/'neural-preview.png')
        scene.render.film_transparent=False;bpy.ops.render.render(write_still=True)
    assert (out/'neural_lab.blend').stat().st_size>1000 and (out/'neural_lab.glb').read_bytes()[:4]==b'glTF'
    print('NEURAL_ASSETS_READY',out)


if __name__=='__main__':main()
