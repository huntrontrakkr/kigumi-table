"""Blender (Cycles) renderer.  Run inside Blender:

    blender -b --factory-startup --python kigumi/render_blender.py -- \
        --npz design.npz --out render.png --view hero --samples 96

Only numpy and bpy are used here (Blender's own Python).
"""
import argparse
import json
import math
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

SPECIES = {
    # light, dark (late wood), roughness
    "hinoki": ((0.88, 0.76, 0.57), (0.74, 0.58, 0.38), 0.48),
    "white_oak": ((0.74, 0.57, 0.37), (0.52, 0.37, 0.21), 0.5),
    "walnut": ((0.36, 0.23, 0.15), (0.17, 0.10, 0.065), 0.45),
    "cherry": ((0.56, 0.33, 0.22), (0.38, 0.19, 0.11), 0.45),
    "ash": ((0.87, 0.78, 0.62), (0.62, 0.50, 0.35), 0.5),
    "maple": ((0.92, 0.83, 0.67), (0.80, 0.68, 0.50), 0.45),
    "sugi": ((0.76, 0.50, 0.36), (0.52, 0.30, 0.20), 0.5),
    "yakisugi": ((0.045, 0.04, 0.036), (0.012, 0.011, 0.010), 0.82),
    "wenge": ((0.24, 0.17, 0.12), (0.07, 0.05, 0.04), 0.5),
}


def lin(c):
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c)


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", default="hero")
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--res", default="1600x1100")
    ap.add_argument("--device", default="CUDA")
    ap.add_argument("--floor", default="studio")
    ap.add_argument("--azimuth", type=float, default=None)
    ap.add_argument("--elevation", type=float, default=None)
    ap.add_argument("--lens", type=float, default=None)
    ap.add_argument("--hide", default="")
    return ap.parse_args(argv)


def setup_render(a):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = a.device
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == a.device
        sc.cycles.device = "GPU"
    except Exception as e:  # noqa
        print("GPU setup failed, using CPU:", e)
    sc.cycles.samples = a.samples
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 10
    sc.cycles.transmission_bounces = 10
    sc.cycles.glossy_bounces = 6
    w, h = (int(x) for x in a.res.split("x"))
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    sc.render.image_settings.file_format = "PNG"
    return sc


def node(nt, kind, loc=(0, 0), **props):
    n = nt.nodes.new(kind)
    n.location = loc
    for k, v in props.items():
        setattr(n, k, v)
    return n


def wood_material(name, species, seed):
    light, dark, rough = SPECIES.get(species, SPECIES["hinoki"])
    light, dark = lin(light), lin(dark)
    mat = bpy.data.materials.new(f"{name}")
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = node(nt, "ShaderNodeOutputMaterial", (900, 0))
    bsdf = node(nt, "ShaderNodeBsdfPrincipled", (600, 0))
    tc = node(nt, "ShaderNodeTexCoord", (-1200, 0))
    oi = node(nt, "ShaderNodeObjectInfo", (-1200, -300))
    # per-object random offset so every board shows its own grain
    rnd = node(nt, "ShaderNodeMath", (-1000, -300), operation="MULTIPLY")
    rnd.inputs[1].default_value = 37.0
    comb = node(nt, "ShaderNodeCombineXYZ", (-850, -300))
    nt.links.new(oi.outputs["Random"], rnd.inputs[0])
    nt.links.new(rnd.outputs[0], comb.inputs[0])
    # pith offset: boards are cut away from the log centre (flat-sawn look)
    off = node(nt, "ShaderNodeVectorMath", (-850, -100), operation="ADD")
    off.inputs[1].default_value = (0.0, 0.23, 0.41)
    add = node(nt, "ShaderNodeVectorMath", (-700, 0), operation="ADD")
    nt.links.new(tc.outputs["Object"], off.inputs[0])
    nt.links.new(off.outputs[0], add.inputs[0])
    nt.links.new(comb.outputs[0], add.inputs[1])
    mp = node(nt, "ShaderNodeMapping", (-550, 0))
    mp.inputs["Scale"].default_value = (0.05, 1.0, 1.0)
    nt.links.new(add.outputs[0], mp.inputs["Vector"])
    wave = node(nt, "ShaderNodeTexWave", (-350, 100), wave_type="RINGS", rings_direction="X")
    wave.inputs["Scale"].default_value = 70.0
    wave.inputs["Distortion"].default_value = 9.0
    wave.inputs["Detail"].default_value = 3.0
    wave.inputs["Detail Scale"].default_value = 1.2
    nt.links.new(mp.outputs[0], wave.inputs["Vector"])
    ramp = node(nt, "ShaderNodeValToRGB", (-120, 100))
    # late wood is a thin darker band; most of each ring is early wood
    ramp.color_ramp.elements[0].position = 0.55
    ramp.color_ramp.elements[0].color = (*light, 1)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (*[0.55 * a + 0.45 * b for a, b in zip(light, dark)], 1)
    nt.links.new(wave.outputs["Fac"], ramp.inputs["Fac"])
    # fine pores / figure
    mp2 = node(nt, "ShaderNodeMapping", (-550, -250))
    mp2.inputs["Scale"].default_value = (0.04, 1.0, 1.0)
    nt.links.new(add.outputs[0], mp2.inputs["Vector"])
    noise = node(nt, "ShaderNodeTexNoise", (-350, -250))
    noise.inputs["Scale"].default_value = 900.0
    noise.inputs["Detail"].default_value = 2.0
    nt.links.new(mp2.outputs[0], noise.inputs["Vector"])
    mix = node(nt, "ShaderNodeMix", (150, 50), data_type="RGBA", blend_type="MULTIPLY")
    mix.inputs["Factor"].default_value = 0.18
    nt.links.new(ramp.outputs["Color"], mix.inputs["A"])
    nt.links.new(noise.outputs["Color"], mix.inputs["B"])
    tone = node(nt, "ShaderNodeTexNoise", (-350, -650))
    tone.inputs["Scale"].default_value = 3.0
    mp3 = node(nt, "ShaderNodeMapping", (-550, -650))
    mp3.inputs["Scale"].default_value = (0.3, 1.0, 1.0)
    nt.links.new(add.outputs[0], mp3.inputs["Vector"])
    nt.links.new(mp3.outputs[0], tone.inputs["Vector"])
    hsv = node(nt, "ShaderNodeHueSaturation", (350, 50))
    vmath = node(nt, "ShaderNodeMapRange", (150, -650))
    vmath.inputs["To Min"].default_value = 0.85
    vmath.inputs["To Max"].default_value = 1.12
    nt.links.new(tone.outputs["Fac"], vmath.inputs["Value"])
    nt.links.new(vmath.outputs["Result"], hsv.inputs["Value"])
    nt.links.new(mix.outputs["Result"], hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rough
    try:
        bsdf.inputs["Coat Weight"].default_value = 0.10 if species != "yakisugi" else 0.0
        bsdf.inputs["Coat Roughness"].default_value = 0.35
    except KeyError:
        pass
    bump = node(nt, "ShaderNodeBump", (350, -250))
    bump.inputs["Strength"].default_value = 0.04
    bump.inputs["Distance"].default_value = 0.0002
    nt.links.new(wave.outputs["Fac"], bump.inputs["Height"])
    if species == "yakisugi":
        vor = node(nt, "ShaderNodeTexVoronoi", (-350, -500), feature="DISTANCE_TO_EDGE")
        vor.inputs["Scale"].default_value = 90.0
        nt.links.new(mp.outputs[0], vor.inputs["Vector"])
        bump.inputs["Strength"].default_value = 0.6
        cr = node(nt, "ShaderNodeMath", (100, -500), operation="POWER")
        cr.inputs[1].default_value = 0.35
        nt.links.new(vor.outputs["Distance"], cr.inputs[0])
        nt.links.new(cr.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def glass_material(kind):
    mat = bpy.data.materials.new(kind)
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    col = (0.86, 0.96, 0.92, 1) if kind == "glass" else (0.97, 0.98, 0.99, 1)
    b.inputs["Base Color"].default_value = col
    b.inputs["Roughness"].default_value = 0.0
    b.inputs["IOR"].default_value = 1.52 if kind == "glass" else 1.49
    b.inputs["Transmission Weight"].default_value = 1.0
    return mat


def frosted_material():
    mat = bpy.data.materials.new("silicone")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (0.9, 0.9, 0.88, 1)
    b.inputs["Roughness"].default_value = 0.35
    b.inputs["Transmission Weight"].default_value = 0.85
    b.inputs["IOR"].default_value = 1.41
    return mat


def plain_material(name, rgb, rough):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*lin(rgb), 1)
    b.inputs["Roughness"].default_value = rough
    return mat


def load_parts(path, hide):
    z = np.load(path)
    meta = json.loads(bytes(z["meta"]).decode())
    mats = {}
    objs = []
    for i, m in enumerate(meta["parts"]):
        if m["kind"] in hide:
            continue
        V = z[f"v{i}"].astype(np.float64) * 1e-3
        F = z[f"f{i}"]
        M = z[f"m{i}"].copy()
        M[:3, 3] *= 1e-3
        me = bpy.data.meshes.new(m["id"])
        me.from_pydata(V.tolist(), [], F.tolist())
        me.update()
        for poly in me.polygons:
            poly.use_smooth = False
        ob = bpy.data.objects.new(m["id"], me)
        ob.matrix_world = Matrix(M.tolist())
        bpy.context.scene.collection.objects.link(ob)
        key = m["material"]
        if key not in mats:
            if key in ("glass", "acrylic"):
                mats[key] = glass_material(key)
            elif key == "rubber":
                mats[key] = plain_material("rubber", (0.03, 0.03, 0.03), 0.6)
            elif key == "silicone":
                mats[key] = plain_material("silicone", (0.82, 0.82, 0.80), 0.45)
            else:
                mats[key] = wood_material(key, key, i)
        ob.data.materials.append(mats[key])
        # smooth shading for curved parts, keeping hard edges
        if m["kind"] in ("leg", "cap", "post", "bushing"):
            for poly in me.polygons:
                poly.use_smooth = True
            try:
                me.set_sharp_from_angle(angle=math.radians(35))
            except Exception:
                pass
        objs.append(ob)
    return objs, meta


def bbox(objs):
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for ob in objs:
        for c in ob.bound_box:
            w = ob.matrix_world @ Vector(c)
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    return lo, hi


def studio(sc, lo, hi, floor_kind):
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*lin((0.78, 0.79, 0.81)), 1)
    bg.inputs["Strength"].default_value = 0.35
    # floor + cyclorama backdrop
    size = 30.0
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    fl = bpy.context.active_object
    fl.name = "floor"
    if floor_kind == "tatami":
        fm = plain_material("floor", (0.55, 0.52, 0.38), 0.85)
    else:
        fm = plain_material("floor", (0.50, 0.485, 0.47), 0.75)
    fl.data.materials.append(fm)
    span = max(hi.x - lo.x, hi.y - lo.y)
    c = (lo + hi) / 2
    def area(name, loc, power, size_, color):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy = power
        ld.size = size_
        ld.color = color
        ob = bpy.data.objects.new(name, ld)
        ob.location = loc
        d = Vector((c.x, c.y, c.z * 0.6)) - Vector(loc)
        ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        sc.collection.objects.link(ob)
    r = max(span, 0.8)
    area("key", (c.x - 1.6 * r, c.y - 2.0 * r, 2.6 * r), 330 * r * r, 1.4 * r, (1.0, 0.94, 0.86))
    area("fill", (c.x + 2.4 * r, c.y - 0.8 * r, 1.3 * r), 70 * r * r, 2.4 * r, (0.86, 0.92, 1.0))
    area("rim", (c.x + 0.6 * r, c.y + 2.6 * r, 2.1 * r), 210 * r * r, 1.2 * r, (1.0, 0.97, 0.92))
    area("top", (c.x, c.y, 4.0 * r), 90 * r * r, 3.0 * r, (1, 1, 1))


def camera(sc, lo, hi, view, a):
    cam = bpy.data.cameras.new("cam")
    ob = bpy.data.objects.new("cam", cam)
    sc.collection.objects.link(ob)
    sc.camera = ob
    c = (lo + hi) / 2
    span = (hi - lo).length
    if view == "top":
        cam.type = "ORTHO"
        # ortho_scale spans the wider side of the frame: fit the table's depth too
        aspect = sc.render.resolution_x / max(1, sc.render.resolution_y)
        sx, sy = hi.x - lo.x, hi.y - lo.y
        cam.ortho_scale = max(sx, sy) * 1.12 if aspect < 1.0 else max(sx, sy * aspect) * 1.12
        ob.location = (c.x, c.y, hi.z + 3.0)
        ob.rotation_euler = (0, 0, 0)
        return
    az = {"hero": -38.0, "side": -90.0, "low": -25.0, "detail": -42.0, "front": -90.0}.get(view, -38.0)
    el = {"hero": 24.0, "side": 2.0, "low": 6.0, "detail": 18.0, "front": 14.0}.get(view, 24.0)
    lens = {"hero": 60.0, "side": 90.0, "low": 32.0, "detail": 120.0, "front": 70.0}.get(view, 60.0)
    if a.azimuth is not None:
        az = a.azimuth
    if a.elevation is not None:
        el = a.elevation
    if a.lens is not None:
        lens = a.lens
    cam.lens = lens
    fov = 2 * math.atan(18.0 / lens)
    target = Vector((c.x, c.y, (lo.z + hi.z) * 0.45))
    dist = 0.5 * span / math.sin(0.5 * fov) * 1.05
    if view == "detail":
        dist *= 0.33
        target = Vector((c.x + 0.25 * (hi.x - lo.x), c.y - 0.25 * (hi.y - lo.y), hi.z * 0.8))
    if view == "low":
        target = Vector((c.x, c.y, hi.z * 0.62))
        dist *= 0.62
    d = Vector((math.cos(math.radians(el)) * math.cos(math.radians(az)),
                math.cos(math.radians(el)) * math.sin(math.radians(az)),
                math.sin(math.radians(el))))
    ob.location = target + d * dist
    ob.rotation_euler = (target - ob.location).to_track_quat("-Z", "Y").to_euler()
    cam.dof.use_dof = view in ("detail",)
    if cam.dof.use_dof:
        cam.dof.focus_distance = dist
        cam.dof.aperture_fstop = 4.0


def main():
    a = args()
    sc = setup_render(a)
    hide = set(x for x in a.hide.split(",") if x)
    objs, meta = load_parts(a.npz, hide)
    lo, hi = bbox(objs)
    studio(sc, lo, hi, a.floor)
    camera(sc, lo, hi, a.view, a)
    sc.render.filepath = a.out
    bpy.ops.render.render(write_still=True)
    print("WROTE", a.out)


main()
