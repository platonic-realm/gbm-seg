import bpy
import numpy as np
import math
import sys

# Parse custom arguments
args = sys.argv

template_path = args[args.index('--') + 1]
verts_path = args[args.index('--') + 2]
faces_path = args[args.index('--') + 3]
values_path = args[args.index('--') + 4]
output_blend_path = args[args.index('--') + 5]
output_anim_path = args[args.index('--') + 6]


verts = np.load(verts_path)
faces = np.load(faces_path)
values = np.load(values_path)
edges = []

# Load the template Blender file
bpy.ops.wm.open_mainfile(filepath=template_path)

# Deselect all objects
bpy.ops.object.select_all(action='DESELECT')

# Select the default cube (assuming it has the default name "Cube")
if "Cube" in bpy.data.objects:
    bpy.data.objects["Cube"].select_set(True)
    bpy.ops.object.delete()  # Delete selected objects

# Calculate the center of the mesh
center = np.mean(verts, axis=0)

# Scale factor for downscaling the mesh
scale_factor = 0.02  # Adjust as needed

# Apply the scaling to the vertices
scaled_verts = (verts - center) * scale_factor

mesh = bpy.data.meshes.new("Layer")
mesh.from_pydata(scaled_verts, edges, faces)

# Calculate normals for the mesh
mesh.calc_normals_split()

obj = bpy.data.objects.new(mesh.name, mesh)
col = bpy.data.collections["Collection"]
col.objects.link(obj)

# Create a new attribute for the object
obj.data.attributes.new(name='values', type='FLOAT', domain='POINT')

# Normalize the values
none_zero_values = values[values > 0]
none_zero_3std = np.mean(none_zero_values) + 3 * np.std(none_zero_values)
values[values > none_zero_3std] = none_zero_3std

# The min is always zero, so we remove it from the equation
normalized_values = values / np.max(values)


# Store the values in the new attribute
obj.data.attributes['values'].data.foreach_set('value', normalized_values)

# Rotate the object -90 degrees around the Y-axis
rotation_angle = -90  # in degrees
rotation_axis = 'Y'
obj.rotation_euler.rotate_axis(rotation_axis, math.radians(rotation_angle))

# Step 1: Create a new material
material = bpy.data.materials.new(name="Material")

# Optional: Set the material to use nodes (useful for more complex materials)
material.use_nodes = True

# Modify the material (example: change the base color)
if material.use_nodes:
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Roughness'].default_value = 0.8
        bsdf.inputs['IOR'].default_value = 1.45

# Step 2: Get the object (replace 'Cube' with the name of your object)
obj = bpy.data.objects.get("Layer")

if obj is not None:
    # Check if the object already has materials
    if obj.data.materials:
        # Replace the first material
        obj.data.materials[0] = material
    else:
        # Add the new material to the object
        obj.data.materials.append(material)
else:
    print("Object not found")

# Step 3: Add an Attribute node to the material's node tree
if material.use_nodes:
    tree = material.node_tree

    # Add Attribute node
    attr_node = tree.nodes.new(type='ShaderNodeAttribute')
    attr_node.attribute_name = "values"

    # Add ColorRamp node
    color_ramp_node = tree.nodes.new(type='ShaderNodeValToRGB')
    color_ramp_node.color_ramp.interpolation = 'LINEAR'

    # Add a new color stop
    color_stop = color_ramp_node.color_ramp.elements.new(0.331818)
    # Set the color stop to A16266 at 0.33
    color_stop.color = (0.355303,
                        0.122556,
                        0.122556,
                        1.0)
    color_stop.position = 0.331818

    # Set existing color stop to C3C3C3 at 0.0
    color_ramp_node.color_ramp.elements[0].color = (0.543755,
                                                    0.543755,
                                                    0.543755,
                                                    1.0)
    color_ramp_node.color_ramp.elements[0].position = 0.0

    # Set existing color stop to C31E18 at 1.0
    color_ramp_node.color_ramp.elements[2].color = (0.543755,
                                                    0.01301,
                                                    0.009266,
                                                    1.0)  # C31E18
    color_ramp_node.color_ramp.elements[2].position = 1.0

    # Link Attribute node to ColorRamp node
    tree.links.new(attr_node.outputs['Fac'], color_ramp_node.inputs['Fac'])

    # Link ColorRamp node to Base Color input of Principled BSDF node
    tree.links.new(color_ramp_node.outputs['Color'], bsdf.inputs['Base Color'])

    # Add the Decimate modifier
    decimate_modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')

    # Set the ratio for decimation (0.5 reduces the number of faces by half)
    decimate_modifier.ratio = 0.125

    # Apply the modifier
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=decimate_modifier.name)

bpy.ops.wm.save_as_mainfile(filepath=output_blend_path)

# Set the render engine to Cycles
bpy.context.scene.render.engine = 'CYCLES'

print("----------------------------------------------")
print('setting up gpu ......')

bpy.context.scene.cycles.device = "GPU"
for scene in bpy.data.scenes:
    print(scene.name)
    scene.cycles.device = 'GPU'

bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

bpy.context.preferences.addons["cycles"].preferences.get_devices()
print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)

bpy.context.preferences.addons["cycles"].preferences.get_devices()
for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    d.use = True
    if d.type == 'CPU':
        d.use = False
    print("Device '{}' type {} : {}".format(d.name, d.type, d.use))
print('setting up gpu done')
print("----------------------------------------------")

# Enable only the specific GPU (e.g., first GPU)
target_device_index = 0  # Change this index to select a different GPU
bpy.context.preferences.addons['cycles'].preferences.devices[target_device_index].use = True

# Set the device to GPU
bpy.context.scene.cycles.device = 'GPU'

# Save the settings
bpy.ops.wm.save_userpref()

bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
bpy.context.scene.render.ffmpeg.format = 'MPEG4'
bpy.context.scene.render.filepath = output_anim_path

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 360

bpy.ops.render.render(animation=True)
