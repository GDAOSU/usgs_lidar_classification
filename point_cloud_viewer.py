from pygltflib import (
    GLTF2, Buffer, BufferView, Accessor, Mesh,
    Scene, Node, Attributes, Primitive, VEC3, FLOAT, ARRAY_BUFFER
)
import open3d as o3d
import gradio as gr
import os
import tempfile
import numpy as np
import laspy
from open3d import geometry
from scipy.spatial import cKDTree


def convert_las_to_glb(las_file_path, glb_file_path):
    # Read .las file
    las = laspy.read(las_file_path)
    points = las.points
    x = points['X']
    y = points['Y']
    z = points['Z']
    vertices = np.vstack((x, y, z)).T.astype(np.float32)
    print(f"Vertices shape: {vertices.shape}")

    # Define color table for classifications
    value_to_color = {
        0: [1.0, 1.0, 1.0],
        1: [1.0, 0.0, 0.0],
        2: [1.0, 0.5, 0.0],
        3: [1.0, 1.0, 0.0],
        4: [0.5, 1.0, 0.0],
        5: [0.0, 1.0, 0.0],
        6: [0.0, 1.0, 0.5],
        7: [0.0, 1.0, 1.0],
        8: [0.0, 0.5, 1.0],
        9: [0.0, 0.0, 1.0],
        10: [0.5, 0.0, 1.0],
        11: [0.75, 0.0, 1.0],
        12: [1.0, 0.0, 1.0],
        13: [1.0, 0.0, 0.75],
        14: [1.0, 0.25, 0.25],
        15: [0.75, 0.25, 0.25],
        16: [0.75, 0.5, 0.25],
        17: [0.75, 0.75, 0.25],
        18: [0.5, 0.75, 0.25]
    }

    # Assign colors based on classification values
    classifications = points['classification']
    colors = np.array([
        value_to_color.get(c, [0.5, 0.5, 0.5]) for c in classifications
    ], dtype=np.float32).flatten()

    # Create glTF data structure
    gltf = GLTF2()

    # Combine vertex + color data
    vertex_buffer_data = vertices.tobytes()
    color_buffer_data = colors.tobytes()
    total_buffer_data = vertex_buffer_data + color_buffer_data

    # Add Buffer
    buffer = Buffer(uri=None, byteLength=len(total_buffer_data))
    gltf.buffers.append(buffer)

    # Add BufferViews
    vertex_buffer_view = BufferView(
        buffer=0,
        byteOffset=0,
        byteLength=len(vertex_buffer_data),
        target=ARRAY_BUFFER
    )
    color_buffer_view = BufferView(
        buffer=0,
        byteOffset=len(vertex_buffer_data),
        byteLength=len(color_buffer_data),
        target=ARRAY_BUFFER
    )
    gltf.bufferViews.append(vertex_buffer_view)
    gltf.bufferViews.append(color_buffer_view)

    # Add Accessors
    vertex_accessor = Accessor(
        bufferView=0,
        componentType=FLOAT,
        count=len(vertices),
        type=VEC3
    )
    color_accessor = Accessor(
        bufferView=1,
        componentType=FLOAT,
        count=len(colors) // 3,
        type=VEC3
    )
    gltf.accessors.append(vertex_accessor)
    gltf.accessors.append(color_accessor)

    # Add Mesh
    attributes = Attributes(POSITION=0, COLOR_0=1)
    primitive = Primitive(attributes=attributes, mode=0)  # POINTS
    mesh = Mesh(primitives=[primitive])
    gltf.meshes.append(mesh)

    # Add Node and Scene
    node = Node(mesh=0)
    gltf.nodes.append(node)
    scene = Scene(nodes=[0])
    gltf.scenes.append(scene)
    gltf.scene = 0  # default scene

    # Pack and save as .glb
    gltf.set_binary_blob(total_buffer_data)
    gltf.save_binary(glb_file_path)
    print(f"GLB file saved at {glb_file_path}")
    return glb_file_path


def build_interface():
    with gr.Blocks() as main_viewer:
        gr.Markdown("## Point Cloud Viewer")

        original_las_path_state = gr.State()
        downsampled_las_path_state = gr.State()
        classified_las_path_state = gr.State()
        upsampled_classified_las_path_state = gr.State()

        with gr.Row():
            with gr.Column():
                viewer1 = gr.Model3D(
                    label='Original Point Cloud',
                )
            with gr.Column():
                viewer2 = gr.Model3D(
                    label="Classification Result",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                )
        with gr.Row():
            with gr.Column():
                file_input = gr.File(
                    label="Upload LAS File",
                    type="binary",
                )
                method_dd = gr.Dropdown(
                    label='Selet Method',
                    choices=['PointTransformer', 'RandLA-Net'],
                    value="PointTransformer"
                )
                max_points_slider = gr.Slider(
                    label='Max Num. of Points',
                    minimum=1,
                    maximum=1_000_000,
                    value=100_000,
                    step=1
                )
            with gr.Column():
                clear_btn = gr.Button("Clear")
                classify_btn = gr.Button("Classify")
                download_downsampled_btn = gr.Button("Download Downsampled")
                download_upsampled_btn = gr.Button("Download Upsampled")

                downsampled_classified_las_file_out = gr.File(visible=True)
                upsampled_classified_las_file_out = gr.File(visible=True)

        ##############################
        # Event handlers
        ##############################

        def on_file_input_and_slider_change(file_input, max_points):
            if file_input is None:
                return None, None, None

            with tempfile.NamedTemporaryFile(suffix='.las', delete=False) as tmp:
                tmp.write(file_input)
                tmp.close()
                original_las_pth = tmp.name

            downsampled_las_pth = original_las_pth.replace('.las', '_downsampled.las')
            original_las = laspy.read(original_las_pth)
            total_points = len(original_las)
            print(f"Total points: {total_points}")
            if total_points < max_points:
                downsampled_las = original_las
                downsampled_las.write(downsampled_las_pth)
            else:
                # use open3d voxel downsample
                x = original_las.points['X']
                y = original_las.points['Y']
                z = original_las.points['Z']
                vertices = np.column_stack((x, y, z)).astype('float64')

                original_pcd = o3d.geometry.PointCloud()
                original_pcd.points = o3d.utility.Vector3dVector(vertices)
                original_points = np.asarray(original_pcd.points)
                voxel_size = np.cbrt((original_pcd.get_max_bound() - original_pcd.get_min_bound()).prod() / max_points)
                quantized = np.floor(original_points / voxel_size)

                voxel_dict = {}
                indices = []
                for i, voxel in enumerate(quantized):
                    voxel_key = tuple(voxel)
                    if voxel_key not in voxel_dict:
                        voxel_dict[voxel_key] = i
                        indices.append(i)
                downsampled_points = original_las.points[indices].copy()

                downsampled_las = laspy.read(original_las_pth)
                downsampled_las.points = downsampled_points
                downsampled_las.write(downsampled_las_pth)

            # convert downsampled las to glb
            glb_path = downsampled_las_pth.replace('.las', '.glb')
            convert_las_to_glb(downsampled_las_pth, glb_path)

            return original_las_pth, downsampled_las_pth, glb_path

        file_input.change(
            fn=on_file_input_and_slider_change,
            inputs=[file_input, max_points_slider],
            outputs=[original_las_path_state, downsampled_las_path_state, viewer1]
        )

        max_points_slider.change(
            fn=on_file_input_and_slider_change,
            inputs=[file_input, max_points_slider],
            outputs=[original_las_path_state, downsampled_las_path_state, viewer1]
        )

        def on_clear():
            return (
                None, None, None, None, None, None, None, "PointTransformer", 100_000, None, None)

        clear_btn.click(
            fn=on_clear,
            inputs=[],
            outputs=[
                original_las_path_state,
                downsampled_las_path_state,
                classified_las_path_state,
                upsampled_classified_las_path_state,
                viewer1,
                viewer2,
                file_input,
                method_dd,
                max_points_slider,
                downsampled_classified_las_file_out,
                upsampled_classified_las_file_out,
            ]
        )

        def on_classify(original_las_path, downsampled_las_path, method):
            if original_las_path is None:
                return None, None

            # classify point cloud
            # TODO: implement classification logic
            downsampled_las = laspy.read(downsampled_las_path)
            classified_las = downsampled_las

            classified_las_path = downsampled_las_path.replace('_downsampled.las', '_classified.las')
            classified_las.write(classified_las_path)
            classified_glb_path = classified_las_path.replace('.las', '.glb')
            convert_las_to_glb(downsampled_las_path, classified_glb_path)
            return classified_las_path, classified_glb_path

        classify_btn.click(
            fn=on_classify,
            inputs=[
                original_las_path_state,
                downsampled_las_path_state,
                method_dd,
            ],
            outputs=[classified_las_path_state, viewer2]
        )

        def on_download_downsampled(downsampled_las_path):
            if downsampled_las_path is None:
                return None
            return downsampled_las_path

        download_downsampled_btn.click(
            fn=on_download_downsampled,
            inputs=[classified_las_path_state],
            outputs=[downsampled_classified_las_file_out]
        )

        def on_download_upsampled(original_las_path, downsampled_las_path, classified_las_path, upsampled_las_path):
            if original_las_path is None:
                return None
            # check if points in original las and downsampled las are the same
            original_las = laspy.read(original_las_path)
            downsampled_las = laspy.read(downsampled_las_path)
            ration = len(downsampled_las.points) / len(original_las.points)
            if ration > 0.9:
                # use classified result as upsampled result
                upsampled_las_path = classified_las_path
            else:
                classified_las = laspy.read(classified_las_path)
                classified_X = classified_las.points['X']
                classified_Y = classified_las.points['Y']
                classified_Z = classified_las.points['Z']
                classified_class = classified_las.points['classification']
                classified_points = np.column_stack((classified_X, classified_Y, classified_Z, classified_class)).astype('float64')
                tree = cKDTree(classified_points[:, :3])
                original_xyz = np.column_stack((
                    original_las.points['X'],
                    original_las.points['Y'],
                    original_las.points['Z'],
                )).astype(np.float64)
                _, idx = tree.query(original_xyz, k=1)
                original_las.points['classification'] = classified_points[idx, 3].astype(np.uint8)
                upsampled_las_path = original_las_path.replace('.las', '_upsampled_classified.las')
                original_las.write(upsampled_las_path)
            return upsampled_las_path

        download_upsampled_btn.click(
            fn=on_download_upsampled,
            inputs=[
                original_las_path_state,
                downsampled_las_path_state,
                classified_las_path_state,
                upsampled_classified_las_path_state,
            ],
            outputs=[upsampled_classified_las_file_out]
        )
    return main_viewer


if __name__ == "__main__":
    viewer = build_interface()
    viewer.launch(debug=True)
