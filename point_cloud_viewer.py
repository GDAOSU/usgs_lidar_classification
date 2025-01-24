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
import shutil
from tiling import tiling

TEMP_DIR = r'/local/scratch0/hanyang/Codes/point_cloud_visualizer/tmp_dir'
os.makedirs(TEMP_DIR, exist_ok=True)


def clear_temp_folder():
    try:
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)
    except Exception as e:
        print(f"Error clearing temp folder: {e}")


def get_temp_file(suffix):
    return tempfile.NamedTemporaryFile(dir=TEMP_DIR, suffix=suffix, delete=False)


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
        gr.Markdown("## Point Cloud Classification Visualizer")

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
                    type="filepath",
                )
                method_dd = gr.Dropdown(
                    label='Selet Method',
                    choices=['PointTransformer', 'RandLA-Net'],
                    value="PointTransformer"
                )
                max_points_slider = gr.Slider(
                    label='Max Num. of Points',
                    minimum=1,
                    maximum=10_000_000,
                    value=100_000,
                    step=1
                )
            with gr.Column():
                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    classify_btn = gr.Button("Classify")
                with gr.Row():
                    download_downsampled_btn = gr.Button("Download Downsampled")
                    download_upsampled_btn = gr.Button("Download Upsampled")
                downsampled_classified_las_file_out = gr.File(visible=True, label="Downsampled Classified LAS")
                upsampled_classified_las_file_out = gr.File(visible=True, label="Upsampled Classified LAS")

        ##############################
        # Event handlers
        ##############################

        def on_file_input_and_slider_change(file_input, max_points):
            if file_input is None:
                clear_temp_folder()
                return None, None, None

            if not file_input.endswith('.las') and not file_input.endswith('.laz'):
                raise ValueError("Only .las and .laz files are supported")

            clear_temp_folder()
            global SUFFIX
            SUFFIX = '.las' if file_input.endswith('.las') else '.laz'
            if SUFFIX == '.las':
                original_las = laspy.read(file_input)
            elif SUFFIX == '.laz':
                original_las = laspy.read(file_input, laz_backend=laspy.LazBackend.Laszip)
            original_las_pth = shutil.copy(file_input, TEMP_DIR)
            original_las.write(original_las_pth)

            downsampled_las_pth = original_las_pth.replace(SUFFIX, '_downsampled.las')
            # original_las = laspy.read(original_las_pth)
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
                # strategy 1: use cube voxel size
                # voxel_size = np.cbrt((original_pcd.get_max_bound() - original_pcd.get_min_bound()).prod() / max_points)
                # strategy 2: use square voxel size
                # area = (original_pcd.get_max_bound() - original_pcd.get_min_bound())[:2].prod()
                # voxel_size = np.sqrt(area / max_points)
                # strategy 3: 1m x 1m x 1m voxel size
                voxel_size = 5 / original_las.points.scales[0]  # m / scale
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
            clear_temp_folder()
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

            # classification logic
            # 1. tiling point cloud
            conda_env_name = "o3d_ml"
            tiling_script = "/local/scratch0/hanyang/Codes/point_cloud_visualizer/tiling.py"
            tiling_save_dir = TEMP_DIR + '/tiles'
            block_width = 512
            os.system(
                f"conda run -n {conda_env_name} python {tiling_script} --in_path {downsampled_las_path} --out_dir {tiling_save_dir} --block_width {block_width}")
            # 2. inference using a different conda environment
            interface_script = "/local/scratch0/hanyang/Codes/point_cloud_visualizer/inference.py"
            test_folder = tiling_save_dir
            test_result_folder = TEMP_DIR + '/tile_classification_results'
            os.system(f"conda run -n {conda_env_name} python {interface_script} --test_folder {test_folder} --test_result_folder {test_result_folder}")
            # 3. merge classified tiles
            in_pkl_dir = tiling_save_dir
            in_label_dir = test_result_folder
            classified_las_path = downsampled_las_path.replace('.las', '_classified.las')
            os.system(f"conda run -n {conda_env_name} python /local/scratch0/hanyang/Codes/point_cloud_visualizer/stitching.py --in_pkl_dir {in_pkl_dir} --in_label_dir {in_label_dir} --out_las_path {classified_las_path}")

            classified_glb_path = classified_las_path.replace('.las', '.glb')
            convert_las_to_glb(classified_las_path, classified_glb_path)
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

        def on_download_downsampled(downsampled_classified_las_path, original_las_path):
            if downsampled_classified_las_path is None:
                return None

            if SUFFIX == '.laz':
                revised_downsampled_classified_las = laspy.read(downsampled_classified_las_path)
                new_las_path = downsampled_classified_las_path.replace('.las', '.laz')
                revised_downsampled_classified_las.write(new_las_path, laz_backend=laspy.LazBackend.Laszip)
                downsampled_classified_las_path = new_las_path
            return downsampled_classified_las_path

        download_downsampled_btn.click(
            fn=on_download_downsampled,
            inputs=[classified_las_path_state, original_las_path_state],
            outputs=[downsampled_classified_las_file_out]
        )

        def on_download_upsampled(original_las_path, downsampled_las_path, classified_las_path, upsampled_las_path):
            if original_las_path is None:
                return None
            original_las = laspy.read(original_las_path)
            downsampled_las = laspy.read(downsampled_las_path)

            classified_las = laspy.read(classified_las_path)
            classified_las_ori_new_x = (
                (classified_las.points['X'] * classified_las.header.scale[0] + classified_las.header.offset[0] - downsampled_las.header.offset[0]) /
                downsampled_las.header.scale[0]).astype(
                np.int32)
            classified_las_ori_new_y = (
                (classified_las.points['Y'] * classified_las.header.scale[1] + classified_las.header.offset[1] - downsampled_las.header.offset[1]) /
                downsampled_las.header.scale[1]).astype(
                np.int32)
            classified_las_ori_new_z = (
                (classified_las.points['Z'] * classified_las.header.scale[2] + classified_las.header.offset[2] - downsampled_las.header.offset[2]) /
                downsampled_las.header.scale[2]).astype(
                np.int32)
            
            classified_class = classified_las.points['classification']
            classified_points = np.column_stack((classified_las_ori_new_x, classified_las_ori_new_y, classified_las_ori_new_z, classified_class)).astype('float64')
            tree = cKDTree(classified_points[:, :3])
            original_xyz = np.column_stack((
                original_las.points['X'],
                original_las.points['Y'],
                original_las.points['Z'],
            )).astype(np.float64)
            _, idx = tree.query(original_xyz, k=1)
            original_las.points['classification'] = classified_points[idx, 3].astype(np.uint8)
            if SUFFIX == '.laz':
                upsampled_las_path = original_las_path.replace('.laz', '_upsampled_classified.laz')
            elif SUFFIX == '.las':
                upsampled_las_path = original_las_path.replace('.las', '_upsampled_classified.las')
            else:
                raise ValueError("Unknown file type")
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
    viewer.launch(debug=True, share=True)
