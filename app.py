import streamlit as st

st.set_page_config(
    page_title="DermSynth3D",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": "https://github.com/sfu-mial/DermSynth3D",
        "Report a bug": "https://github.com/sfu-mial/DermSynth3D/issues",
        "About": "This is the demo app to try out the pipeline proposed in the paper DermSynth3D: A Dermatological Image Synthesis Framework for 3D Skin Lesions",
    },
)
from stpyvista import stpyvista
import pandas as pd
import numpy as np
from glob import glob
import os, sys
from PIL import Image
import torch
import torch.nn as nn
import pyrender
import trimesh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pyvista as pv
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
)
import math
from trimesh import transformations as tf
import os

import streamlit.components.v1 as components
from math import pi
from IPython.display import display
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly

plotly.__version__
import plotly.graph_objects as go
from skimage import io
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot

init_notebook_mode(connected=True)

view_width = 400
view_height = 400

import mediapy as mpy

sys.path.append("./dermsynth3d")
sys.path.append("./skin3d/")
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes

st.title("DermSynth3D - Dermatological Image Synthesis Framework")
# st.write('A dermatological image synthesis framework for 3D skin lesions')


def setup_paths():
    # get the meshes
    mesh_paths = glob("./data/3dbodytex-1.1-highres/*/*.obj")
    mesh_names = [os.path.basename(os.path.dirname(x)) for x in mesh_paths]

    # get the textures
    all_textures = glob("./data/3dbodytex-1.1-highres/*/*.png")

    get_no_lesion_path = lambda x, y: os.path.join(
        "./data/3dbodytex-1.1-highres", x, "model_highres_0_normalized.png"
    )
    get_mesh_path = lambda x: os.path.join(
        "./data/3dbodytex-1.1-highres", x, "model_highres_0_normalized.obj"
    )
    # get the textures with the lesions
    get_mask_path = lambda x, y: os.path.join(
        "./data/processed_textures/", x, "model_highres_0_normalized.png"
    )
    get_dilated_lesion_path = lambda x, y: os.path.join(
        "./data/processed_textures/",
        x,
        f"model_highres_0_normalized_dilated_lesion_{y}.png",
    )
    get_blended_lesion_path = lambda x, y: os.path.join(
        "./data/processed_textures/",
        x,
        f"model_highres_0_normalized_blended_lesion_{y}.png",
    )
    get_pasted_lesion_path = lambda x, y: os.path.join(
        "./data/processed_textures/",
        x,
        f"model_highres_0_normalized_pasted_lesion_{y}.png",
    )
    get_texture_module = lambda x: getattr(
        sys.modules[__name__],
        f"get_{x.lower().replace(' ', '_')}_path",
    )
    # Update the global namespace with the functions
    global_namespace = globals()
    global_namespace.update(
        {
            "mesh_paths": mesh_paths,
            "mesh_names": mesh_names,
            "all_textures": all_textures,
            "get_no_lesion_path": get_no_lesion_path,
            "get_mesh_path": get_mesh_path,
            "get_mask_path": get_mask_path,
            "get_dilated_lesion_path": get_dilated_lesion_path,
            "get_blended_lesion_path": get_blended_lesion_path,
            "get_pasted_lesion_path": get_pasted_lesion_path,
            "get_texture_module": get_texture_module,
        }
    )


@st.cache_data
def set_texture_map_py3d(mesh_name, texture_name, num_lesion=1, device="cpu"):
    mesh_path = get_mesh_path(mesh_name)
    texture_path = get_texture_module(texture_name)(mesh_name)
    mesh = load_objs_as_meshes([mesh_path], device=device)
    texture_img = Image.open(texture_path).convert("RGB")
    texture_tensor = torch.from_numpy(np.array(texture_img))

    tmap = TexturesUV(
        maps=texture_tensor.float().to(device=mesh.device).unsqueeze(0),
        verts_uvs=mesh.textures.verts_uvs_padded(),
        faces_uvs=mesh.textures.faces_uvs_padded(),
    )
    new_mesh = Meshes(
        verts=mesh.verts_padded(), faces=mesh.faces_padded(), textures=tmap
    )
    return new_mesh, texture_img


import tempfile


def render_images(mesh_name, texture_name, num_lesion=1, device="cuda"):
    raise NotImplementedError


@st.cache_data
def get_trimesh_attrs(mesh_name):
    mesh_path = get_mesh_path(mesh_name)
    tri_mesh = trimesh.load(mesh_path)
    angle = -math.pi / 2
    direction = [0, 1, 0]
    center = [0, 0, 0]
    rot_matrix = tf.rotation_matrix(angle, direction, center)
    tri_mesh = tri_mesh.apply_transform(rot_matrix)
    tri_mesh.apply_transform(tf.rotation_matrix(math.pi, [0, 0, 1], [-1, -1, -1]))

    verts, faces = tri_mesh.vertices, tri_mesh.faces
    uvs = tri_mesh.visual.uv
    colors = tri_mesh.visual.to_color()
    vc = colors.vertex_colors  # / 255.0
    timg = tri_mesh.visual.material.image

    return verts, faces, vc, mesh_name


@st.cache_data
def plotly_image(image):
    fig = go.Figure()
    fig.add_trace(go.Image(z=image))
    fig.update_layout(
        width=view_width,
        height=view_height,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


@st.cache_data
def plotly_mesh(verts, faces, vc, mesh_name):
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=vc,
            )
        ]
    )
    fig.update_layout(scene_aspectmode="manual", scene_aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False)))
    fig.update_layout(scene=dict(zaxis=dict(visible=False)))
    fig.update_layout(scene=dict(camera=dict(up=dict(x=1, y=0, z=1))))
    fig.update_layout(scene=dict(camera=dict(eye=dict(x=-2, y=-2, z=-1))))
    # fig.update_layout(scene=dict(camera=dict(center=dict(x=0, y=0, z=0))))

    return fig


@st.cache_data
def load_mesh_and_texture(mesh_name, texture_name, num_lesion=1, device="cuda"):
    mesh_path = get_mesh_path(mesh_name)
    texture_path = get_texture_module(texture_name)(mesh_name, num_lesion)
    # glb_mesh = convert_to_glb(mesh_path)
    mesh = load_objs_as_meshes([mesh_path], device=device)
    verts = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()
    normals = mesh.verts_normals_packed().detach().cpu().numpy()
    # tri_mesh = trimesh.load(mesh_path)

    texture_img = Image.open(texture_path).convert("RGB")
    texture_tensor = torch.from_numpy(np.array(texture_img))

    tmap = TexturesUV(
        maps=texture_tensor.float().to(device=mesh.device).unsqueeze(0),
        verts_uvs=mesh.textures.verts_uvs_padded(),
        faces_uvs=mesh.textures.faces_uvs_padded(),
    )
    new_mesh = Meshes(
        verts=mesh.verts_padded(), faces=mesh.faces_padded(), textures=tmap
    )
    pl_mesh = plotly_mesh(*get_trimesh_attrs(mesh_name))
    # print(tri_mesh, new_mesh, texture_img.resize((256, 256)))
    return pl_mesh, new_mesh, texture_img  # .resize((256, 256))


@st.cache_resource
def display_mesh(mesh_name, texture_name, num_lesion=1, device="cuda"):
    tri_mesh, render_mesh, texture_img = load_mesh_and_texture(
        mesh_name, texture_name, num_lesion, device
    )

    plotter = pv.Plotter(border=True, window_size=[view_width, view_width])
    pv_mesh = pv.wrap(tri_mesh)
    plotter.add_mesh(pv_mesh)
    plotter.background_color = "white"
    plotter.view_isometric()
    return plotter, render_mesh, texture_img


@st.cache_data
def setup_cameras(dist, elev, azim, device="cuda"):
    R, T = look_at_view_transform(dist, elev, azim, degrees=True)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    return cameras


@st.cache_data
def setup_lights(
    light_pos, ambient_color, diffuse_color, specular_color, device="cuda"
):
    lights = PointLights(
        device=device,
        location=[[light_pos, light_pos, light_pos]],
        ambient_color=[[ambient_color, ambient_color, ambient_color]],
        diffuse_color=[[diffuse_color, diffuse_color, diffuse_color]],
        specular_color=[[specular_color, specular_color, specular_color]],
    )
    return lights


@st.cache_data
def setup_materials(shininess, specularity, device="cuda"):
    materials = Materials(
        device=device,
        specular_color=[[specularity, specularity, specularity]],
        shininess=[shininess],
    )
    return materials


# @st.cache_data
def setup_renderer(cameras, lights, materials, device="cuda"):
    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius=0.0,
        faces_per_pixel=10,
        # max_faces_per_bin=100,
        bin_size=0,
        perspective_correct=True,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights, materials=materials
        ),
    )
    return renderer


# @st.cache_resource
def render_images(renderer, mesh, lights, cameras, materials, nviews, device="cuda"):
    images = renderer(mesh, lights=lights, cameras=cameras, materials=materials)
    return images


def main():
    st.sidebar.title("Mesh")
    selected_mesh = st.sidebar.selectbox(
        "Select the mesh to be used for the synthesis", mesh_names
    )

    # set the texture
    st.sidebar.title("Texture Map")
    selected_texture = st.sidebar.selectbox(
        "Select the texture map to view",
        ["No Lesion", "Pasted Lesion", "Blended Lesion", "Dilated Lesion"],
    )

    st.sidebar.title("Lesion Count")
    num_lesion = st.sidebar.slider(
        "Set the number of lesions to be added to the mesh",
        min_value=0,
        max_value=30,
        value=1,
        step=1,
    )
    if num_lesion not in [0, 1, 2, 5, 10]:
        st.sidebar.error("The number of lesions is not in the default list!")
        st.sidebar.warning("We currently only support 1, 5, 15, 30.")
        num_lesion = 1

    # load the texture
    texture_img = Image.open(
        get_texture_module(selected_texture)(selected_mesh, num_lesion)
    )

    # display the mesh and texture
    # based on the  selected parameters
    with st.spinner(text="Loading Mesh with texture..."):
        mesh_place, texture_place = st.columns(2)
        tmesh, render_mesh, texture_img = load_mesh_and_texture(
            selected_mesh, selected_texture, num_lesion
        )
        with mesh_place and st.spinner("Loading Mesh..."):
            # mesh_place.info("click on reset camera, if unable to see the whole mesh!")
            mesh_place.plotly_chart(tmesh, use_container_width=True, theme=None)
            # mesh_place.info("The mesh will be displayed here. Please wait...")
            # st.sidebar.success("Mesh with texture loaded!")
            # stpyvista(tmesh, key="mesh")
        with texture_place and st.spinner("Loading texture..."):
            pl_img = plotly_image(texture_img.resize((512, 512)))
            texture_place.plotly_chart(pl_img, use_container_width=True, theme=None)
            # texture_place.info("The texture will be displayed here. Please wait...")
            # texture_place.image(
            #     texture_img,
            #     caption=f"Texture map with {selected_texture} for {selected_mesh}",
            #     use_column_width="auto",
            #     # width=view_width,
            #     clamp=True,
            #     channels="RGB",
            # )
        st.sidebar.success("Mesh with texture loaded!", icon="👏")
        finished_loading = True
        # randomize the rendering parameters
    st.sidebar.title("Randomize View Parameters")
    activated = st.sidebar.toggle("Randomize?", value=False)
    # default = False
    st.session_state["randomized"] = False
    if activated and finished_loading:
        with st.spinner("Randomizing..."):
            # set the camera parameters
            dist = np.random.uniform(0.0, 1.0)
            elev = np.random.uniform(0.0, 1.0)
            azim = np.random.uniform(0.0, 1.0)
            # set the lighting parameters
            light_pos = np.random.uniform(0.0, 1.0)
            ambient_color = np.random.uniform(0.0, 1.0)
            diffuse_color = np.random.uniform(0.0, 1.0)
            specular_color = np.random.uniform(0.0, 1.0)
            # set the material parameters
            shininess = np.random.uniform(0.0, 1.0)
            specularity = np.random.uniform(0.0, 1.0)
            camera_values = setup_cameras(dist, elev, azim)
            light_values = setup_lights(
                light_pos, ambient_color, diffuse_color, specular_color
            )
            material_values = setup_materials(shininess, specularity)
            renderer = setup_renderer(camera_values, light_values, material_values)
            st.session_state["camera_values"] = camera_values
            st.session_state["light_values"] = light_values
            st.session_state["material_values"] = material_values
            st.session_state["renderer"] = renderer
            st.session_state["randomized"] = True
            if (
                "camera_values" in st.session_state
                and "light_values" in st.session_state
                and "material_values" in st.session_state
                and "renderer" in st.session_state
            ):
                st.sidebar.success("Randomization done!", icon="👏")
    else:
        if not st.session_state["randomized"]:
            st.sidebar.warning(
                "Randomization is disabled!\nDefine the rendering parameters!"
            )
        with st.spinner("Set View Parameters"):
            # with st.sidebar.expander("Set View Parameters", expanded=True) as view_cont:
            # camera parameters
            # st.sidebar.header("View Parameters")
            cam = st.sidebar.form("Change camera parameters")
            cam.subheader("Camera Parameters")  # , expanded=False)
            dist = cam.slider(
                "Distance", min_value=0.0, max_value=10.0, value=0.5, step=0.5
            )
            elev = cam.slider(
                "Elevation", min_value=-90, max_value=90, value=0, step=10
            )
            azim = cam.slider("Azimuth", min_value=-90, max_value=90, value=90, step=10)
            # camera_values = cam.form_submit_button(
            #     "Update Camera Parameters",
            #     on_click=setup_cameras,
            #     args=(dist, elev, azim),
            # )
            # cam.form_submit_button("Update Camera Parameters")
            # camera_values = setup_cameras(dist, elev, azim)

            # print(camera_values)
            # cam.warning(
            #     "*Note:* The camera parameters are set to the default values used in the paper"
            # )

            # lighting parameters
            # st.sidebar.subheader("Lighting Parameters")
            # light = st.sidebar.expander("Change lighting parameters", expanded=False)
            # light = st.sidebar.form("Change lighting parameters")  # , expanded=False)
            # light_pos = light.slider(
            #     "Light Position", min_value=0.0, max_value=1.0, value=0.5, step=0.1
            # )
            # light_ac = light.slider(
            #     "Ambient Color", min_value=0.0, max_value=1.0, value=0.5, step=0.1
            # )
            # light_dc = light.slider(
            #     "Diffuse Color", min_value=0.0, max_value=1.0, value=0.5, step=0.1
            # )
            # light_sc = light.slider(
            #     "Specular Color", min_value=0.0, max_value=1.0, value=0.5, step=0.1
            # )
            cam.subheader("Lighting Parameters")  # , expanded=False)
            light_pos = cam.slider(
                "Light Position", min_value=0.0, max_value=1.0, value=0.5, step=0.1
            )
            light_ac = cam.slider(
                "Ambient Color", min_value=0.0, max_value=1.0, value=0.5, step=0.1
            )
            light_dc = cam.slider(
                "Diffuse Color", min_value=0.0, max_value=1.0, value=0.5, step=0.1
            )
            light_sc = cam.slider(
                "Specular Color", min_value=0.0, max_value=1.0, value=0.5, step=0.1
            )
            # light.form_submit_button("Update Lighting Parameters")
            # light_values = setup_lights(light_pos, light_ac, light_dc, light_sc)
            # light.warning(
            #     "*Note:* The lighting parameters are set to the default values used in the paper"
            # )
            # print(light_values)
            # material parameters
            # st.sidebar.write("Material Parameters")
            # mat = st.sidebar.expander("Change material parameters", expanded=False)
            # mat = st.sidebar.form("Change material parameters")  # , expanded=False)

            # mat_sh = mat.slider(
            #     "Shininess", min_value=0, max_value=100, value=50, step=10
            # )
            # mat_sc = mat.slider(
            #     "Specularity", min_value=0.0, max_value=1.0, value=0.5, step=0.1
            # )
            cam.subheader("Material Parameters")  # , expanded=False)
            mat_sh = cam.slider(
                "Shininess", min_value=0, max_value=100, value=50, step=10
            )
            mat_sc = cam.slider(
                "Specularity", min_value=0.0, max_value=1.0, value=0.5, step=0.1
            )
            # mat.form_submit_button("Update Material Parameters")
            # material_values = setup_materials(mat_sh, mat_sc)
            # mat.warning(
            #     "*Note:* The material parameters are set to the default values used in the paper"
            # )
            # update_button = st.form_submit_button("Update Parameters")
            cam.form_submit_button("Update View Parameters")
            camera_values = setup_cameras(dist, elev, azim)
            light_values = setup_lights(light_pos, light_ac, light_dc, light_sc)
            material_values = setup_materials(mat_sh, mat_sc)
            renderer = setup_renderer(camera_values, light_values, material_values)

            st.session_state["selected_camera_values"] = camera_values
            st.session_state["selected_light_values"] = light_values
            st.session_state["updated_renderer"] = renderer
            st.session_state["selected_material_values"] = material_values
    # Rendered Views
    st.header("Rendered Views")
    st.info("The rendered views will be displayed here. Click on the button to render!")
    with st.form("Render Views"):
        number_of_views = st.slider(
            "Number of views to be rendered", 2, 16, 4, 2
        )  # , key="num_views")
        render_button = st.form_submit_button("Render Views")
        st.session_state["number_of_views"] = number_of_views
        st.session_state["render_button"] = render_button
    if st.session_state["render_button"]:
        with st.spinner("Rendering..."):
            images = render_images(
                renderer,
                render_mesh,
                light_values,
                camera_values,
                material_values,
                number_of_views,
            )
            images = images.detach().cpu().numpy()
            rendered_img = plotly_image(images[0][..., :3])
            silhouette_img = plotly_image(images[0][..., 3:])
            st.plotly_chart(rendered_img, use_container_width=True, theme=None)


if __name__ == "__main__":
    setup_paths()
    main()
