from gloss import ViewerHeadless
from gloss.types import PointColorType
from gloss.components import (
    Verts, 
    VisPoints,
    Colors,
)

from smpl_rs import SmplCache
from smpl_rs.plugins import SmplPlugin
from smpl_rs.types import SmplType, Gender, UpAxis, FollowerType
from smpl_rs.components import (
    SmplParams,
    Betas,
    Animation,
    GlossInterop,
    Follower,
)



import numpy as np 


def select_smpl_type_gender(smpl_type='smplx', gender='neutral'):
    smpl_type = {'smplx': SmplType.SmplX, 'smpl': SmplType.SmplH}[smpl_type]
    smpl_gender = {'neutral': Gender.Neutral, 'female': Gender.Female, 'male': Gender.Male}[gender]
    return smpl_type, smpl_gender



def prepare_smpl_model(smpl_model_path, smpl_type, smpl_gender):
    #insert a resource which is a component that can be shared between multiple entities
    #this one just lazy loads all smpl models you might need
    smpl_models = SmplCache.default()
    smpl_models.set_lazy_loading(smpl_type, smpl_gender, smpl_model_path)
    return smpl_models



class HeadlessVisualizer():
    def __init__(self, 
                 config_path=None, 
                 smpl_model_path=None, 
                 smpl_type='smplx', 
                 gender='neutral', 
                 fps=30, 
                 frame_size=(512, 512), 
                 with_default_texture=True, 
                 floor_type='default'):
        if floor_type == 'no_floor':
            config_path = config_path.replace('viewer_config.toml', 'no_floor_config.toml')
        elif floor_type == 'low_floor':
            config_path = config_path.replace('viewer_config.toml', 'viewer_config_lowfloor.toml')
        else:
            pass
        
        self.viewer = ViewerHeadless(*frame_size, config_path=config_path) #resolution of the rendering window
        
        self.device=self.viewer.get_device()
        self.queue=self.viewer.get_queue()
        self.scene=self.viewer.get_scene()
        self.camera=self.viewer.get_camera()

        self.animation_fps = fps
        self.smpl_model_path = smpl_model_path
        self.smpl_type, self.smpl_gender = select_smpl_type_gender(smpl_type=smpl_type, gender=gender)
        #insert a resource which is a component that can be shared between multiple entities
        self.scene.add_resource(prepare_smpl_model(self.smpl_model_path, self.smpl_type, self.smpl_gender))
        
        #insert a plugin which governs the logic functions that run on the entities depending on the components they have
        self.viewer.insert_plugin(SmplPlugin(autorun=True))

        self.with_default_texture = with_default_texture
        self.up_axis = UpAxis.Y
    
    

    def load_smpl_motion_sequence(self, poses, trans, betas=None,):
        #read a mesh from file and add it to the scene
        mesh = self.viewer.get_or_create_entity(name="mesh")
        
        #insert the needed components
        mesh.insert(SmplParams(self.smpl_type, self.smpl_gender, enable_pose_corrective=True))
        if betas is not None:
            mesh.insert(Betas(betas.astype(np.float32)))
        else:
            mesh.insert(Betas.default())

        mesh.insert(
            Animation.from_matrices(
                poses.astype(np.float32), 
                trans.astype(np.float32), 
                None,
                fps=self.animation_fps,
                up_axis=self.up_axis,
                smpl_type=self.smpl_type
            )
        )
        print(f'[data loading] motion length = {poses.shape[0]} frames')
        #mesh.insert(PoseOverride.allow_all().overwrite_hands(HandType.Relaxed))
        #mesh.insert(DiffuseImg(path_diffuse))
        #mesh.insert(NormalImg(path_normal))

        if self.with_default_texture:
            mesh.insert(GlossInterop(with_uv=True))
        else:
            raise NotImplementedError
            # mesh.insert(GlossInterop(with_uv=False))
            # mesh.insert(VisMesh(color_type=MeshColorType.PerVert))
        
        return mesh


    
    def let_camera_following_this_mesh(self, mesh):
        '''
        hey! those are the parameters that control how that mesh is being followed. The Camera and the Lights in the scene will follow a certain mesh and will smoothly interpolate their position to match the mesh. For the args
        max_strength = how aggressive the following is. If it's a low value the camera will react slowly to the mesh changing position but if it's high then it will behave as if it's "locked" to look at that entity. Its sort of an arbitrary parameter and for most cases i guess the default would work well enough.
        dist_start = movement below 10cm will not be tracked or followed. This is to avoid the mesh moving slightly and the camera trying to immediately follow it which leads to jitter. Essentially at <10cm (dist_start) the strength of following is 0
        dist_end = distance between the lookat point of the camera and the mesh centroid where the strength of following is maximum (the max_strength) so if the mesh is very far away from where the camera is looking. the camera needs to react quickly to this and move towards it.
        
        '''
        #allows the camera to follow the movement of this entity
        print('[visualizer] set the cam to follow')
        self.scene.add_resource(
            Follower(
                max_strength=50.0, dist_start=2, dist_end=10,
                follower_type=FollowerType.CamAndLights
            )
        )
    


    def mesh_go_next_frame(self, mesh, contacts=None, fps=30, advance=True):
        self.viewer.start_frame()  
        
        # Advance the animation
        anim=mesh.get(Animation)
        anim.pause() 
        if advance:
            anim.advance_sec(1/fps)
        current_time = anim.get_cur_time_sec()
        current_frame_id = round(current_time * fps)
        # print(f'current_time={current_time:.5f}, current_frame_id={current_frame_id:d}')

        mesh.insert(anim) #update the animation
        self.viewer.update()

        return current_time, current_frame_id
    

    def mesh_kpts_go_next_frame(
            self, 
            mesh, 
            kpts, 
            kpts_color, 
            fps=30, advance=True):
        self.viewer.start_frame()  
        anim=mesh.get(Animation)
        anim.pause() #so that the SmplPlugin doesn't automatically advance the animation
        if advance:
            anim.advance_sec(1/fps)
        current_time = anim.get_cur_time_sec()
        current_frame_id = round(current_time * fps)
        mesh.insert(anim) #update the animation

        # visualize point cloud
        if kpts_color.sum()>=0:
            # cloud = self.scene.get_or_spawn_renderable('cloud')
            cloud = self.viewer.get_or_create_entity(name = "cloud")
            cloud.insert(Verts(kpts))
            cloud.insert(Colors(kpts_color))
            ## it has bugs
            cloud.insert(
                VisPoints(show_points=True, 
                    point_size=5.0, 
                    color_type=PointColorType.PerVert
                )
            )

        self.viewer.update()
        return current_time, current_frame_id


    def get_rendering(self):
        #print(self.camera.position)
        #get the last image we rendered and save it to disk
        tex=self.viewer.get_final_tex()
        tex_numpy = tex.numpy(self.device, self.queue)
        return tex_numpy
    

    def set_camera_positions(self, position=(0,-4,0), lookat=(0,0,0)):
        self.camera.set_position((position[0], position[1], position[2])) #xyz right-hand coordinate system
        self.camera.set_lookat((lookat[0], lookat[1], lookat[2]))


    def whether_animation_end(self, mesh):
        return mesh.get(Animation).is_finished()


    def stop_mesh_animation(self, mesh):
        anim=mesh.get(Animation)
        anim.pause()
