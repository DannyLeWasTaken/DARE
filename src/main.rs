use crate::graphics::acceleration_structure::BLASBufferAddresses;
use anyhow::Result;
use phobos::domain::All;
use phobos::util::deferred_delete::DeletionQueue;
use phobos::vk;
use phobos::{GraphicsCmdBuffer, IncompleteCmdBuffer, RecordGraphToCommandBuffer};
use std::ops::Deref;
use std::sync::{Arc, RwLock};
use winit::event::Event;

mod app;
mod asset;
mod graphics;
mod spirv;
mod utils;
mod world;

/// The basic renderer
struct Raytracing {
    scene: Arc<RwLock<asset::Scene>>,
    acceleration_structure: Arc<graphics::acceleration_structure::SceneAccelerationStructure>,
    sampler: phobos::Sampler,
    camera: graphics::camera::Camera,
    last_frame_time: std::time::Instant,
    delta_time: f32,
    last_camera_pos: Option<glam::Vec2>,
    left_mouse_button_down: bool,
    ctx: app::Context,
    render_extent: vk::Extent2D,
    color: Attachment,
    deferred_delete: DeletionQueue<Attachment>,
}

struct Attachment {
    image: phobos::Image,
    view: phobos::ImageView,
}

impl Attachment {
    pub fn new(
        ctx: &mut app::Context,
        format: vk::Format,
        extent: vk::Extent2D,
        extra_usage: vk::ImageUsageFlags,
    ) -> Result<Self> {
        let (usage, aspect) = if format == vk::Format::D32_SFLOAT {
            (
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                vk::ImageAspectFlags::DEPTH,
            )
        } else {
            (
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::ImageAspectFlags::COLOR,
            )
        };
        let image = phobos::Image::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            phobos::image::ImageCreateInfo {
                width: extent.width,
                height: extent.height,
                depth: 1,
                usage: usage | extra_usage,
                format,
                samples: vk::SampleCountFlags::TYPE_1,
                mip_levels: 1,
                layers: 1,
            },
        )?;
        let view = image.whole_view(aspect)?;
        Ok(Self { image, view })
    }
}

/// Debugging purposes only: gets the gltf name from my folder
fn gltf_sample_name(name: &str) -> String {
    format!(
        "C:/Users/Danny/Documents/glTF-Sample-Models/2.0/{0}/glTF/{0}.gltf",
        name
    )
}

impl app::App for Raytracing {
    fn new(mut ctx: app::Context) -> Result<Self>
    where
        Self: Sized,
    {
        let loader = asset::gltf_asset_loader::GltfAssetLoader::new();
        let scene = loader.load_asset_from_file(
            std::path::Path::new(
                //gltf_sample_name("BoxTextured").as_str(),
                gltf_sample_name("BoomBoxWithAxes").as_str(),
                //gltf_sample_name("Sponza").as_str(),
                //gltf_sample_name("ABeautifulGame").as_str(),
                //gltf_sample_name("Lantern").as_str(),
                //"C:/Users/Danny/Documents/deccer-cubes/SM_Deccer_Cubes_With_Rotation.gltf",
                //"C:/Users/Danny/Documents/deccer-cubes/SM_Deccer_Cubes_Textured_Complex.gltf",
                //"C:/Users/Danny/Documents/deccer-cubes/SM_Deccer_Cubes_Textured.gltf",
                //"C:/Users/Danny/Documents/Assets/Junk Shop/Blender 2.gltf",
                //"C:/Users/Danny/Documents/Assets/Classroom/classroom.gltf",
                //"C:/Users/Danny/Documents/Assets/among_us_astronaut_-_clay/scene.gltf",
            ),
            &mut ctx,
        );
        let scene_acceleration_structure =
            graphics::acceleration_structure::convert_scene_to_blas(&mut ctx, &scene);

        let rgen = spirv::create_shader("shaders/raygen.spv", vk::ShaderStageFlags::RAYGEN_KHR);
        let rchit =
            spirv::create_shader("shaders/rayhit.spv", vk::ShaderStageFlags::CLOSEST_HIT_KHR);
        let rmiss = spirv::create_shader("shaders/raymiss.spv", vk::ShaderStageFlags::MISS_KHR);

        let pci = phobos::RayTracingPipelineBuilder::new("rt")
            .max_recursion_depth(1)
            .add_ray_gen_group(rgen)
            .add_ray_hit_group(Some(rchit), None)
            .add_ray_miss_group(rmiss)
            .build();
        ctx.resource_pool
            .pipelines
            .create_named_raytracing_pipeline(pci)?;

        // Load shader
        let vertex = spirv::load_spirv_file(std::path::Path::new("shaders/vert.spv"));
        let fragment = spirv::load_spirv_file(std::path::Path::new("shaders/frag.spv"));

        let vertex = phobos::ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::VERTEX, vertex);
        let fragment =
            phobos::ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::FRAGMENT, fragment);

        // Now we can start using the pipeline builder to create our full pipeline.
        let pci = phobos::PipelineBuilder::new("sample".to_string())
            .vertex_input(0, vk::VertexInputRate::VERTEX)
            .vertex_attribute(0, 0, vk::Format::R32G32_SFLOAT)?
            .vertex_attribute(0, 1, vk::Format::R32G32_SFLOAT)?
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .blend_attachment_none()
            .cull_mask(vk::CullModeFlags::NONE)
            .attach_shader(vertex)
            .attach_shader(fragment)
            .build();

        // Store the pipeline in the pipeline cache
        ctx.resource_pool.pipelines.create_named_pipeline(pci)?;

        let sampler = phobos::Sampler::default(ctx.device.clone())?;

        Ok(Self {
            scene: Arc::new(RwLock::new(scene)),
            acceleration_structure: Arc::new(scene_acceleration_structure),
            sampler,
            camera: graphics::camera::Camera::new(),
            last_frame_time: std::time::Instant::now(),
            delta_time: 0.0,
            last_camera_pos: None,
            left_mouse_button_down: false,
            render_extent: vk::Extent2D {
                width: 800,
                height: 600,
            },
            color: make_attachments(
                ctx.clone(),
                vk::Extent2D {
                    width: 800,
                    height: 600,
                },
            )
            .unwrap(),
            deferred_delete: DeletionQueue::new(4),
            ctx,
        })
    }

    fn frame(
        &mut self,
        ctx: app::Context,
        ifc: phobos::InFlightContext,
    ) -> Result<phobos::sync::submit_batch::SubmitBatch<phobos::domain::All>> {
        let window_aspect_ratio =
            self.render_extent.width as f32 / self.render_extent.height as f32;
        let swap = phobos::image!("swapchain");
        let rt_image = phobos::image!("rt_out");

        let mut pool = phobos::pool::LocalPool::new(ctx.resource_pool.clone())?;

        let rt_pass = phobos::PassBuilder::new("raytrace")
            .write_storage_image(&rt_image, phobos::PipelineStage::RAY_TRACING_SHADER_KHR)
            .execute_fn(|cmd, ifc, bindings, _| {
                let view = self.camera.view;
                let projection = glam::Mat4::perspective_rh(
                    90.0_f32.to_radians(),
                    self.render_extent.width as f32 / self.render_extent.height as f32,
                    0.001,
                    1024.0,
                );
                let scene_read = self.scene.read().unwrap();

                let mut object_description_sratch_buffer = ifc
                    .allocate_scratch_buffer(
                        (self.acceleration_structure.addresses.len()
                            * std::mem::size_of::<asset::CMesh>())
                            as vk::DeviceSize,
                    )
                    .expect("Unable to allocate scratch buffer"); // lmao
                let mut material_descriptor_scratch_buffer = ifc
                    .allocate_scratch_buffer(
                        (scene_read.materials.len() * std::mem::size_of::<asset::CMaterial>())
                            as vk::DeviceSize,
                    )
                    .expect("Unable to allocate scratch buffer");

                let material_slice = scene_read
                    .materials
                    .iter()
                    .map(|handle| {
                        scene_read
                            .material_storage
                            .get_immutable(handle)
                            .unwrap()
                            .to_c_struct(scene_read.deref())
                    })
                    .collect::<Vec<asset::CMaterial>>();

                let mesh_slice = self.acceleration_structure.addresses.clone();

                object_description_sratch_buffer
                    .mapped_slice::<asset::CMesh>()
                    .unwrap()
                    .copy_from_slice(mesh_slice.as_slice());
                material_descriptor_scratch_buffer
                    .mapped_slice::<asset::CMaterial>()
                    .unwrap()
                    .copy_from_slice(material_slice.as_slice());
                println!(
                    "Material address: {}",
                    material_descriptor_scratch_buffer.address()
                );
                let images: Vec<phobos::ImageView> = scene_read
                    .images
                    .iter()
                    .map(|x| {
                        scene_read
                            .image_storage
                            .get_immutable(x)
                            .unwrap()
                            .whole_view(vk::ImageAspectFlags::COLOR)
                            .unwrap()
                    })
                    .collect();
                let sampler = scene_read
                    .sampler_storage
                    .get_immutable(scene_read.samplers.get(0).unwrap())
                    .unwrap();
                //println!("[Frame]: {}", images.len());
                cmd.bind_ray_tracing_pipeline("rt")?
                    .bind_storage_buffer(1, 1, &object_description_sratch_buffer)
                    .expect("Failed to bind storage buffer")
                    .bind_storage_buffer(1, 2, &material_descriptor_scratch_buffer)
                    .expect("Failed to bind storage buffer")
                    .bind_sampled_image_array(1, 3, images.as_slice(), sampler.as_ref())
                    .expect("Failed to bind image array")
                    .push_constant(vk::ShaderStageFlags::RAYGEN_KHR, 0, &view)
                    .push_constant(vk::ShaderStageFlags::RAYGEN_KHR, 64, &projection)
                    .bind_acceleration_structure(
                        0,
                        0,
                        self.acceleration_structure
                            .tlas
                            .resources
                            .acceleration_structures
                            .get(0)
                            .unwrap(),
                    )?
                    .resolve_and_bind_storage_image(0, 1, &rt_image, bindings)?
                    .trace_rays(self.render_extent.width, self.render_extent.height, 1)
            })
            .build();

        let render_pass = phobos::PassBuilder::render("copy")
            .clear_color_attachment(&swap, phobos::ClearColor::Float([0.0, 0.0, 0.0, 0.0]))?
            .sample_image(
                rt_pass.output(&rt_image).unwrap(),
                phobos::PipelineStage::FRAGMENT_SHADER,
            )
            .execute_fn(|cmd, ifc, bindings, _| {
                let vertices: Vec<f32> = vec![
                    -1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0,
                    1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
                ];
                let mut vtx_buffer = ifc.allocate_scratch_buffer(
                    (vertices.len() * std::mem::size_of::<f32>()) as vk::DeviceSize,
                )?;
                let slice = vtx_buffer.mapped_slice::<f32>()?;
                slice.copy_from_slice(vertices.as_slice());
                cmd.full_viewport_scissor()
                    .bind_graphics_pipeline("sample")?
                    .bind_vertex_buffer(0, &vtx_buffer)
                    .resolve_and_bind_sampled_image(0, 0, &rt_image, &self.sampler, bindings)?
                    .draw(6, 1, 0, 0)
            })
            .build();

        let present = phobos::PassBuilder::present("present", render_pass.output(&swap).unwrap());
        let mut graph = phobos::PassGraph::new()
            .add_pass(rt_pass)?
            .add_pass(render_pass)?
            .add_pass(present)?
            .build()?;

        let mut bindings = phobos::PhysicalResourceBindings::new();
        bindings.bind_image("swapchain", &ifc.swapchain_image);
        bindings.bind_image("rt_out", &self.color.view);
        let cmd = ctx.execution_manager.on_domain::<All>()?;
        let cmd = graph.record(cmd, &bindings, &mut pool, None, &mut ())?;
        let cmd = cmd.finish()?;
        let mut batch = ctx.execution_manager.start_submit_batch()?;
        batch.submit_for_present(cmd, ifc, pool)?;
        let now = std::time::Instant::now();
        self.delta_time = (now - self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        Ok(batch)
    }

    fn handle_event(&mut self, winit_event: &Event<()>) -> Result<()> {
        use winit::event::ElementState;
        use winit::event::VirtualKeyCode;
        let camera_speed: f32 = self.camera.speed;
        let delta_time = self.delta_time;
        let camera_front = self.camera.front;
        let camera_up = self.camera.up;
        let right = self.camera.right;

        let camera_sensitivity: f32 = 16.0f32;

        #[allow(clippy::single_match)]
        #[allow(clippy::collapsible_match)]
        match winit_event {
            Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::Resized(new_size) => {
                    // Handle resizing
                    let mut color = make_attachments(
                        self.ctx.clone(),
                        vk::Extent2D {
                            width: new_size.width,
                            height: new_size.height,
                        },
                    )
                    .unwrap();

                    self.render_extent = vk::Extent2D {
                        width: new_size.width,
                        height: new_size.height,
                    };
                    std::mem::swap(&mut self.color, &mut color);
                    self.deferred_delete.push(color);
                }
                winit::event::WindowEvent::KeyboardInput { input, .. } => match input {
                    winit::event::KeyboardInput {
                        virtual_keycode,
                        state,
                        ..
                    } => match (virtual_keycode, state) {
                        (Some(VirtualKeyCode::W), ElementState::Pressed) => {
                            self.camera.position += camera_front * camera_speed * delta_time;
                        }
                        (Some(VirtualKeyCode::S), ElementState::Pressed) => {
                            self.camera.position -= camera_front * camera_speed * delta_time;
                        }
                        (Some(VirtualKeyCode::A), ElementState::Pressed) => {
                            self.camera.position -= right * camera_speed * delta_time;
                        }
                        (Some(VirtualKeyCode::D), ElementState::Pressed) => {
                            self.camera.position += right * camera_speed * delta_time;
                        }
                        (Some(VirtualKeyCode::Q), ElementState::Pressed) => {
                            self.camera.position += camera_up * camera_speed * delta_time;
                        }
                        (Some(VirtualKeyCode::E), ElementState::Pressed) => {
                            self.camera.position -= camera_up * camera_speed * delta_time;
                        }
                        _ => {}
                    },
                    _ => {}
                },
                winit::event::WindowEvent::MouseWheel { delta, .. } => match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => {
                        self.camera.speed += (self.camera.speed.abs()) * y * delta_time;
                        self.camera.speed = self.camera.speed.clamp(0.0, f32::MAX);
                        println!("Camera speed: {}", self.camera.speed);
                    }
                    _ => {}
                },
                winit::event::WindowEvent::CursorMoved { position, .. } => {
                    if self.left_mouse_button_down {
                        let current_position: glam::Vec2 =
                            glam::Vec2::new(position.x as f32, position.y as f32);
                        let last_position: glam::Vec2 =
                            self.last_camera_pos.unwrap_or(current_position);
                        let offset_position: glam::Vec2 =
                            (current_position - last_position) * delta_time * camera_sensitivity;
                        self.camera.yaw -= offset_position.x;
                        self.camera.pitch += offset_position.y;
                        self.camera.pitch = self.camera.pitch.clamp(-89.0f32, 89.0f32);
                        self.last_camera_pos = Some(current_position);
                    }
                }
                winit::event::WindowEvent::CursorLeft { .. } => {
                    self.last_camera_pos = None;
                }
                winit::event::WindowEvent::MouseInput { button, state, .. } => {
                    match (button, state) {
                        (winit::event::MouseButton::Left, ElementState::Released) => {
                            self.last_camera_pos = None;
                            self.left_mouse_button_down = false;
                        }
                        (winit::event::MouseButton::Left, ElementState::Pressed) => {
                            self.last_camera_pos = None;
                            self.left_mouse_button_down = true;
                        }
                        _ => {}
                    }
                }
                _ => {}
            },
            _ => {}
        };
        self.camera.update_camera();
        Ok(())
    }
}

fn make_attachments(mut ctx: app::Context, render_width: vk::Extent2D) -> Result<Attachment> {
    // Make a color attachment
    Attachment::new(
        &mut ctx,
        vk::Format::R32G32B32A32_SFLOAT,
        render_width,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
    )
}

fn main() -> Result<()> {
    let window = app::WindowContext::with_size("DARE", 800.0, 600.0)?;
    app::Runner::new("DARE", Some(&window), |settings| {
        settings.raytracing(true).build()
    })?
    .run::<Raytracing>(Some(window))
}
