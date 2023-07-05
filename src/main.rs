use anyhow::Result;
use phobos::domain::All;
use phobos::vk;
use phobos::vk::Handle;
use phobos::{GraphicsCmdBuffer, IncompleteCmdBuffer, RecordGraphToCommandBuffer};
use std::sync::{Arc, RwLock};

mod app;
mod asset;
mod graphics;
mod spirv;
mod utils;

/// The basic renderer
struct Raytracing {
    scene: Arc<RwLock<asset::Scene>>,
    acceleration_structure: Arc<graphics::acceleration_structure::SceneAccelerationStructure>,
    attachment: phobos::Image,
    attachment_view: phobos::ImageView,
    sampler: phobos::Sampler,
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
                //gltf_sample_name("Suzanne").as_str(),
                gltf_sample_name("ToyCar").as_str(),
                //"C:/Users/Danny/Documents/Assets/Junk Shop/Blender 2.gltf",
                //"C:/Users/Danny/Documents/Assets/Classroom/classroom.gltf",
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

        let vertex =
            phobos::ShaderCreateInfo::from_spirv(phobos::vk::ShaderStageFlags::VERTEX, vertex);
        let fragment =
            phobos::ShaderCreateInfo::from_spirv(phobos::vk::ShaderStageFlags::FRAGMENT, fragment);

        // Now we can start using the pipeline builder to create our full pipeline.
        let pci = phobos::PipelineBuilder::new("sample".to_string())
            .vertex_input(0, phobos::vk::VertexInputRate::VERTEX)
            .vertex_attribute(0, 0, phobos::vk::Format::R32G32_SFLOAT)?
            .vertex_attribute(0, 1, phobos::vk::Format::R32G32_SFLOAT)?
            .dynamic_states(&[
                phobos::vk::DynamicState::VIEWPORT,
                phobos::vk::DynamicState::SCISSOR,
            ])
            .blend_attachment_none()
            .cull_mask(phobos::vk::CullModeFlags::NONE)
            .attach_shader(vertex.clone())
            .attach_shader(fragment)
            .build();

        // Store the pipeline in the pipeline cache
        ctx.resource_pool.pipelines.create_named_pipeline(pci)?;

        let attachment = phobos::Image::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            800,
            600,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
            vk::Format::R32G32B32A32_SFLOAT,
            vk::SampleCountFlags::TYPE_1,
        )?;
        let view = attachment.view(vk::ImageAspectFlags::COLOR)?;
        let sampler = phobos::Sampler::default(ctx.device.clone())?;

        Ok(Self {
            scene: Arc::new(RwLock::new(scene)),
            acceleration_structure: Arc::new(scene_acceleration_structure),
            attachment,
            attachment_view: view,
            sampler,
        })
    }

    fn frame(
        &mut self,
        ctx: app::Context,
        ifc: phobos::InFlightContext,
    ) -> Result<phobos::sync::submit_batch::SubmitBatch<phobos::domain::All>> {
        let swap = phobos::image!("swapchain");
        let rt_image = phobos::image!("rt_out");

        let mut pool = phobos::pool::LocalPool::new(ctx.resource_pool.clone())?;

        let rt_pass = phobos::PassBuilder::new("raytrace")
            .write_storage_image(&rt_image, phobos::PipelineStage::RAY_TRACING_SHADER_KHR)
            .execute_fn(|cmd, ifc, bindings, _| {
                let view = glam::Mat4::look_at_rh(
                    glam::Vec3::new(0.0, 0.0, -0.0),
                    glam::Vec3::new(0.0, 0.0, 1.0),
                    glam::Vec3::new(0.0, 1.0, 0.0),
                );
                let projection =
                    glam::Mat4::perspective_rh(90.0_f32.to_radians(), 800.0 / 600.0, 0.001, 100.0);
                cmd.bind_ray_tracing_pipeline("rt")?
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
                    .trace_rays(800, 600, 1)
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
                let mut vtx_buffer = ifc.allocate_scratch_vbo(
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
        bindings.bind_image("rt_out", &self.attachment_view);
        let cmd = ctx.execution_manager.on_domain::<All>()?;
        let cmd = graph.record(cmd, &bindings, &mut pool, None, &mut ())?;
        let cmd = cmd.finish()?;
        let mut batch = ctx.execution_manager.start_submit_batch()?;
        batch.submit_for_present(cmd, ifc, pool)?;
        Ok(batch)
    }
}

fn main() -> Result<()> {
    let window = app::WindowContext::new("DARE")?;
    app::Runner::new("DARE", Some(&window), |settings| {
        settings.raytracing(true).build()
    })?
    .run::<Raytracing>(Some(window))
}
