use anyhow::Result;
use phobos::vk;
use phobos::vk::Handle;
use phobos::{GraphicsCmdBuffer, IncompleteCmdBuffer, RecordGraphToCommandBuffer};
use std::sync::{Arc, RwLock};

mod app;
mod asset;
mod graphics;
mod spirv;
mod utils;

/// Resources used by the basic renderer
struct Resources {
    pub offscreen: phobos::Image,
    pub offscreen_view: phobos::ImageView,
    pub sampler: phobos::Sampler,
    pub vertex_buffer: phobos::Buffer,
}

/// The basic renderer
struct Basic {
    resources: Resources,
    scene: Arc<RwLock<asset::Scene>>,
    blas: Arc<graphics::acceleration_structure::SceneAccelerationStructure>,
}

impl app::App for Basic {
    fn new(mut ctx: app::Context) -> Result<Self>
    where
        Self: Sized,
    {
        let loader = asset::gltf_asset_loader::GltfAssetLoader::new();
        let scene = loader.load_asset_from_file(
            std::path::Path::new(
                "C:/Users/Danny/Documents/glTF-Sample-Models/2.0/Suzanne/glTF/Suzanne.gltf",
            ),
            &mut ctx,
        );
        let blas = graphics::acceleration_structure::convert_scene_to_blas(&mut ctx, &scene);

        // Load shader
        let vtx_code = spirv::load_spirv_file(std::path::Path::new("shaders/vert.spv"));
        let frag_code = spirv::load_spirv_file(std::path::Path::new("shaders/frag.spv"));

        let vertex =
            phobos::ShaderCreateInfo::from_spirv(phobos::vk::ShaderStageFlags::VERTEX, vtx_code);
        let fragment =
            phobos::ShaderCreateInfo::from_spirv(phobos::vk::ShaderStageFlags::FRAGMENT, frag_code);

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

        let frag_code = spirv::load_spirv_file(std::path::Path::new("shaders/blue.spv"));
        let fragment =
            phobos::ShaderCreateInfo::from_spirv(phobos::vk::ShaderStageFlags::FRAGMENT, frag_code);

        let pci = phobos::PipelineBuilder::new("offscreen".to_string())
            .vertex_input(0, phobos::vk::VertexInputRate::VERTEX)
            .vertex_attribute(0, 0, phobos::vk::Format::R32G32_SFLOAT)?
            .vertex_attribute(0, 1, phobos::vk::Format::R32G32_SFLOAT)?
            .dynamic_states(&[
                phobos::vk::DynamicState::VIEWPORT,
                phobos::vk::DynamicState::SCISSOR,
            ])
            .blend_attachment_none()
            .cull_mask(phobos::vk::CullModeFlags::NONE)
            .attach_shader(vertex)
            .attach_shader(fragment)
            .build();
        ctx.resource_pool.pipelines.create_named_pipeline(pci)?;

        {
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
        }

        // Define some resources we will use for rendering
        let image = phobos::Image::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            800,
            600,
            phobos::vk::ImageUsageFlags::COLOR_ATTACHMENT | phobos::vk::ImageUsageFlags::SAMPLED,
            phobos::vk::Format::R8G8B8A8_SRGB,
            phobos::vk::SampleCountFlags::TYPE_1,
        )?;
        let data: Vec<f32> = vec![
            -1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ];

        let resources = Resources {
            offscreen_view: image.view(phobos::vk::ImageAspectFlags::COLOR)?,
            offscreen: image,
            sampler: phobos::Sampler::default(ctx.device.clone())?,
            vertex_buffer: app::staged_buffer_upload(
                ctx.clone(),
                data.as_slice(),
                phobos::vk::BufferUsageFlags::VERTEX_BUFFER,
            )?,
        };

        // Debug naming
        let buffer_name = std::ffi::CString::new("Test_naming").unwrap();
        let name_info = phobos::prelude::vk::DebugUtilsObjectNameInfoEXT::builder()
            .object_type(phobos::prelude::vk::ObjectType::BUFFER)
            .object_handle(unsafe { resources.vertex_buffer.handle().as_raw() })
            .object_name(&buffer_name)
            .build();

        println!("{} is buffer!", unsafe {
            resources.vertex_buffer.handle().as_raw()
        });

        unsafe {
            ctx.debug_utils
                .set_debug_utils_object_name(ctx.device.handle().handle(), &name_info)
                .expect("Failed to set object name!");
        };

        Ok(Self {
            resources,
            scene: Arc::new(RwLock::new(scene)),
            blas: Arc::new(blas),
        })
    }

    fn frame(
        &mut self,
        ctx: app::Context,
        ifc: phobos::InFlightContext,
    ) -> Result<phobos::sync::submit_batch::SubmitBatch<phobos::domain::All>> {
        // Define a virtual resource pointing to the swapchain
        let swap_resource = phobos::image!("swapchain");
        let offscreen = phobos::image!("offscreen");

        let vertices: Vec<f32> = vec![
            -1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ];

        // Define a render graph with one pass that clears the swapchain image
        let graph = phobos::PassGraph::new();

        let mut pool = phobos::pool::LocalPool::new(ctx.resource_pool.clone())?;

        // Render pass that renders to an offscreen attachment
        let offscreen_pass = phobos::PassBuilder::render("offscreen")
            .color([1.0, 0.0, 0.0, 1.0])
            .clear_color_attachment(&offscreen, phobos::ClearColor::Float([0.0, 0.0, 0.0, 0.0]))?
            .execute_fn(|mut cmd, ifc, _bindings, _| {
                // Our pass will render a fullscreen quad that 'clears' the screen, just so we can test pipeline creation
                let mut buffer = ifc.allocate_scratch_vbo(
                    (vertices.len() * std::mem::size_of::<f32>()) as phobos::vk::DeviceSize,
                )?;
                let slice = buffer.mapped_slice::<f32>()?;
                slice.copy_from_slice(vertices.as_slice());
                cmd = cmd
                    .bind_vertex_buffer(0, &buffer)
                    .bind_graphics_pipeline("offscreen")?
                    .full_viewport_scissor()
                    .draw(6, 1, 0, 0)?;
                Ok(cmd)
            })
            .build();

        // Render pass that samples the offscreen attachment, and possibly does some postprocessing to it
        let sample_pass = phobos::PassBuilder::render(String::from("sample"))
            .color([0.0, 1.0, 0.0, 1.0])
            .clear_color_attachment(
                &swap_resource,
                phobos::ClearColor::Float([0.0, 0.0, 0.0, 0.0]),
            )?
            .sample_image(
                offscreen_pass.output(&offscreen).unwrap(),
                phobos::PipelineStage::FRAGMENT_SHADER,
            )
            .execute_fn(|cmd, _ifc, bindings, _| {
                cmd.full_viewport_scissor()
                    .bind_graphics_pipeline("sample")?
                    .resolve_and_bind_sampled_image(
                        0,
                        0,
                        &offscreen,
                        &self.resources.sampler,
                        bindings,
                    )?
                    .draw(6, 1, 0, 0)
            })
            .build();
        // Add another pass to handle presentation to the screen
        let present_pass = phobos::PassBuilder::present(
            "present",
            // This pass uses the output from the clear pass on the swap resource as its input
            sample_pass.output(&swap_resource).unwrap(),
        );
        let mut graph = graph
            .add_pass(offscreen_pass)?
            .add_pass(sample_pass)?
            .add_pass(present_pass)?
            // Build the graph, now we can bind physical resources and use it.
            .build()?;

        let mut bindings = phobos::PhysicalResourceBindings::new();
        bindings.bind_image("swapchain", &ifc.swapchain_image);
        bindings.bind_image("offscreen", &self.resources.offscreen_view);
        // create a command buffer capable of executing graphics commands
        let cmd = ctx
            .execution_manager
            .on_domain::<phobos::domain::All>()
            .unwrap();
        // record render graph to this command buffer
        let cmd = graph
            .record(cmd, &bindings, &mut pool, None, &mut ())?
            .finish()?;
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
    .run::<Basic>(Some(window))
}
