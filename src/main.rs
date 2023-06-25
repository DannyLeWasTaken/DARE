use winit::window::WindowBuilder;
use anyhow::Result;
use phobos::image;
use phobos::prelude::*;

mod app;
mod spirv;

struct Resources {
    pub offscreen: phobos::Image,
    pub offscreen_view: phobos::ImageView,
    pub sampler: phobos::Sampler,
    pub vertex_buffer: phobos::Buffer,
}

struct Basic {
    resources: Resources,
}

impl app::App for Basic {
    fn new(mut ctx: app::Context) -> Result<Self>
    where
        Self: Sized, {
        // Load shader
        let vtx_code = spirv::load_spirv_file(std::path::Path::new("shaders/vert.spv"));
        let frag_code = spirv::load_spirv_file(std::path::Path::new("shaders/frag.spv"));

        let vertex = phobos::ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::VERTEX, vtx_code);
        let fragment = phobos::ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::FRAGMENT, frag_code);

        // Now we can start using the pipeline builder to create our full pipeline.
        let pci = PipelineBuilder::new("sample".to_string())
            .vertex_input(0, vk::VertexInputRate::VERTEX)
            .vertex_attribute(0, 0, vk::Format::R32G32_SFLOAT)?
            .vertex_attribute(0, 1, vk::Format::R32G32_SFLOAT)?
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .blend_attachment_none()
            .cull_mask(vk::CullModeFlags::NONE)
            .attach_shader(vertex.clone())
            .attach_shader(fragment)
            .build();

        // Store the pipeline in the pipeline cache
        ctx.resource_pool.pipelines.create_named_pipeline(pci)?;

        let frag_code = spirv::load_spirv_file(std::path::Path::new("shaders/blue.spv"));
        let fragment = phobos::ShaderCreateInfo::from_spirv(vk::ShaderStageFlags::FRAGMENT, frag_code);

        let pci = PipelineBuilder::new("offscreen".to_string())
            .vertex_input(0, vk::VertexInputRate::VERTEX)
            .vertex_attribute(0, 0, vk::Format::R32G32_SFLOAT)?
            .vertex_attribute(0, 1, vk::Format::R32G32_SFLOAT)?
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .blend_attachment_none()
            .cull_mask(vk::CullModeFlags::NONE)
            .attach_shader(vertex)
            .attach_shader(fragment)
            .build();
        ctx.resource_pool.pipelines.create_named_pipeline(pci)?;

        // Define some resources we will use for rendering
        let image = phobos::Image::new(
            ctx.device.clone(),
            &mut ctx.allocator,
            800,
            600,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::Format::R8G8B8A8_SRGB,
            vk::SampleCountFlags::TYPE_1,
        )?;
        let data: Vec<f32> = vec![
            -1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ];

        let resources = Resources {
            offscreen_view: image.view(vk::ImageAspectFlags::COLOR)?,
            offscreen: image,
            sampler: phobos::Sampler::default(ctx.device.clone())?,
            vertex_buffer: app::staged_buffer_upload(
                ctx.clone(),
                data.as_slice(),
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?,
        };

        Ok(Self {
            resources,
        })
    }

    fn frame(&mut self, ctx: app::Context, ifc: InFlightContext) -> Result<phobos::sync::submit_batch::SubmitBatch<domain::All>> {
        // Define a virtual resource pointing to the swapchain
        let swap_resource = phobos::image!("swapchain");
        let offscreen = phobos::image!("offscreen");

        let vertices: Vec<f32> = vec![
            -1.0, 1.0, 0.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ];

        // Define a render graph with one pass that clears the swapchain image
        let graph = PassGraph::new();

        let mut pool = phobos::pool::LocalPool::new(ctx.resource_pool.clone())?;

        // Render pass that renders to an offscreen attachment
        let offscreen_pass = PassBuilder::render("offscreen")
            .color([1.0, 0.0, 0.0, 1.0])
            .clear_color_attachment(&offscreen, ClearColor::Float([0.0, 0.0, 0.0, 0.0]))?
            .execute_fn(|mut cmd, ifc, _bindings, _| {
                // Our pass will render a fullscreen quad that 'clears' the screen, just so we can test pipeline creation
                let mut buffer = ifc.allocate_scratch_vbo(
                    (vertices.len() * std::mem::size_of::<f32>()) as vk::DeviceSize,
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
        let sample_pass = PassBuilder::render(String::from("sample"))
            .color([0.0, 1.0, 0.0, 1.0])
            .clear_color_attachment(&swap_resource, ClearColor::Float([0.0, 0.0, 0.0, 0.0]))?
            .sample_image(
                offscreen_pass.output(&offscreen).unwrap(),
                PipelineStage::FRAGMENT_SHADER,
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
        let present_pass = PassBuilder::present(
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

        let mut bindings = PhysicalResourceBindings::new();
        bindings.bind_image("swapchain", &ifc.swapchain_image);
        bindings.bind_image("offscreen", &self.resources.offscreen_view);
        // create a command buffer capable of executing graphics commands
        let cmd = ctx.execution_manager.on_domain::<phobos::domain::All>().unwrap();
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
    let window = app::WindowContext::new(
        "DARE"
    )?;

    app::Runner::new("DARE",
                     Some(&window),
        |s| s.build()
    )?.run::<Basic>(Some(window))
}