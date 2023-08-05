use anyhow::Result;
use phobos::prelude::vk;
use phobos::prelude::*;

#[derive(Debug)]
pub struct WindowContext {
    pub event_loop: winit::event_loop::EventLoop<()>,
    pub window: winit::window::Window,
}

impl WindowContext {
    pub fn new(title: impl Into<String>) -> Result<Self> {
        Self::with_size(title, 800.0, 600.0)
    }

    pub fn with_size(title: impl Into<String>, width: f32, height: f32) -> Result<Self> {
        let event_loop = winit::event_loop::EventLoopBuilder::new().build();
        let window = winit::window::WindowBuilder::new()
            .with_title(title)
            .with_inner_size(winit::dpi::LogicalSize::new(width, height))
            .build(&event_loop)?;
        Ok(Self { event_loop, window })
    }
}

pub struct VulkanContext {
    pub frame: Option<phobos::FrameManager>,
    pub pool: phobos::pool::ResourcePool,
    pub execution_manager: phobos::ExecutionManager,
    pub allocator: phobos::DefaultAllocator,
    pub device: phobos::Device,
    pub physical_device: phobos::PhysicalDevice,
    pub surface: Option<phobos::Surface>,
    pub debug_messenger: phobos::DebugMessenger,
    pub instance: phobos::Instance,
}

#[derive(Clone)]
pub struct Context {
    pub device: phobos::Device,
    pub execution_manager: phobos::ExecutionManager,
    pub allocator: phobos::DefaultAllocator,
    pub resource_pool: phobos::pool::ResourcePool,
    pub debug_utils: ash::extensions::ext::DebugUtils,
}

pub trait App {
    fn new(ctx: Context) -> Result<Self>
    where
        Self: Sized;

    fn frame(
        &mut self,
        _ctx: Context,
        _ifc: phobos::InFlightContext,
    ) -> Result<phobos::sync::submit_batch::SubmitBatch<phobos::domain::All>> {
        anyhow::bail!("frame() not implemented for non-headless example app");
    }

    fn run(&mut self, _ctx: Context) -> Result<()> {
        Ok(())
    }

    fn handle_event(&mut self, _event: &winit::event::Event<()>) -> Result<()> {
        Ok(())
    }
}

pub struct Runner {
    vk: VulkanContext,
}

impl Runner {
    pub fn new(
        name: impl Into<String>,
        window: Option<&WindowContext>,
        make_settings: impl Fn(
            phobos::AppBuilder<winit::window::Window>,
        ) -> phobos::AppSettings<winit::window::Window>,
    ) -> Result<Self> {
        std::env::set_var("RUST_LOG", "trace");
        pretty_env_logger::init();
        let mut settings = phobos::AppBuilder::new()
            .version((1, 0, 0))
            .name(name)
            .validation(true)
            .present_mode(phobos::vk::PresentModeKHR::MAILBOX)
            .scratch_chunk_size(1 * 1024u64)
            .gpu(phobos::GPURequirements {
                dedicated: false,
                min_video_memory: 4 * 1024 * 1024 * 1024, // 1 GiB
                min_dedicated_video_memory: 4 * 1024 * 1024 * 1024,
                queues: vec![
                    phobos::QueueRequest {
                        dedicated: false,
                        queue_type: phobos::QueueType::Graphics,
                    },
                    phobos::QueueRequest {
                        dedicated: true,
                        queue_type: phobos::QueueType::Transfer,
                    },
                    phobos::QueueRequest {
                        dedicated: true,
                        queue_type: phobos::QueueType::Compute,
                    },
                ],
                features: vk::PhysicalDeviceFeatures::builder()
                    .shader_int64(true)
                    .build(),
                device_extensions: vec![String::from("VK_EXT_scalar_block_layout")],
                ..Default::default()
            });

        match window {
            None => {}
            Some(window) => {
                settings = settings.window(&window.window);
            }
        };
        let settings = make_settings(settings);

        let (instance, physical_device, surface, device, allocator, pool,
		exec, frame, Some(debug_messenger)) = phobos::initialize(&settings,
		window.is_none())? else {
			panic!("Asked for debug messenger but didn't get one")
		};

        let vk = VulkanContext {
            frame,
            pool,
            execution_manager: exec,
            allocator,
            device,
            physical_device,
            surface,
            debug_messenger,
            instance,
        };

        Ok(Self { vk })
    }

    fn run_headless<E: App + 'static>(self, mut app: E) -> ! {
        app.run(self.make_context()).unwrap();
        self.vk.device.wait_idle().unwrap();
        drop(app);
        std::process::exit(0);
    }

    fn make_context(&self) -> Context {
        Context {
            device: self.vk.device.clone(),
            execution_manager: self.vk.execution_manager.clone(),
            allocator: self.vk.allocator.clone(),
            resource_pool: self.vk.pool.clone(),
            debug_utils: self.vk.debug_messenger.clone(),
        }
    }

    fn frame<E: App + 'static>(
        &mut self,
        app: &mut E,
        window: &winit::window::Window,
    ) -> Result<()> {
        let ctx = self.make_context();
        let frame = self.vk.frame.as_mut().unwrap();
        let surface = self.vk.surface.as_ref().unwrap();
        futures::executor::block_on(frame.new_frame(
            self.vk.execution_manager.clone(),
            window,
            surface,
            |ifc| app.frame(ctx, ifc),
        ))?;
        Ok(())
    }

    fn run_windowed<E: App + 'static>(mut self, app: E, window: WindowContext) -> ! {
        let event_loop = window.event_loop;
        let window = window.window;
        let mut app = Some(app);
        event_loop.run(move |event, _, control_flow| {
            // Do not render a frame if Exit control flow is specified, to avoid
            // sync issues.
            if let winit::event_loop::ControlFlow::ExitWithCode(_) = *control_flow {
                self.vk.device.wait_idle().unwrap();
                return;
            }
            *control_flow = winit::event_loop::ControlFlow::Poll;

            match &mut app {
                None => {}
                Some(app) => {
                    app.handle_event(&event).unwrap();
                }
            }

            // Note that we want to handle events after processing our current frame, so that
            // requesting an exit doesn't attempt to render another frame, which causes
            // sync issues.
            match event {
                winit::event::Event::WindowEvent {
                    event: winit::event::WindowEvent::CloseRequested,
                    window_id,
                } if window_id == window.id() => {
                    *control_flow = winit::event_loop::ControlFlow::Exit;
                    self.vk.device.wait_idle().unwrap();
                    let app = app.take();
                    match app {
                        None => {}
                        Some(app) => {
                            drop(app);
                        }
                    }
                }
                winit::event::Event::MainEventsCleared => {
                    window.request_redraw();
                }
                winit::event::Event::RedrawRequested(_) => match app.as_mut() {
                    None => {}
                    Some(app) => {
                        self.frame(app, &window).unwrap();
                        self.vk.pool.next_frame();
                    }
                },
                _ => (),
            }
        })
    }

    pub fn run<E: App + 'static>(self, window: Option<WindowContext>) -> ! {
        let app = E::new(self.make_context()).unwrap();
        match window {
            None => self.run_headless(app),
            Some(window) => self.run_windowed(app, window),
        }
    }
}
