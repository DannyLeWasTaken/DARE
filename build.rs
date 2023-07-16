#[cfg(feature = "shaderc")]
extern crate shaderc;

#[allow(unused_imports)]
use std::fs;
#[allow(unused_imports)]
use std::io::{Read, Write};
#[allow(unused_imports)]
use std::path::Path;

#[cfg(feature = "shaderc")]
use shaderc::{CompileOptions, EnvVersion, TargetEnv};

#[cfg(feature = "shaderc")]
fn load_file(path: &Path) -> String {
    let mut out = String::new();
    fs::File::open(path)
        .unwrap()
        .read_to_string(&mut out)
        .unwrap();
    out
}

#[cfg(feature = "shaderc")]
fn save_file(path: &Path, binary: &[u8]) {
    fs::File::create(path).unwrap().write_all(binary).unwrap();
}

#[cfg(feature = "shaderc")]
fn compile_shader(path: &Path, kind: shaderc::ShaderKind, output: &Path) {
    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = CompileOptions::new().unwrap();
    options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_2 as u32);
    // Handle includes
    options.set_include_callback(
        |requested_source, directive_type, requesting_source, _include_depth| {
            // Construct the path of the included file based on the type of include
            let shader_path = Path::new("./shaders");
            let included_path = shader_path.join(requested_source);
            // Read the source code of the included file
            match fs::read_to_string(&included_path) {
                Ok(source) => Ok(shaderc::ResolvedInclude {
                    resolved_name: included_path.to_string_lossy().into_owned(),
                    content: source,
                }),
                Err(err) => Err(format!("Error reading file: {}", err)),
            }
        },
    );

    let binary = compiler
        .compile_into_spirv(
            &load_file(path),
            kind,
            path.as_os_str().to_str().unwrap(),
            "main",
            Some(&options),
        )
        .expect("Ran into an error while compiling GLSL");
    save_file(output, binary.as_binary_u8());
}

#[cfg(feature = "shaderc")]
fn compile_shaders() {
    println!("cargo:rerun-if-changed=./shaders/vert.vert");
    println!("cargo:rerun-if-changed=./shaders/frag.frag");
    println!("cargo:rerun-if-changed=./shaders/blue.frag");
    compile_shader(
        Path::new("shaders/vert.vert"),
        shaderc::ShaderKind::Vertex,
        Path::new("./shaders/vert.spv"),
    );
    compile_shader(
        Path::new("shaders/frag.frag"),
        shaderc::ShaderKind::Fragment,
        Path::new("./shaders/frag.spv"),
    );
    compile_shader(
        Path::new("shaders/blue.frag"),
        shaderc::ShaderKind::Fragment,
        Path::new("./shaders/blue.spv"),
    );
}

fn main() {
    //#[cfg(feature = "shaderc")]
    //compile_shaders();

    #[cfg(feature = "shaderc")]
    {
        let paths = fs::read_dir("./shaders").unwrap();
        for path in paths {
            let path = path.unwrap().path();
            if !path.is_file() {
                continue;
            }
            if let Some(filename) = path.file_name().and_then(|s| s.to_str()) {
                // Ignore include-only files
                if filename.ends_with(".inc.glsl") {
                    continue;
                }
            }
            if let Some(extension) = path.extension().and_then(|s| s.to_str()) {
                // Skip include-only files
                println!("====");
                println!("{:?}", path.file_name().unwrap_or_default());
                println!("{:?}", extension);
                println!("====");
                if extension == "inc.glsl" || extension == "spv" {
                    continue;
                }

                // Print a rerun-if-changed for each .glsl file
                println!("carog:rerun-if-changed={}", path.display());

                // Determine shader kind based on file extension
                let shader_kind = match path.extension().unwrap().to_str().unwrap() {
                    "vert" => shaderc::ShaderKind::Vertex,
                    "frag" => shaderc::ShaderKind::Fragment,
                    "rgen" => shaderc::ShaderKind::RayGeneration,
                    "rchit" => shaderc::ShaderKind::ClosestHit,
                    "rmiss" => shaderc::ShaderKind::Miss,
                    _ => panic!("Unsupported shader kind: {:?}", path.file_name().unwrap()),
                };
                let output = path.with_extension("spv");
                println!("Building shader: {:?}", path.file_name().unwrap());
                compile_shader(&path, shader_kind, &output);
            }
        }
    }
}
