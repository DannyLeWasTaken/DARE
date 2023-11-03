#[cfg(feature = "shaderc")]
extern crate shaderc;

use std::collections::HashSet;
#[allow(unused_imports)]
use std::fs;
#[allow(unused_imports)]
use std::io::{Read, Write};
#[allow(unused_imports)]
use std::path::Path;
use std::sync::{Arc, Mutex};

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

    let include_paths: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new())); // Contains all the source data to include.
    // This will terminate and throw if it detects a recursive include

    options.set_include_callback({
        move |requested_source, directive_type, requesting_source, _include_depth| {
            // Construct the path of the included file based on the type of include
            let shader_path = Path::new("./shaders");
            let mut include_paths = include_paths.lock().unwrap();
            let included_path = shader_path.join(requested_source);
            let included_path_str = included_path.to_string_lossy().into_owned();

            // Check if it hasn't already been included
            if include_paths.contains(&included_path_str) {
                println!("Already found! {}", included_path_str.clone());
                Ok(shaderc::ResolvedInclude {
                    resolved_name: shader_path.to_string_lossy().into_owned(),
                    content: "".to_string()
                })
            } else {
                // Read the source code of the included file
                println!("Including {}", included_path_str.clone());
                match fs::read_to_string(&included_path) {
                    Ok(source) =>{
                        include_paths.insert(included_path_str.clone());
                        Ok(shaderc::ResolvedInclude {
                            resolved_name: included_path_str,
                            content: source,
                        })
                    },
                    Err(err) => Err(format!("Error reading file: {}", err)),
                }
            }
        }
    });

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
